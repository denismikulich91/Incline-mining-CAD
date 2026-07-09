//! Reusable background-compute job plumbing.
//!
//! Generalizes the off-thread load/save pattern (`triangulation/session.rs`,
//! `commands/file.rs`) into a single queue so heavy, discrete operations
//! (include, cut, create) can run off the UI thread without hand-rolling a
//! `pending_*` vec + poll fn per operation.
//!
//! The compute closure and its owned inputs live on a worker thread; the
//! apply closure stays App-side and runs on the UI thread when the result
//! arrives. Type information is captured inside each job's `poll` closure, so
//! one queue can hold jobs with heterogeneous result types.
//!
//! Compute runs on a small fixed worker pool (rather than one thread per
//! job), so rapidly re-triggering a heavy operation queues work instead of
//! stacking OS threads. Each job carries a [`CancelFlag`]; `cancel_jobs` sets
//! it so a cancelled compute is skipped if it hasn't started and can bail out
//! early if it checks the flag mid-computation.

use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use super::App;
use crate::model::triangulation::TriangulationId;

/// Identifies the source a job derives from, so stale in-flight jobs can be
/// cancelled when their source changes or is removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JobKey {
    Triangulation(TriangulationId),
    /// A job not tied to any tracked source (always applied).
    #[allow(dead_code)]
    Anonymous,
}

/// Shared cancellation signal for one background job. Compute closures should
/// poll [`CancelFlag::is_cancelled`] at convenient points in long loops and
/// return early (any error is fine — the result is discarded anyway).
#[derive(Clone, Default)]
pub(crate) struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub(crate) fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// A heavy operation running on a background thread.
pub(crate) struct BackgroundJob<'a> {
    pub(crate) label: String,
    #[allow(dead_code)]
    pub(crate) key: JobKey,
    cancel: CancelFlag,
    /// Polls the worker channel. Returns `true` once the job has settled
    /// (result applied, or the worker vanished) and should be dropped.
    poll: Box<dyn FnMut(&mut App<'a>) -> bool + 'a>,
}

type PoolTask = Box<dyn FnOnce() + Send>;

/// Fixed-size pool shared by all background jobs; bounds how many heavy
/// computes run at once no matter how fast jobs are spawned.
fn job_pool() -> &'static mpsc::Sender<PoolTask> {
    static POOL: OnceLock<mpsc::Sender<PoolTask>> = OnceLock::new();
    POOL.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<PoolTask>();
        let rx = Arc::new(Mutex::new(rx));
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .clamp(1, 4);
        for index in 0..workers {
            let rx = Arc::clone(&rx);
            std::thread::Builder::new()
                .name(format!("job-worker-{index}"))
                .spawn(move || {
                    loop {
                        let task = rx.lock().expect("job pool poisoned").recv();
                        match task {
                            Ok(task) => task(),
                            Err(_) => break,
                        }
                    }
                })
                .expect("failed to spawn job worker thread");
        }
        tx
    })
}

impl<'a> App<'a> {
    /// Run `compute` on the background worker pool; when it finishes, `apply`
    /// runs on the UI thread with the result. Increments the shared progress
    /// counter (progress cursor + status bar) for the job's lifetime.
    ///
    /// `compute` must capture only owned data / `Arc`s — never `&self`, and
    /// should poll the passed [`CancelFlag`] in long loops. `apply` runs later
    /// with `&mut App`, so it must re-resolve anything by stable id: sources
    /// may have been edited or removed while the job ran.
    pub(crate) fn spawn_job<T, C, A>(
        &mut self,
        label: impl Into<String>,
        key: JobKey,
        compute: C,
        apply: A,
    ) where
        T: Send + 'static,
        C: FnOnce(&CancelFlag) -> anyhow::Result<T> + Send + 'static,
        A: FnOnce(&mut App<'a>, anyhow::Result<T>) + 'a,
    {
        let label = label.into();
        self.begin_topology_load();

        let (tx, rx) = mpsc::channel();
        let window = self.window.clone();
        let cancel = CancelFlag::default();
        let worker_cancel = cancel.clone();
        let task: PoolTask = Box::new(move || {
            // A job cancelled while still queued never starts computing;
            // dropping `tx` settles its poll via Disconnected.
            if !worker_cancel.is_cancelled() {
                let _ = tx.send(compute(&worker_cancel));
            }
            if let Some(window) = window {
                window.request_redraw();
            }
        });
        let _ = job_pool().send(task);

        let mut apply = Some(apply);
        let poll = Box::new(move |app: &mut App<'a>| -> bool {
            match rx.try_recv() {
                Ok(result) => {
                    if let Some(apply) = apply.take() {
                        apply(app, result);
                    }
                    true
                }
                Err(mpsc::TryRecvError::Empty) => false,
                // Worker dropped the sender without a value (panicked or was
                // cancelled before starting): settle so the progress
                // counter/cursor don't get stuck.
                Err(mpsc::TryRecvError::Disconnected) => true,
            }
        });

        self.pending_jobs.push(BackgroundJob {
            label,
            key,
            cancel,
            poll,
        });
    }

    /// Drain finished background jobs, running their apply closures on the UI
    /// thread. Call once per frame alongside the other polls.
    pub(crate) fn poll_jobs(&mut self) {
        if self.pending_jobs.is_empty() {
            return;
        }

        let mut still_pending = Vec::with_capacity(self.pending_jobs.len());
        for mut job in std::mem::take(&mut self.pending_jobs) {
            if (job.poll)(self) {
                // Settled: balance the begin_topology_load() from spawn_job.
                self.finish_background_save();
                self.redraw_requested = true;
            } else {
                still_pending.push(job);
            }
        }
        // An apply closure may itself spawn a follow-up job; keep those
        // (currently in self.pending_jobs) after the ones still running.
        still_pending.append(&mut self.pending_jobs);
        self.pending_jobs = still_pending;

        // Reflect the first in-flight job in the status bar, but never clobber
        // an active save message (poll_saves owns the bar while saving).
        if self.pending_saves.is_empty()
            && let Some(job) = self.pending_jobs.first()
        {
            self.editor.status_message = Some(crate::ui::state::StatusBarMessage {
                text: job.label.clone(),
                progress: None,
            });
        }
    }

    /// Cancel in-flight jobs whose key matches `pred`: their cancel flag is
    /// set (so queued computes never start and running ones can bail) and
    /// their results are discarded. Settles the progress counter for each so
    /// the progress cursor doesn't stick.
    #[allow(dead_code)]
    pub(crate) fn cancel_jobs(&mut self, pred: impl Fn(&JobKey) -> bool) {
        let before = self.pending_jobs.len();
        self.pending_jobs.retain(|job| {
            if pred(&job.key) {
                job.cancel.cancel();
                false
            } else {
                true
            }
        });
        let cancelled = before - self.pending_jobs.len();
        for _ in 0..cancelled {
            self.finish_background_save();
        }
    }
}
