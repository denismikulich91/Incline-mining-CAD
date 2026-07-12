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
//! stacking OS threads. Blocking file writes run on a separate small I/O pool
//! so a slow disk cannot starve compute jobs (or vice versa). Each job
//! carries a [`CancelFlag`] and a set of [`JobKey`] dependencies;
//! `cancel_jobs` sets the flag so a cancelled compute is skipped if it hasn't
//! started and can bail out early if it checks the flag mid-computation.
//!
//! A panicking task is converted into an error result and the worker loop
//! survives: repeated panics can never silently shrink the pool, and the
//! failed job is reported like any other error.

use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use super::App;
use crate::model::triangulation::TriangulationId;

/// Identifies the sources a job derives from, so stale in-flight jobs can be
/// cancelled when their source changes or is removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JobKey {
    Triangulation(TriangulationId),
    PointCloud(crate::model::point_cloud::PointCloudId),
    BlockModel(crate::model::block_model::BlockModelId),
    Project {
        runtime_id: u32,
        document_revision: u64,
    },
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
    pub(crate) ticket: crate::app::BackgroundTaskTicket,
    pub(crate) label: String,
    /// Every source this job depends on; cancelling any of them cancels the job.
    pub(crate) keys: Vec<JobKey>,
    cancel: CancelFlag,
    /// Polls the worker channel. Returns `true` once the job has settled
    /// (result applied, or the worker vanished) and should be dropped.
    poll: Box<dyn FnMut(&mut App<'a>) -> bool + 'a>,
}

type PoolTask = Box<dyn FnOnce() + Send>;

fn build_pool(name: &'static str, workers: usize) -> mpsc::Sender<PoolTask> {
    let (tx, rx) = mpsc::channel::<PoolTask>();
    let rx = Arc::new(Mutex::new(rx));
    for index in 0..workers {
        let rx = Arc::clone(&rx);
        std::thread::Builder::new()
            .name(format!("{name}-{index}"))
            .spawn(move || {
                loop {
                    let task = match rx.lock() {
                        Ok(receiver) => receiver.recv(),
                        Err(_) => break,
                    };
                    match task {
                        // The worker loop must survive task panics: the task
                        // wrapper reports the failure through its own result
                        // channel, and anything escaping it is contained here
                        // so the pool never silently loses capacity.
                        Ok(task) => {
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
                        }
                        Err(_) => break,
                    }
                }
            })
            .expect("failed to spawn worker thread");
    }
    tx
}

/// Fixed-size pool shared by all background compute jobs; bounds how many
/// heavy computes run at once no matter how fast jobs are spawned.
fn job_pool() -> &'static mpsc::Sender<PoolTask> {
    static POOL: OnceLock<mpsc::Sender<PoolTask>> = OnceLock::new();
    POOL.get_or_init(|| {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .clamp(1, 4);
        build_pool("job-worker", workers)
    })
}

/// Small pool for blocking file I/O (mesh exports and other long writes),
/// kept separate so disk-bound work and CPU-bound work cannot starve each
/// other.
fn io_pool() -> &'static mpsc::Sender<PoolTask> {
    static POOL: OnceLock<mpsc::Sender<PoolTask>> = OnceLock::new();
    POOL.get_or_init(|| build_pool("io-worker", 2))
}

/// Queue a CPU-bound task on the bounded compute pool. Prefer this over
/// `std::thread::spawn` for loads/parses: a folder of files queues instead of
/// creating one OS thread (and one peak allocation) per file.
pub(crate) fn spawn_pool_task(task: impl FnOnce() + Send + 'static) {
    let _ = job_pool().send(Box::new(task));
}

/// Queue a blocking-I/O task (file writes/exports) on the bounded I/O pool.
pub(crate) fn spawn_io_task(task: impl FnOnce() + Send + 'static) {
    let _ = io_pool().send(Box::new(task));
}

/// Run `compute`, converting a panic into an error result so the caller's
/// channel always observes an outcome.
pub(crate) fn run_compute_catching_panic<T>(
    compute: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(compute)) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_owned());
            Err(anyhow::anyhow!("background task panicked: {message}"))
        }
    }
}

impl<'a> App<'a> {
    fn job_dependencies_are_current(&self, keys: &[JobKey]) -> bool {
        keys.iter().all(|key| match *key {
            JobKey::Triangulation(id) => self.triangulations.iter().any(|item| item.id == id),
            JobKey::PointCloud(id) => self.point_clouds.iter().any(|item| item.id == id),
            JobKey::BlockModel(id) => self.block_models.iter().any(|item| item.id == id),
            JobKey::Project {
                runtime_id,
                document_revision,
            } => self
                .workspace
                .projects
                .iter()
                .find(|project| project.runtime_id == runtime_id)
                .is_some_and(|project| project.pidb.document.revision() == document_revision),
            JobKey::Anonymous => true,
        })
    }

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
        keys: Vec<JobKey>,
        compute: C,
        apply: A,
    ) where
        T: Send + 'static,
        C: FnOnce(&CancelFlag) -> anyhow::Result<T> + Send + 'static,
        A: FnOnce(&mut App<'a>, anyhow::Result<T>) + 'a,
    {
        let label = label.into();
        let ticket = self.begin_topology_load();

        let (tx, rx) = mpsc::channel();
        let window = self.window.clone();
        let cancel = CancelFlag::default();
        let worker_cancel = cancel.clone();
        let task: PoolTask = Box::new(move || {
            // A job cancelled while still queued never starts computing;
            // dropping `tx` settles its poll via Disconnected.
            if !worker_cancel.is_cancelled() {
                let _ = tx.send(run_compute_catching_panic(|| compute(&worker_cancel)));
            }
            if let Some(window) = window {
                window.request_redraw();
            }
        });
        let _ = job_pool().send(task);

        let mut apply = Some(apply);
        let poll_label = label.clone();
        let poll_cancel = cancel.clone();
        let poll_keys = keys.clone();
        let poll = Box::new(move |app: &mut App<'a>| -> bool {
            match rx.try_recv() {
                Ok(result) => {
                    if !app.job_dependencies_are_current(&poll_keys) {
                        crate::userspace_warn!(
                            "Discarded stale background result for '{poll_label}' because a source changed or closed"
                        );
                    } else if let Some(apply) = apply.take() {
                        apply(app, result);
                    }
                    true
                }
                Err(mpsc::TryRecvError::Empty) => false,
                // Worker dropped the sender without a value. A cancelled job
                // settles silently; anything else is an unexpected loss and
                // must be visible rather than quietly treated as done.
                Err(mpsc::TryRecvError::Disconnected) => {
                    if !poll_cancel.is_cancelled() {
                        crate::userspace_error!(
                            "Background task '{poll_label}' ended without a result"
                        );
                    }
                    true
                }
            }
        });

        self.pending_jobs.push(BackgroundJob {
            ticket,
            label,
            keys,
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
                self.finish_background_task(job.ticket, false);
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

    /// Cancel in-flight jobs where any dependency key matches `pred`: their
    /// cancel flag is set (so queued computes never start and running ones
    /// can bail) and their results are discarded. Settles the progress
    /// counter for each so the progress cursor doesn't stick.
    pub(crate) fn cancel_jobs(&mut self, pred: impl Fn(&JobKey) -> bool) {
        let mut cancelled_tickets = Vec::new();
        self.pending_jobs.retain(|job| {
            if job.keys.iter().any(&pred) {
                job.cancel.cancel();
                cancelled_tickets.push(job.ticket);
                false
            } else {
                true
            }
        });
        for ticket in cancelled_tickets {
            self.cancel_background_task(ticket);
        }
    }
}
