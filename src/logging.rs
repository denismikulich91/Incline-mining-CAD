//! Handles program logging

use std::{
    collections::VecDeque,
    fmt, fs, io,
    path::PathBuf,
    sync::{Mutex, OnceLock, TryLockError},
};

use chrono::{DateTime, Local};
use colored::Colorize;

/// Length of the rolling log buffer
const MAX_BUFFER_BYTES: usize = 2 * 1024 * 1024;

static RUNTIME_LOG: OnceLock<Mutex<RuntimeLog>> = OnceLock::new();

struct RuntimeLog {
    started_at: DateTime<Local>,
    lines: VecDeque<ConsoleLine>,
    bytes: usize,
}

#[derive(Clone)]
pub(crate) struct ConsoleLine {
    pub(crate) text: String,
    pub(crate) level: Level,
}

impl RuntimeLog {
    fn new() -> Self {
        Self {
            started_at: Local::now(),
            lines: VecDeque::new(),
            bytes: 0,
        }
    }

    fn push(&mut self, line: ConsoleLine) {
        self.bytes += line.text.len() + 1;
        self.lines.push_back(line);
        while self.bytes > MAX_BUFFER_BYTES && self.lines.len() > 1 {
            if let Some(removed) = self.lines.pop_front() {
                self.bytes = self.bytes.saturating_sub(removed.text.len() + 1);
            }
        }
    }

    fn contents(&self) -> String {
        let mut output = String::with_capacity(self.bytes);
        for line in &self.lines {
            output.push_str(&line.text);
            output.push('\n');
        }
        output
    }
}

fn runtime_log() -> &'static Mutex<RuntimeLog> {
    RUNTIME_LOG.get_or_init(|| Mutex::new(RuntimeLog::new()))
}

fn with_runtime_log<T>(f: impl FnOnce(&mut RuntimeLog) -> T) -> T {
    let mut log = runtime_log()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(&mut log)
}

struct Logger;

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format_line(record.level().as_str(), format_args!("{}", record.args()));
        let level = match record.level() {
            log::Level::Error => Level::Error,
            log::Level::Warn => Level::Warn,
            log::Level::Info | log::Level::Debug | log::Level::Trace => Level::Info,
        };
        match level {
            Level::Error => eprintln!("{}", line.red()),
            Level::Warn => eprintln!("{}", line.yellow()),
            Level::Info => eprintln!("{}", line.blue()),
        }
        append_line(line, level);
    }

    fn flush(&self) {}
}

pub(crate) fn init() {
    let _ = runtime_log();

    log::set_boxed_logger(Box::new(Logger)).unwrap();
    log::set_max_level(log::LevelFilter::Info);

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        if PANIC_HOOK_SUPPRESSED.get() {
            return;
        }
        if let Err(error) = save_crash_log(panic_info) {
            eprintln!("Failed to save crash log: {error}");
        }
        default_hook(panic_info);
    }));
}

thread_local! {
    static PANIC_HOOK_SUPPRESSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `operation`, converting a panic into `None`. While it runs, the
/// process panic hook (crash-log write + stderr report) is suppressed on this
/// thread, so a recovered panic in third-party code doesn't leave a spurious
/// crash log behind. The caller is responsible for judging that the panicking
/// code leaves its data structures in a usable state.
pub(crate) fn catch_panic_quietly<T>(operation: impl FnOnce() -> T) -> Option<T> {
    PANIC_HOOK_SUPPRESSED.set(true);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
    PANIC_HOOK_SUPPRESSED.set(false);
    result.ok()
}

/// Log levels for the in-app console.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

impl Level {
    fn prefix(self) -> &'static str {
        match self {
            Level::Info => "INFO",
            Level::Warn => "WARN",
            Level::Error => "ERROR",
        }
    }
}

fn timestamp() -> String {
    Local::now().format("%H:%M:%S").to_string()
}

fn format_line(level: &str, args: fmt::Arguments<'_>) -> String {
    format!("[{}] [{level}] {args}", timestamp())
}

fn append_line(text: String, level: Level) {
    with_runtime_log(|log| log.push(ConsoleLine { text, level }));
}

pub(crate) fn console_lines() -> Vec<ConsoleLine> {
    with_runtime_log(|log| log.lines.iter().cloned().collect())
}

pub(crate) fn log_to_distributor(args: fmt::Arguments<'_>, level: Level) {
    let line = format_line(level.prefix(), args);

    match level {
        Level::Error => eprintln!("{}", line.red()),
        Level::Warn => eprintln!("{}", line.yellow()),
        Level::Info => eprintln!("{}", line.blue()),
    }

    append_line(line, level);
}

pub(crate) fn save_runtime_log() -> io::Result<PathBuf> {
    save_log("runtime")
}

fn save_crash_log(panic_info: &std::panic::PanicHookInfo<'_>) -> io::Result<PathBuf> {
    let panic_line = format!("[{}] [PANIC] {panic_info}", timestamp());
    let (started_at, contents) = match runtime_log().try_lock() {
        Ok(mut log) => {
            log.push(ConsoleLine {
                text: panic_line,
                level: Level::Error,
            });
            (log.started_at, log.contents())
        }
        Err(TryLockError::Poisoned(poisoned)) => {
            let mut log = poisoned.into_inner();
            log.push(ConsoleLine {
                text: panic_line,
                level: Level::Error,
            });
            (log.started_at, log.contents())
        }
        // If the panicking thread already owns the log lock, never deadlock the
        // panic hook. A minimal crash report is still more useful than no file.
        Err(TryLockError::WouldBlock) => (
            Local::now(),
            format!("{panic_line}\nRuntime log was locked while the process panicked.\n"),
        ),
    };
    write_log("crash", started_at, &contents)
}

fn save_log(kind: &str) -> io::Result<PathBuf> {
    let (started_at, contents) = with_runtime_log(|log| (log.started_at, log.contents()));
    write_log(kind, started_at, &contents)
}

fn write_log(kind: &str, started_at: DateTime<Local>, contents: &str) -> io::Result<PathBuf> {
    let logs_dir = crate::app::io::data_path("logs")?;
    fs::create_dir_all(&logs_dir)?;
    let filename = format!("{}.{}.log", started_at.format("%Y-%m-%d_%H-%M-%S"), kind);
    let path = logs_dir.join(filename);
    fs::write(&path, contents)?;
    Ok(path)
}

/// Log an informational message visible in the Debug/Console window.
#[macro_export]
macro_rules! userspace_log {
    ($($arg:tt)*) => {
        $crate::logging::log_to_distributor(format_args!($($arg)*), $crate::logging::Level::Info)
    };
}

/// Log a warning message visible in the Debug/Console window.
#[macro_export]
macro_rules! userspace_warn {
    ($($arg:tt)*) => {
        $crate::logging::log_to_distributor(format_args!($($arg)*), $crate::logging::Level::Warn)
    };
}

/// Log an error message visible in the Debug/Console window.
#[macro_export]
macro_rules! userspace_error {
    ($($arg:tt)*) => {
        $crate::logging::log_to_distributor(format_args!($($arg)*), $crate::logging::Level::Error)
    };
}
