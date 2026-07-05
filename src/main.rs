// Disable console window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod logging;
mod model;
mod rendering;
mod startup;
mod ui;

use anyhow::Result;
use winit::event_loop::EventLoop;

use crate::app::App;

pub(crate) type Size = (f32, f32);
/// Name of the design software
pub(crate) const APP_NAME: &str = "Incline";
/// Internal name of the design software
pub(crate) const APP_ID: &str = env!("CARGO_PKG_NAME");
/// Version of release
pub(crate) const APP_RELEASE: &str = env!("CARGO_PKG_VERSION");

fn main() -> Result<()> {
    logging::init();
    let result: Result<()> = (|| {
        let event_loop = EventLoop::new()?;
        startup::init();
        let mut app = App::new()?;
        event_loop.run_app(&mut app)?;
        Ok(())
    })();
    let log_result = logging::save_runtime_log();
    result?;
    log_result?;
    Ok(())
}
