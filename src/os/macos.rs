//! MacOS specific startup code

use anyhow::{Context, Result};
use objc2::{AnyThread, MainThreadMarker};
use objc2_app_kit::{NSApplication, NSImage};
use objc2_foundation::NSData;

pub(crate) fn set_dock_icon() -> Result<()> {
    let mtm = MainThreadMarker::new().context("Dock icon must be set on the main thread")?;
    // Dock icons are visually smaller than their full image canvas. Give the
    // existing edge-to-edge logo a transparent macOS-style safe area without
    // changing the logo asset used by other platforms.
    let svg =
        include_str!("../../res/logo.svg").replacen("<svg", r#"<svg viewBox="-15 -15 150 150""#, 1);
    let bytes = svg.as_bytes();
    let data = unsafe { NSData::dataWithBytes_length(bytes.as_ptr().cast(), bytes.len()) };
    let image = NSImage::initWithData(NSImage::alloc(), &data)
        .context("macOS could not decode the application logo")?;
    let application = NSApplication::sharedApplication(mtm);

    unsafe {
        application.setApplicationIconImage(Some(&image));
    }
    Ok(())
}
