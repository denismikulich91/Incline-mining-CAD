//! Code that runs on program Start

#[cfg(target_os = "linux")]
pub(crate) mod linux;
#[cfg(target_os = "macos")]
pub(crate) mod macos;

use crate::userspace_log;

pub(crate) fn init() {
    userspace_log!("APPLICATION NAME: {}", crate::APP_NAME);
    userspace_log!("Application ID: {}", crate::APP_ID);
    userspace_log!("Release Version: {}", crate::APP_RELEASE);
    userspace_log!(
        "Build Target: {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    userspace_log!("Pointer Width: {}-bit", usize::BITS);
    userspace_log!(
        "Rustc Host: {}",
        std::env::var("HOST").unwrap_or_else(|_| "unknown".to_string())
    );
    userspace_log!("Process ID: {}", std::process::id());
    userspace_log!(
        "Locale Env: LANG={:?}, LC_ALL={:?}, TZ={:?}",
        std::env::var_os("LANG"),
        std::env::var_os("LC_ALL"),
        std::env::var_os("TZ")
    );

    #[cfg(target_os = "linux")]
    {
        use crate::userspace_log;

        userspace_log!("Operating System: GNU / Linux");
        userspace_log!(
            "Desktop Session: XDG_SESSION_TYPE={:?}, XDG_CURRENT_DESKTOP={:?}, WAYLAND_DISPLAY={:?}, DISPLAY={:?}",
            std::env::var_os("XDG_SESSION_TYPE"),
            std::env::var_os("XDG_CURRENT_DESKTOP"),
            std::env::var_os("WAYLAND_DISPLAY"),
            std::env::var_os("DISPLAY")
        );
        if let Err(error) = crate::startup::linux::install_wayland_desktop_entry() {
            log::warn!("Failed to install Wayland desktop integration: {error}");
        }
    }

    #[cfg(target_os = "windows")]
    {
        use crate::userspace_log;
        userspace_log!("Operating System: Microsoft Windows");
        userspace_log!(
            "Windows Session: SESSIONNAME={:?}, USERNAME={:?}",
            std::env::var_os("SESSIONNAME"),
            std::env::var_os("USERNAME")
        );
    }

    #[cfg(target_os = "macos")]
    {
        use crate::userspace_log;
        userspace_log!("Operating System: MacOS");
        userspace_log!(
            "macOS Session: USER={:?}, SHELL={:?}",
            std::env::var_os("USER"),
            std::env::var_os("SHELL")
        );
    }
}
