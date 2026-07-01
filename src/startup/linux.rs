//! Linux specific startup code

use std::path::{Path, PathBuf};

const ICON: &[u8] = include_bytes!("../../res/logo.svg");

pub(crate) fn install_wayland_desktop_entry() -> anyhow::Result<()> {
    {
        if !is_wayland_session() {
            return Ok(());
        }

        let data_home = data_home().ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
        let applications_dir = data_home.join("applications");
        let icons_dir = data_home.join("icons/hicolor/scalable/apps");
        std::fs::create_dir_all(&applications_dir)?;
        std::fs::create_dir_all(&icons_dir)?;

        let icon_path = icons_dir.join(format!("{}.svg", crate::APP_ID));
        let icon_changed = write_if_changed(&icon_path, ICON)?;

        let executable = std::env::current_exe()?;
        let desktop_entry = format!(
            "[Desktop Entry]\n\
             Type=Application\n\
             Name={}\n\
             Comment=Inspect and edit survey projects\n\
             Exec={}\n\
             Icon={}\n\
             Terminal=false\n\
             Categories=Graphics;Engineering;\n",
            crate::APP_NAME,
            quote_exec_path(&executable),
            icon_path.display(),
        );
        let desktop_path = applications_dir.join(format!("{}.desktop", crate::APP_ID));
        let desktop_changed = write_if_changed(&desktop_path, desktop_entry.as_bytes())?;

        if (icon_changed || desktop_changed) && is_kde_desktop() {
            refresh_kde_service_cache()?;
        }

        Ok(())
    }
}

fn is_wayland_session() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some()
        || std::env::var("XDG_SESSION_TYPE")
            .is_ok_and(|value| value.eq_ignore_ascii_case("wayland"))
}

fn data_home() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
}

fn is_kde_desktop() -> bool {
    std::env::var("XDG_CURRENT_DESKTOP").is_ok_and(|desktop| {
        desktop
            .split(':')
            .any(|name| name.eq_ignore_ascii_case("kde"))
    })
}

fn refresh_kde_service_cache() -> anyhow::Result<()> {
    let status = std::process::Command::new("kbuildsycoca6")
        .env("LANG", "C.UTF-8")
        .env("LC_ALL", "C.UTF-8")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()?;
    anyhow::ensure!(status.success(), "kbuildsycoca6 exited with {status}");
    Ok(())
}

fn quote_exec_path(path: &Path) -> String {
    let path = path.to_string_lossy();
    let escaped = path.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn write_if_changed(path: &Path, contents: &[u8]) -> std::io::Result<bool> {
    if std::fs::read(path).is_ok_and(|existing| existing == contents) {
        return Ok(false);
    }
    std::fs::write(path, contents)?;
    Ok(true)
}
