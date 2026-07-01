// io.rs
// responsible for managing config files, and other editor files storage.

use std::{env, fs, io, path::PathBuf};

use serde::{Deserialize, Serialize};

pub(crate) fn default_renderer_background_color() -> [f32; 4] {
    crate::rendering::color::hex_to_linear_rgba(0x232c36)
}

pub(crate) fn default_selection_color() -> [f32; 4] {
    [87.0 / 255.0, 163.0 / 255.0, 1.0, 1.0]
}

pub(crate) const fn default_snap_poll_rate() -> u32 {
    30
}

pub(crate) const fn default_frame_rate_cap() -> u32 {
    144
}

pub(crate) const fn default_resize_frame_rate_cap() -> u32 {
    80
}

pub(crate) const fn default_topology_folder_search_depth() -> u32 {
    0
}

pub(crate) const fn default_show_world_axis_gizmo() -> bool {
    true
}

pub(crate) const fn default_show_view_cube() -> bool {
    true
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct Session {
    /// Paths of all pidb files that were open, sorted alphabetically by name.
    #[serde(default)]
    pub(crate) project_paths: Vec<PathBuf>,
    /// Which of those paths was the active project.
    pub(crate) active_path: Option<PathBuf>,
    /// Directories whose .00t contents are shown in the explorer.
    #[serde(default)]
    pub(crate) triangulation_paths: Vec<PathBuf>,
    /// Individually-opened .00t files (not whole directories).
    #[serde(default)]
    pub(crate) triangulation_file_paths: Vec<PathBuf>,
    /// .00t paths explicitly removed by the user; hidden even if their dir is scanned.
    #[serde(default)]
    pub(crate) triangulation_excluded_paths: Vec<PathBuf>,
    /// Individually imported Vulcan block model pairs.
    #[serde(default)]
    pub(crate) block_model_sources: Vec<crate::model::block_model::BlockModelSource>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Config {
    /// Show wireframes for every topology mesh. Selection still displays a
    /// highlighted wireframe when this global option is disabled.
    #[serde(default)]
    pub(crate) topology_wireframes_enabled: bool,
    /// Use egui's dark visuals and the dark UI icon set.
    #[serde(default)]
    pub(crate) dark_mode: bool,
    /// Show the console pannel
    #[serde(default)]
    pub(crate) show_console: bool,
    /// Linear RGBA clear colour used behind the rendered scene.
    #[serde(default = "default_renderer_background_color")]
    pub(crate) renderer_background_color: [f32; 4],
    /// RGBA colour used for UI accents and selected renderer geometry.
    #[serde(default = "default_selection_color")]
    pub(crate) selection_color: [f32; 4],
    #[serde(default = "default_snap_poll_rate")]
    pub(crate) snap_poll_rate: u32,
    #[serde(default = "default_frame_rate_cap")]
    pub(crate) frame_rate_cap: u32,
    #[serde(default = "default_resize_frame_rate_cap")]
    pub(crate) resize_frame_rate_cap: u32,
    #[serde(default)]
    pub(crate) frame_counter_enabled: bool,
    #[serde(default = "default_topology_folder_search_depth")]
    pub(crate) topology_folder_search_depth: u32,
    #[serde(default = "default_show_world_axis_gizmo")]
    pub(crate) show_world_axis_gizmo: bool,
    #[serde(default = "default_show_view_cube")]
    pub(crate) show_view_cube: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            topology_wireframes_enabled: false,
            dark_mode: false,
            show_console: false,
            renderer_background_color: default_renderer_background_color(),
            selection_color: default_selection_color(),
            snap_poll_rate: default_snap_poll_rate(),
            frame_rate_cap: default_frame_rate_cap(),
            resize_frame_rate_cap: default_resize_frame_rate_cap(),
            frame_counter_enabled: false,
            topology_folder_search_depth: default_topology_folder_search_depth(),
            show_world_axis_gizmo: default_show_world_axis_gizmo(),
            show_view_cube: default_show_view_cube(),
        }
    }
}

pub(crate) fn save_config(config: &Config) -> io::Result<()> {
    let path = local_to_global_path("data/config.toml")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let contents = toml::to_string_pretty(config).map_err(io::Error::other)?;
    fs::write(path, contents)
}

pub(crate) fn load_config() -> io::Result<Config> {
    let contents = fs::read_to_string(local_to_global_path("data/config.toml")?)?;
    toml::from_str(&contents).map_err(io::Error::other)
}

pub(crate) fn save_session(session: &Session) -> io::Result<()> {
    let path = local_to_global_path("data/last_session.toml")?;

    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let toml_string = toml::to_string_pretty(session).map_err(io::Error::other)?;

    fs::write(path, toml_string)?;

    Ok(())
}

pub(crate) fn load_session() -> io::Result<Session> {
    let contents = fs::read_to_string(local_to_global_path("data/last_session.toml")?)?;

    let session: Session = toml::from_str(&contents).map_err(io::Error::other)?;

    Ok(session)
}

pub(crate) fn local_to_global_path(path: &str) -> Result<PathBuf, std::io::Error> {
    let exe = env::current_exe()?;
    let base = exe
        .parent()
        .ok_or_else(|| std::io::Error::other("executable has no parent directory"))?;
    Ok(base.join(path))
}
