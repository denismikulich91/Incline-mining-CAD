// io.rs
// responsible for managing config files, and other editor files storage.

use std::{
    fs, io,
    io::Write,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

pub(crate) fn default_renderer_background_color() -> [f32; 4] {
    crate::rendering::color::hex_to_linear_rgba(0x232c36)
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

pub(crate) const fn default_block_model_interaction_resolution_divisor() -> u32 {
    3
}

pub(crate) const fn default_show_world_axis_gizmo() -> bool {
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
    /// Individually imported point cloud files (.las/.laz/.xyz/.pts/.pcd).
    #[serde(default)]
    pub(crate) point_cloud_file_paths: Vec<PathBuf>,
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
    #[serde(default = "default_snap_poll_rate")]
    pub(crate) snap_poll_rate: u32,
    #[serde(default = "default_frame_rate_cap")]
    pub(crate) frame_rate_cap: u32,
    #[serde(default = "default_resize_frame_rate_cap")]
    pub(crate) resize_frame_rate_cap: u32,
    #[serde(default = "default_block_model_interaction_resolution_divisor")]
    pub(crate) block_model_interaction_resolution_divisor: u32,
    #[serde(default)]
    pub(crate) frame_counter_enabled: bool,
    #[serde(default = "default_show_world_axis_gizmo")]
    pub(crate) show_world_axis_gizmo: bool,
    #[serde(default)]
    pub(crate) debug_chunk_coloring: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            topology_wireframes_enabled: false,
            dark_mode: false,
            show_console: true,
            renderer_background_color: default_renderer_background_color(),
            snap_poll_rate: default_snap_poll_rate(),
            frame_rate_cap: default_frame_rate_cap(),
            resize_frame_rate_cap: default_resize_frame_rate_cap(),
            block_model_interaction_resolution_divisor:
                default_block_model_interaction_resolution_divisor(),
            frame_counter_enabled: false,
            show_world_axis_gizmo: default_show_world_axis_gizmo(),
            debug_chunk_coloring: false,
        }
    }
}

pub(crate) fn save_config(config: &Config) -> io::Result<()> {
    let path = data_path("config.toml")?;
    let contents = toml::to_string_pretty(config).map_err(io::Error::other)?;
    write_atomic(&path, contents.as_bytes())
}

pub(crate) fn load_config() -> io::Result<Config> {
    let contents = fs::read_to_string(data_path("config.toml")?)?;
    toml::from_str(&contents).map_err(io::Error::other)
}

pub(crate) fn save_session(session: &Session) -> io::Result<()> {
    let path = data_path("last_session.toml")?;
    let contents = toml::to_string_pretty(session).map_err(io::Error::other)?;
    write_atomic(&path, contents.as_bytes())
}

pub(crate) fn load_session() -> io::Result<Session> {
    let contents = fs::read_to_string(data_path("last_session.toml")?)?;

    let session: Session = toml::from_str(&contents).map_err(io::Error::other)?;

    Ok(session)
}

/// Resolve a path inside the editor's data directory: the platform config
/// directory (`$XDG_CONFIG_HOME`, `~/Library/Application Support`,
/// `%APPDATA%`) under `incline/`.
pub(crate) fn data_path(relative: &str) -> io::Result<PathBuf> {
    dirs::config_dir()
        .map(|dir| dir.join("incline").join(relative))
        .ok_or_else(|| io::Error::other("no platform config directory"))
}

/// Write via temp file + `sync_all` + rename so a crash or full disk cannot
/// leave a truncated file behind (mirrors `pidb::save`).
fn write_atomic(path: &Path, contents: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut tmp_name = path.file_name().unwrap_or_default().to_owned();
    tmp_name.push(".tmp");
    let tmp_path = path.with_file_name(tmp_name);
    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(contents)?;
        file.sync_all()?;
    }
    fs::rename(&tmp_path, path)
}
