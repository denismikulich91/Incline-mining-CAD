// io.rs
// responsible for managing config files, and other editor files storage.

use std::{fs, io, io::Write, path::PathBuf};

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

pub(crate) const fn default_show_console() -> bool {
    true
}

pub(crate) const fn default_plan_orbit_sensitivity() -> f64 {
    0.003
}

pub(crate) const fn default_plan_zoom_sensitivity() -> f64 {
    0.005
}

pub(crate) const fn default_plan_zoom_towards_cursor() -> bool {
    true
}

pub(crate) const fn default_fly_field_of_view_degrees() -> f64 {
    40.0
}

pub(crate) const fn default_fly_mouse_look_sensitivity() -> f64 {
    0.003
}

pub(crate) const fn default_fly_near_clip_limit() -> f64 {
    0.25
}

pub(crate) const fn default_fly_max_clip_span() -> f64 {
    150_000.0
}

pub(crate) fn finite_clamped(value: f64, min: f64, max: f64, default: f64) -> f64 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        default
    }
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
    /// Individually imported georeferenced imagery (.tif/.tiff).
    #[serde(default)]
    pub(crate) raster_file_paths: Vec<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Config {
    /// Show wireframes for every topology mesh. Selection still displays a
    /// highlighted wireframe when this global option is disabled.
    #[serde(default)]
    pub(crate) topology_wireframes_enabled: bool,
    /// Show vertices for all visible design objects.
    #[serde(default)]
    pub(crate) show_points: bool,
    /// Use egui's dark visuals and the dark UI icon set.
    #[serde(default)]
    pub(crate) dark_mode: bool,
    /// Show the console pannel
    #[serde(default = "default_show_console")]
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
    #[serde(default)]
    pub(crate) debug_clip_planes: bool,
    #[serde(default = "default_plan_orbit_sensitivity")]
    pub(crate) plan_orbit_sensitivity: f64,
    #[serde(default = "default_plan_zoom_sensitivity")]
    pub(crate) plan_zoom_sensitivity: f64,
    #[serde(default)]
    pub(crate) plan_invert_vertical_look: bool,
    #[serde(default)]
    pub(crate) plan_invert_horizontal_look: bool,
    #[serde(default = "default_plan_zoom_towards_cursor")]
    pub(crate) plan_zoom_towards_cursor: bool,
    #[serde(default = "default_fly_field_of_view_degrees")]
    pub(crate) fly_field_of_view_degrees: f64,
    #[serde(default = "default_fly_mouse_look_sensitivity")]
    pub(crate) fly_mouse_look_sensitivity: f64,
    #[serde(default)]
    pub(crate) fly_invert_vertical_look: bool,
    #[serde(default)]
    pub(crate) fly_invert_horizontal_look: bool,
    #[serde(default = "default_fly_near_clip_limit")]
    pub(crate) fly_near_clip_limit: f64,
    #[serde(default = "default_fly_max_clip_span")]
    pub(crate) fly_max_clip_span: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            topology_wireframes_enabled: false,
            show_points: false,
            dark_mode: false,
            show_console: default_show_console(),
            renderer_background_color: default_renderer_background_color(),
            snap_poll_rate: default_snap_poll_rate(),
            frame_rate_cap: default_frame_rate_cap(),
            resize_frame_rate_cap: default_resize_frame_rate_cap(),
            block_model_interaction_resolution_divisor:
                default_block_model_interaction_resolution_divisor(),
            frame_counter_enabled: false,
            show_world_axis_gizmo: default_show_world_axis_gizmo(),
            debug_chunk_coloring: false,
            debug_clip_planes: false,
            plan_orbit_sensitivity: default_plan_orbit_sensitivity(),
            plan_zoom_sensitivity: default_plan_zoom_sensitivity(),
            plan_invert_vertical_look: false,
            plan_invert_horizontal_look: false,
            plan_zoom_towards_cursor: default_plan_zoom_towards_cursor(),
            fly_field_of_view_degrees: default_fly_field_of_view_degrees(),
            fly_mouse_look_sensitivity: default_fly_mouse_look_sensitivity(),
            fly_invert_vertical_look: false,
            fly_invert_horizontal_look: false,
            fly_near_clip_limit: default_fly_near_clip_limit(),
            fly_max_clip_span: default_fly_max_clip_span(),
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

/// Write through the shared atomic-file helper so a crash, full disk, or a
/// concurrent writer cannot leave a truncated or missing file behind.
fn write_atomic(path: &std::path::Path, contents: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    crate::model::atomic_file::write_atomic(path, |file| {
        file.write_all(contents)?;
        Ok(())
    })
    .map_err(io::Error::other)
}
