pub(crate) mod canvas; // Handles anything to do with dragging and stuff
pub(crate) mod commands; // Handles UI commands
pub(crate) mod events; // Handles window events
pub(crate) mod io; /* Handles session serialisation */

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use anyhow::Result;
use glam::DVec3;
#[cfg(target_os = "linux")]
use winit::platform::wayland::WindowAttributesExtWayland;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, *},
    event_loop::ControlFlow,
    keyboard::ModifiersState,
    window::{CursorIcon, Icon, Window},
};

use crate::{
    app::commands::file::FileDialogAction,
    model::{
        Document, LayerId, Object, ObjectId, SceneEntityId,
        block_model::{BlockModelId, BlockModelSource, OpenBlockModel},
        formats::MeshFormat,
        pidb::{self, OpenProject, Workspace},
        spatial::ObjectSnapIndex,
        triangulation::{OpenTriangulation, TriangulationId},
    },
    rendering::graphics::Graphics,
    ui::state::{
        EditorState, UiBlockModelEntry, UiLayerEntry, UiProjectEntry, UiProjectView,
        UiTriangulationEntry,
    },
    userspace_warn,
};

pub(crate) const PICK_THRESHOLD_PX: f32 = 8.0;

fn rate_interval(rate: u32) -> Duration {
    Duration::from_secs_f64(1.0 / f64::from(rate.clamp(1, 1000)))
}

fn window_icon() -> Option<Icon> {
    let image = egui_extras::image::load_svg_bytes(
        include_bytes!("../../res/logo.svg"),
        &Default::default(),
    )
    .map_err(|error| log::error!("Failed to rasterize window icon: {error}"))
    .ok()?;
    let [width, height] = image.size;
    let rgba = image
        .pixels
        .iter()
        .flat_map(egui::Color32::to_srgba_unmultiplied)
        .collect();

    Icon::from_rgba(rgba, width as u32, height as u32)
        .map_err(|error| log::error!("Failed to create window icon: {error}"))
        .ok()
}

struct DragState {
    object_id: ObjectId,
    before: Object,
    plane_z: f64,
    last_world: DVec3,
    moved: bool,
}

pub(crate) struct GizmoDragState {
    pub(crate) axis: DVec3,
    pub(crate) start_cursor_screen_px: (f32, f32),
    pub(crate) axis_screen_dir: (f32, f32),
    pub(crate) px_per_world_unit: f64,
    pub(crate) start_delta: DVec3,
}

pub(crate) struct App<'a> {
    close_requested: bool,
    redraw_requested: bool,
    window: Option<Arc<Window>>,
    graphics: Option<Graphics<'a>>,
    /// Latest non-zero window size awaiting surface reconfiguration. Resize
    /// events arrive in bursts while dragging, so intermediate sizes are
    /// deliberately replaced instead of configuring a swapchain for each one.
    pending_resize: Option<winit::dpi::PhysicalSize<u32>>,
    last_render_time: Option<Instant>,
    last_scroll_instant: Option<Instant>,
    last_snap_poll_instant: Option<Instant>,
    editor: EditorState,
    workspace: Workspace,
    startup_dialog_dismissed: bool,
    triangulations: Vec<OpenTriangulation>,
    triangulation_dirs: Vec<PathBuf>,
    /// Supported mesh paths discovered in each tracked directory. Directory
    /// contents change much less often than the UI redraws, so keep filesystem
    /// scanning and sorting out of the render loop.
    triangulation_dir_entries: BTreeMap<PathBuf, Vec<PathBuf>>,
    triangulation_files: Vec<PathBuf>,
    triangulation_excluded_paths: BTreeSet<PathBuf>,
    active_triangulation: Option<TriangulationId>,
    next_triangulation_id: u64,
    block_models: Vec<OpenBlockModel>,
    block_model_files: Vec<BlockModelSource>,
    active_block_model: Option<BlockModelId>,
    next_block_model_id: u64,
    empty_document: Document,
    scene_document: Document,
    snap_index: ObjectSnapIndex,
    /// Set by `invalidate_geometry`; the index rebuilds lazily on the next
    /// snap/orbit query via `refresh_snap_index`.
    snap_index_dirty: bool,
    /// `Workspace::composite_key()` of the last `scene_document` build;
    /// `None` forces the next invalidation to rebuild.
    scene_document_key: Option<u64>,
    history: crate::model::History,
    modifiers: ModifiersState,
    drag: Option<DragState>,
    pub(crate) gizmo_drag: Option<GizmoDragState>,
    /// Screen position where the right mouse button was pressed (physical px).
    /// Used to distinguish a quick context-menu click from a camera orbit drag.
    right_press_px: Option<(f32, f32)>,
    /// True after a pending right press has become an active camera orbit drag.
    right_orbit_active: bool,
    pending_topology_click: Option<(SceneEntityId, DVec3)>,
    move_session_original: Option<Vec<Object>>,
    move_session_project_dirty: Option<bool>,
    topology_load_pending_gpu: bool,
    /// Number of triangulation loads currently running on background threads.
    pending_loads: usize,
    pending_triangulation_loads: Vec<(
        PathBuf,
        mpsc::Receiver<anyhow::Result<crate::model::triangulation::LoadedTriangulation>>,
    )>,
    pending_block_model_loads: Vec<(
        BlockModelSource,
        mpsc::Receiver<anyhow::Result<crate::model::block_model::LoadedBlockModel>>,
    )>,
    pub(crate) pending_file_dialogs: Vec<mpsc::Receiver<Option<FileDialogAction>>>,
    window_focused: bool,
}

impl<'a> App<'a> {
    pub(crate) fn new() -> Result<Self> {
        let mut app = Self {
            close_requested: false,
            redraw_requested: false,
            window: None,
            graphics: None,
            pending_resize: None,
            last_render_time: None,
            last_scroll_instant: None,
            last_snap_poll_instant: None,
            editor: EditorState::new(),
            workspace: Workspace::default(),
            startup_dialog_dismissed: false,
            triangulations: Vec::new(),
            triangulation_dirs: Vec::new(),
            triangulation_dir_entries: BTreeMap::new(),
            triangulation_files: Vec::new(),
            triangulation_excluded_paths: BTreeSet::new(),
            active_triangulation: None,
            next_triangulation_id: 0,
            block_models: Vec::new(),
            block_model_files: Vec::new(),
            active_block_model: None,
            next_block_model_id: 0,
            empty_document: Document::new(),
            scene_document: Document::new(),
            snap_index: ObjectSnapIndex::default(),
            snap_index_dirty: false,
            scene_document_key: None,
            history: crate::model::History::new(),
            modifiers: ModifiersState::empty(),
            drag: None,
            gizmo_drag: None,
            right_press_px: None,
            right_orbit_active: false,
            pending_topology_click: None,
            move_session_original: None,
            move_session_project_dirty: None,
            topology_load_pending_gpu: false,
            pending_loads: 0,
            pending_triangulation_loads: Vec::new(),
            pending_block_model_loads: Vec::new(),
            pending_file_dialogs: Vec::new(),
            window_focused: true,
        };

        let session = match io::load_session() {
            Ok(session) => session,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                crate::app::io::Session::default()
            }
            Err(e) => {
                userspace_warn!("Failed to load session file: {e}");
                crate::app::io::Session::default()
            }
        };
        let config = match io::load_config() {
            Ok(config) => config,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => crate::app::io::Config::default(),
            Err(e) => {
                userspace_warn!("Failed to load config file: {e}");
                crate::app::io::Config::default()
            }
        };

        app.load_session_projects(&session);
        app.editor.topology_wireframes_enabled = config.topology_wireframes_enabled;
        app.editor.dark_mode = config.dark_mode;
        app.editor.show_console = config.show_console;
        app.editor.show_world_axis_gizmo = config.show_world_axis_gizmo;
        app.editor.renderer_background_color = config.renderer_background_color;
        app.editor.snap_poll_rate = config.snap_poll_rate.clamp(5, 1000);
        app.editor.frame_rate_cap = config.frame_rate_cap.clamp(20, 1000);
        app.editor.resize_frame_rate_cap = config.resize_frame_rate_cap.clamp(20, 1000);
        app.editor.block_model_interaction_resolution_divisor = config
            .block_model_interaction_resolution_divisor
            .clamp(1, 64);
        app.editor.frame_counter_enabled = config.frame_counter_enabled;

        Ok(app)
    }

    fn active_document(&self) -> &Document {
        self.workspace
            .active_document()
            .unwrap_or(&self.empty_document)
    }

    fn active_layer(&self) -> Option<LayerId> {
        self.editor.active_layer.and_then(|layer| {
            self.workspace.active_project().and_then(|project| {
                (project.loaded_layers.contains(&layer)
                    && project.pidb.document.layer(layer).is_some())
                .then_some(layer)
            })
        })
    }

    fn active_layer_object(&self, object_id: ObjectId) -> Option<&Object> {
        let active_layer = self.active_layer()?;
        self.active_document()
            .get_object(object_id)
            .filter(|object| object.layer() == active_layer)
    }

    fn editing_ready(&mut self) -> bool {
        self.workspace.has_active_project()
    }

    fn set_active_project(&mut self, project: OpenProject) {
        self.workspace.add_and_activate(project);
        self.clear_editor_transient_state();
        self.invalidate_geometry();
        self.persist_session();
    }

    fn clear_editor_transient_state(&mut self) {
        self.history.clear();
        self.editor.selected_handles.clear();
        self.editor.hidden_handles.clear();
        self.editor.frozen_handles.clear();
        self.editor.translucent_handles.clear();
        self.editor.active_layer = None;
        self.editor.active_tool = crate::ui::state::ActiveTool::None;
        self.editor.selection_box_start_px = None;
        self.editor.selection_box_current_px = None;
        self.editor.move_to_layer_dialog = None;
        self.editor.move_layer_dialog = None;
        self.pending_topology_click = None;
        self.editor.measurement_start = None;
        self.editor.measurement_end = None;
        self.editor.text_editing_enabled = false;
        self.editor.editing_labels_id = None;
        self.editor.text_edit_dialog_px = None;
        self.editor.text_edit_position_frames = 0;
        self.editor.text_edit_focus_requested = false;
        self.editor.text_edit_created = false;
        self.editor.text_edit_was_dirty = false;
        // Cancel any in-progress offset operation.
        self.editor.offset_dialog_open = false;
        self.editor.offset_target_id = None;
        self.editor.offset_awaiting_side_pick = false;
        self.editor.offset_preview_world.clear();
        self.editor.offset_source_world.clear();
        // Cancel any in-progress relimit operation.
        self.editor.relimit_dialog_open = false;
        self.editor.relimit_source_id = None;
        self.editor.relimit_awaiting_source_pick = false;
        self.editor.relimit_waiting_for_pick = false;
        self.editor.relimit_confirming_end = false;
        // Clear chamfer tool state.
        self.editor.chamfer_poly_id = None;
        self.editor.chamfer_corner_index = None;
        self.editor.chamfer_gizmo_drag_start_px = None;
        self.editor.chamfer_gizmo_hovered = false;
        self.editor.chamfer_preview_screen_px.clear();
        // Clear bezier tool state.
        self.clear_bezier_state();
        // Clear any in-progress move session so it cannot bleed into the new PIDB.
        self.move_session_original = None;
        self.move_session_project_dirty = None;
        self.drag = None;
        self.gizmo_drag = None;
        self.editor.gizmo_drag_axis_index = None;
    }

    fn invalidate_geometry(&mut self) {
        // Many of the ~90 invalidation sites fire for editor-state reasons
        // (selection, tool changes) with the documents untouched; the
        // composite clone and snap index only need refreshing when the
        // workspace contents actually changed.
        let composite_key = self.workspace.composite_key();
        if Some(composite_key) != self.scene_document_key {
            self.scene_document = self.workspace.scene_document();
            self.scene_document_key = Some(composite_key);
            // The snap index rebuild is deferred to the next snap/orbit
            // query: many edits never snap before the next edit, and the
            // BVH build is the expensive part.
            self.snap_index_dirty = true;
        }
        if let Some(graphics) = self.graphics.as_mut() {
            graphics.invalidate_geometry();
        }
        self.redraw_requested = true;
    }

    /// Request a redraw for topology-only style/selection changes without
    /// rebuilding the document vector scene.
    ///
    /// Triangulations and block models render from their own per-item GPU
    /// caches (`triangulation_gpu` / `block_model_gpu`), which re-sync every
    /// frame with per-id dirty checks.
    fn request_topology_redraw(&mut self) {
        self.redraw_requested = true;
    }

    /// Request a topology redraw and refresh cached scene bounds, without
    /// rebuilding the document vector scene.
    fn invalidate_topology_bounds_and_redraw(&mut self) {
        if let Some(graphics) = self.graphics.as_mut() {
            graphics.invalidate_scene_bounds();
        }
        self.request_topology_redraw();
    }

    /// Rebuild the snap index from the current scene document if an edit
    /// invalidated it. Call before handing `self.snap_index` to a query.
    fn refresh_snap_index(&mut self) {
        if self.snap_index_dirty {
            self.snap_index = ObjectSnapIndex::build(self.scene_document.objects());
            self.snap_index_dirty = false;
        }
    }

    fn invalidate_overlay(&mut self) {
        if let Some(graphics) = self.graphics.as_mut() {
            graphics.invalidate_overlay();
        }
        self.redraw_requested = true;
    }

    pub(crate) fn begin_topology_load(&mut self) {
        self.pending_loads += 1;
        if let Some(window) = &self.window {
            window.set_cursor(CursorIcon::Progress);
        }
        self.redraw_requested = true;
    }

    pub(crate) fn finish_topology_load(&mut self) {
        self.topology_load_pending_gpu = false;
        if self.pending_loads == 0
            && let Some(window) = &self.window
        {
            window.set_cursor(CursorIcon::Default);
        }
    }

    fn fit_view_to_extents(&mut self) {
        if let Some(graphics) = self.graphics.as_mut() {
            graphics.fit_to_extents(
                &self.scene_document,
                &self.triangulations,
                &self.block_models,
                &self.editor.hidden_handles,
            );
            self.redraw_requested = true;
        }
    }

    fn teardown_window(&mut self) {
        self.graphics = None;
        self.window = None;
        self.pending_resize = None;
        self.last_render_time = None;
        self.redraw_requested = false;
    }

    fn project_view(&self) -> UiProjectView {
        let projects = self
            .workspace
            .projects
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let dirty_layers = p.dirty_layers();
                let dirty = p.dirty || !dirty_layers.is_empty();
                UiProjectEntry {
                    name: p.pidb.metadata.name.clone(),
                    dirty,
                    index: i,
                    is_active: self.workspace.active_index == Some(i),
                    path: p.path.clone(),
                    layers: p
                        .pidb
                        .document
                        .layers()
                        .iter()
                        .map(|l| {
                            let is_loaded = p.loaded_layers.contains(&l.id);
                            UiLayerEntry {
                                id: l.id,
                                name: l.name.clone(),
                                is_loaded,
                                dirty: dirty_layers.contains(&l.id),
                            }
                        })
                        .collect(),
                }
            })
            .collect();
        let loaded_by_path: BTreeMap<PathBuf, &OpenTriangulation> = self
            .triangulations
            .iter()
            .map(|tri| (tri.path.clone(), tri))
            .collect();
        let mut triangulations = Vec::new();
        // Individually-opened files — no group, shown flat and removable.
        let mut seen_paths: std::collections::BTreeSet<PathBuf> = std::collections::BTreeSet::new();
        for path in &self.triangulation_files {
            if seen_paths.contains(path) {
                continue;
            }
            seen_paths.insert(path.clone());
            if let Some(tri) = loaded_by_path.get(path) {
                triangulations.push(UiTriangulationEntry {
                    id: Some(tri.id),
                    name: tri.name.clone(),
                    visible: tri.visible,
                    is_active: self.active_triangulation == Some(tri.id),
                    is_loaded: true,
                    is_saved: tri.is_saved,
                    path: path.clone(),
                    group: None,
                });
            } else {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_owned();
                triangulations.push(UiTriangulationEntry {
                    id: None,
                    name,
                    visible: false,
                    is_active: false,
                    is_loaded: false,
                    is_saved: true,
                    path: path.clone(),
                    group: None,
                });
            }
        }
        // Directory-scanned files — grouped under their source dir.
        for dir in &self.triangulation_dirs {
            if let Some(entries) = self.triangulation_dir_entries.get(dir) {
                for path in entries {
                    if !seen_paths.contains(path) {
                        seen_paths.insert(path.clone());
                        if let Some(tri) = loaded_by_path.get(path) {
                            triangulations.push(UiTriangulationEntry {
                                id: Some(tri.id),
                                name: tri.name.clone(),
                                visible: tri.visible,
                                is_active: self.active_triangulation == Some(tri.id),
                                is_loaded: true,
                                is_saved: tri.is_saved,
                                path: path.clone(),
                                group: Some(dir.clone()),
                            });
                        } else {
                            let name = path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or("")
                                .to_owned();
                            triangulations.push(UiTriangulationEntry {
                                id: None,
                                name,
                                visible: false,
                                is_active: false,
                                is_loaded: false,
                                is_saved: true,
                                path: path.clone(),
                                group: Some(dir.clone()),
                            });
                        }
                    }
                }
            }
        }
        // Include in-memory (generated) triangulations not covered by any file/dir list.
        for tri in &self.triangulations {
            if !seen_paths.contains(&tri.path) {
                triangulations.push(UiTriangulationEntry {
                    id: Some(tri.id),
                    name: tri.name.clone(),
                    visible: tri.visible,
                    is_active: self.active_triangulation == Some(tri.id),
                    is_loaded: true,
                    is_saved: tri.is_saved,
                    path: tri.path.clone(),
                    group: None,
                });
            }
        }
        let loaded_block_by_bmf: BTreeMap<PathBuf, &OpenBlockModel> = self
            .block_models
            .iter()
            .map(|model| (model.source.bmf_path.clone(), model))
            .collect();
        let mut block_models = Vec::new();
        let mut seen_block_models = BTreeSet::new();
        for source in &self.block_model_files {
            if !seen_block_models.insert(source.bmf_path.clone()) {
                continue;
            }
            if let Some(model) = loaded_block_by_bmf.get(&source.bmf_path) {
                block_models.push(UiBlockModelEntry {
                    id: Some(model.id),
                    name: model.name.clone(),
                    source: model.source.clone(),
                    visible: model.visible,
                    is_active: self.active_block_model == Some(model.id),
                    is_loaded: true,
                    _block_count: model.renderable_block_indices.len(),
                    numeric_variable_count: model.model.numeric_variables().len(),
                });
            } else {
                let name = source
                    .bmf_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("")
                    .to_owned();
                block_models.push(UiBlockModelEntry {
                    id: None,
                    name,
                    source: source.clone(),
                    visible: false,
                    is_active: false,
                    is_loaded: false,
                    _block_count: 0,
                    numeric_variable_count: 0,
                });
            }
        }
        for model in &self.block_models {
            if !seen_block_models.contains(&model.source.bmf_path) {
                block_models.push(UiBlockModelEntry {
                    id: Some(model.id),
                    name: model.name.clone(),
                    source: model.source.clone(),
                    visible: model.visible,
                    is_active: self.active_block_model == Some(model.id),
                    is_loaded: true,
                    _block_count: model.renderable_block_indices.len(),
                    numeric_variable_count: model.model.numeric_variables().len(),
                });
            }
        }

        let active_path = self.workspace.active_project().and_then(|p| p.path.clone());
        let active_triangulation_for_menu = self.active_triangulation.and_then(|id| {
            self.triangulations
                .iter()
                .find(|tri| tri.id == id)
                .map(|tri| (tri.id, tri.color))
        });
        UiProjectView {
            projects,
            triangulations,
            block_models,
            active_index: self.workspace.active_index,
            needs_startup_dialog: self.workspace.projects.is_empty()
                && !self.startup_dialog_dismissed,
            active_path,
            active_triangulation_for_menu,
        }
    }

    /// Restore projects from a saved session. Called from `main` after startup.
    /// Non-existent paths are silently skipped. If no CLI arg activated a project,
    /// the session's active path determines which project becomes active.
    pub(crate) fn load_session_projects(&mut self, session: &crate::app::io::Session) {
        let had_active = self.workspace.has_active_project();
        for path in &session.project_paths {
            if !path.exists() {
                log::warn!(
                    "Session project no longer exists, skipping: {}",
                    path.display()
                );
                continue;
            }
            // Skip if already open (e.g. opened via CLI arg).
            if self
                .workspace
                .projects
                .iter()
                .any(|p| p.path.as_deref() == Some(path.as_path()))
            {
                continue;
            }
            match pidb::load(path)
                .and_then(|pidb| pidb::open_project(Some(path.clone()), pidb, false))
            {
                Ok(project) => {
                    self.workspace.add_inactive(project);
                }
                Err(e) => {
                    log::warn!("Failed to reopen session project {}: {e}", path.display());
                }
            }
        }
        // Sort projects alphabetically by name (stable sort keeps existing order for ties).
        if let Some(active_idx) = self.workspace.active_index {
            let active_path = self
                .workspace
                .projects
                .get(active_idx)
                .and_then(|p| p.path.clone());
            self.workspace
                .projects
                .sort_by(|a, b| a.pidb.metadata.name.cmp(&b.pidb.metadata.name));
            // Restore the active index after sort.
            self.workspace.active_index = active_path.as_deref().and_then(|ap| {
                self.workspace
                    .projects
                    .iter()
                    .position(|p| p.path.as_deref() == Some(ap))
            });
        } else {
            self.workspace
                .projects
                .sort_by(|a, b| a.pidb.metadata.name.cmp(&b.pidb.metadata.name));
        }
        // If nothing was active from a CLI arg, activate the session's saved active project.
        if !had_active {
            if let Some(active_path) = &session.active_path
                && let Some(i) = self
                    .workspace
                    .projects
                    .iter()
                    .position(|p| p.path.as_deref() == Some(active_path.as_path()))
            {
                self.workspace.active_index = Some(i);
            }
            // Fall back to first project if still nothing active.
            if !self.workspace.has_active_project() && !self.workspace.projects.is_empty() {
                self.workspace.active_index = Some(0);
            }
        }

        // Restore folder-scanned triangulation directories.
        self.triangulation_excluded_paths = session
            .triangulation_excluded_paths
            .iter()
            .cloned()
            .collect();
        for path in &session.triangulation_paths {
            if path.is_dir() && !self.triangulation_dirs.contains(path) {
                self.triangulation_dirs.push(path.clone());
            }
        }
        self.triangulation_dirs.sort();
        self.triangulation_dirs.dedup();
        self.refresh_triangulation_dir_entries();

        // Restore individually-opened triangulation files.
        for path in &session.triangulation_file_paths {
            if path.is_file()
                && !self.triangulation_excluded_paths.contains(path)
                && !self.triangulation_files.contains(path)
            {
                self.triangulation_files.push(path.clone());
            }
        }

        for source in &session.block_model_sources {
            if source.bmf_path.is_file() && !self.block_model_files.contains(source) {
                self.block_model_files.push(source.clone());
            }
        }
    }

    fn scan_triangulation_dir(dir: &Path) -> Vec<PathBuf> {
        let mut paths: Vec<_> = match fs::read_dir(dir) {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| MeshFormat::from_path(path).is_some())
                .collect(),
            Err(error) => {
                userspace_warn!(
                    "Could not scan triangulation folder {}: {error}",
                    dir.display()
                );
                Vec::new()
            }
        };
        paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
        paths
    }

    fn refresh_triangulation_dir_entries(&mut self) {
        self.triangulation_dir_entries = self
            .triangulation_dirs
            .iter()
            .map(|dir| {
                let files = Self::scan_triangulation_dir(dir)
                    .into_iter()
                    .filter(|path| !self.triangulation_excluded_paths.contains(path))
                    .collect();
                (dir.clone(), files)
            })
            .collect();
    }

    fn persist_session(&self) {
        let mut paths: Vec<std::path::PathBuf> = self
            .workspace
            .projects
            .iter()
            .filter_map(|p| p.path.clone())
            .collect();
        paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
        let active_path = self.workspace.active_project().and_then(|p| p.path.clone());
        let triangulation_paths: Vec<PathBuf> = {
            let deduped: BTreeSet<_> = self.triangulation_dirs.iter().cloned().collect();
            deduped.into_iter().collect()
        };
        let triangulation_file_paths: Vec<PathBuf> = {
            let deduped: BTreeSet<_> = self.triangulation_files.iter().cloned().collect();
            deduped.into_iter().collect()
        };
        let triangulation_excluded_paths: Vec<PathBuf> =
            self.triangulation_excluded_paths.iter().cloned().collect();
        let block_model_sources: Vec<BlockModelSource> = {
            let deduped: BTreeSet<_> = self.block_model_files.iter().cloned().collect();
            deduped.into_iter().collect()
        };
        let session = crate::app::io::Session {
            project_paths: paths,
            active_path,
            triangulation_paths,
            triangulation_file_paths,
            triangulation_excluded_paths,
            block_model_sources,
        };
        if let Err(e) = crate::app::io::save_session(&session) {
            log::warn!("Failed to save session: {e}");
        }
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title(crate::APP_NAME.to_string())
            .with_window_icon(window_icon())
            .with_min_inner_size(winit::dpi::PhysicalSize::new(900, 500))
            .with_inner_size(winit::dpi::PhysicalSize::new(900, 500))
            .with_maximized(true);
        #[cfg(target_os = "linux")]
        let window_attributes = window_attributes.with_name(crate::APP_ID, crate::APP_ID);
        #[cfg(target_os = "macos")]
        if let Err(error) = crate::startup::macos::set_dock_icon() {
            log::warn!("Failed to set macOS Dock icon: {error}");
        }

        let window = match event_loop.create_window(window_attributes) {
            Ok(window) => Arc::new(window),
            Err(e) => {
                log::error!("Failed to create window: {e}");
                self.close_requested = true;
                return;
            }
        };
        match pollster::block_on(Graphics::new(window.clone())) {
            Ok(graphics) => {
                self.graphics = Some(graphics);
                self.window = Some(window);
                self.redraw_requested = true;
                self.fit_view_to_extents();
            }
            Err(e) => {
                log::error!("Failed to initialize graphics: {e:?}");
                self.close_requested = true;
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        self.handle_window_event(event_loop, _window_id, event);
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event
            && let Some(graphics) = self.graphics.as_mut()
            && graphics.process_mouse_motion(delta.0, delta.1)
        {
            self.redraw_requested = true;
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.close_requested {
            self.teardown_window();
            event_loop.exit();
            return;
        }

        self.poll_file_dialogs();
        // Under egui_inspection automation the window may be occluded/unfocused, so it won't
        // receive the OS events a normal redraw relies on. Force the same continuous-redraw
        // path used for animations so inspection requests still get serviced.
        let continuous_redraw = inspection_polling_enabled()
            || self
                .graphics
                .as_ref()
                .is_some_and(Graphics::needs_continuous_redraw);

        if (self.redraw_requested || continuous_redraw)
            && let Some(window) = self.window.as_ref()
        {
            let frame_interval = if self.pending_resize.is_some() {
                rate_interval(self.editor.resize_frame_rate_cap)
            } else {
                rate_interval(self.editor.frame_rate_cap)
            };
            if let Some(last_render) = self.last_render_time {
                let deadline = last_render + frame_interval;
                if Instant::now() < deadline {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
                    return;
                }
            }
            self.redraw_requested = false;
            window.request_redraw();
        }

        if !self.pending_file_dialogs.is_empty() {
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(16),
            ));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.teardown_window();
    }
}

/// True when built with `--features inspection` and run with `EGUI_INSPECTION` set, i.e.
/// `egui_inspection::attach_from_env` actually attached.
fn inspection_polling_enabled() -> bool {
    cfg!(feature = "inspection") && std::env::var_os("EGUI_INSPECTION").is_some()
}

pub(crate) fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("project.pidb")
        .to_string()
}
