pub(crate) mod canvas; // Handles anything to do with dragging and stuff
pub(crate) mod commands; // Handles UI commands
pub(crate) mod events; // Handles window events
pub(crate) mod io; /* Handles session serialisation */
pub(crate) mod jobs; // Reusable background-compute job queue

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    hash::{DefaultHasher, Hash, Hasher},
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
    app::commands::file::PendingFileDialog,
    model::{
        Document, LayerId, Object, ObjectId, SceneEntityId,
        block_model::{BlockModelId, BlockModelSource, OpenBlockModel},
        formats::MeshFormat,
        pidb::{self, OpenProject, Workspace},
        raster::OpenRasterTexture,
        spatial::ObjectSnapIndex,
        triangulation::{OpenTriangulation, TriangulationId},
    },
    rendering::graphics::Graphics,
    ui::state::{
        EditorState, UiBlockModelEntry, UiLayerEntry, UiPointCloudEntry, UiProjectEntry,
        UiProjectView, UiTriangulationEntry,
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

/// A live Move preview belongs to the project whose objects were captured.
/// Keeping that identity with the originals prevents a later project switch
/// from committing or restoring the preview in a different document.
pub(crate) struct MoveSession {
    pub(crate) project_runtime_id: u32,
    pub(crate) originals: Vec<Object>,
}

/// Stable identity for one background operation. Every pending receiver owns
/// exactly one ticket, so cancellation/completion can settle only its own
/// progress state instead of decrementing a shared counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BackgroundTaskTicket(u64);

#[derive(Default)]
struct BackgroundTaskState {
    next_ticket: u64,
    cpu_pending: HashSet<BackgroundTaskTicket>,
    awaiting_apply: HashSet<BackgroundTaskTicket>,
    gpu_pending: HashSet<BackgroundTaskTicket>,
}

impl BackgroundTaskState {
    fn begin(&mut self) -> BackgroundTaskTicket {
        loop {
            let ticket = BackgroundTaskTicket(self.next_ticket);
            self.next_ticket = self.next_ticket.wrapping_add(1);
            if !self.awaiting_apply.contains(&ticket)
                && !self.gpu_pending.contains(&ticket)
                && self.cpu_pending.insert(ticket)
            {
                return ticket;
            }
        }
    }

    fn settle_cpu(&mut self, ticket: BackgroundTaskTicket, needs_gpu: bool) {
        if !self.cpu_pending.remove(&ticket) {
            debug_assert!(
                false,
                "unknown or double-completed background ticket {ticket:?}"
            );
            return;
        }
        let inserted = self.awaiting_apply.insert(ticket);
        debug_assert!(inserted, "background ticket was already awaiting apply");
        let removed = self.awaiting_apply.remove(&ticket);
        debug_assert!(removed, "background apply ticket disappeared");
        if needs_gpu {
            let inserted = self.gpu_pending.insert(ticket);
            debug_assert!(inserted, "background ticket was already pending GPU upload");
        }
    }

    fn cancel(&mut self, ticket: BackgroundTaskTicket) {
        let removed = self.cpu_pending.remove(&ticket)
            || self.awaiting_apply.remove(&ticket)
            || self.gpu_pending.remove(&ticket);
        debug_assert!(
            removed,
            "unknown or double-cancelled background ticket {ticket:?}"
        );
    }

    fn finish_gpu_uploads(&mut self) {
        self.gpu_pending.clear();
    }

    fn has_gpu_uploads(&self) -> bool {
        !self.gpu_pending.is_empty()
    }

    fn is_busy(&self) -> bool {
        !self.cpu_pending.is_empty()
            || !self.awaiting_apply.is_empty()
            || !self.gpu_pending.is_empty()
    }
}

pub(crate) struct App<'a> {
    close_requested: bool,
    /// Set when the renderer failed unrecoverably. Distinct from the ordinary
    /// `close_requested` flag: fatal shutdown first writes recovery copies of
    /// every dirty PIDB and waits for background writers to settle.
    fatal_shutdown: bool,
    /// Consecutive surface-validation failures survived via reconfigure.
    /// Reset by a successful frame; beyond the bound the failure is fatal.
    render_validation_recovery_attempts: u32,
    exit_after_pending_saves: bool,
    discard_changes_on_deferred_exit: bool,
    redraw_requested: bool,
    /// Wake deadline requested by egui (cursor blink, tooltip delay, etc.).
    /// Keeping it in the application event loop prevents timed repaints from
    /// being discarded when there is otherwise no window activity.
    next_ui_repaint_deadline: Option<Instant>,
    window: Option<Arc<Window>>,
    graphics: Option<Graphics<'a>>,
    /// Latest non-zero window size awaiting surface reconfiguration. Resize
    /// events arrive in bursts while dragging, so intermediate sizes are
    /// deliberately replaced instead of configuring a swapchain for each one.
    pending_resize: Option<winit::dpi::PhysicalSize<u32>>,
    last_render_time: Option<Instant>,
    last_scroll_instant: Option<Instant>,
    last_snap_poll_instant: Option<Instant>,
    last_road_preview_update_instant: Option<Instant>,
    editor: EditorState,
    workspace: Workspace,
    /// Explorer/menu snapshot reused while its allocation-free source
    /// fingerprint is unchanged.
    ui_project_view_cache: RefCell<Option<(u64, Arc<UiProjectView>)>>,
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
    point_clouds: Vec<crate::model::point_cloud::OpenPointCloud>,
    point_cloud_files: Vec<PathBuf>,
    next_point_cloud_id: u64,
    raster_textures: Vec<OpenRasterTexture>,
    raster_files: Vec<PathBuf>,
    next_raster_texture_id: u64,
    empty_document: Document,
    scene_document: Document,
    snap_index: ObjectSnapIndex,
    /// Set by `invalidate_geometry`; the index rebuilds lazily on the next
    /// snap/orbit query via `refresh_snap_index`.
    snap_index_dirty: bool,
    /// Ghost-free resolved road network for `scene_document`, used by
    /// SnapToLine road snapping. Rebuilt lazily via
    /// `refresh_scene_road_network`; cleared with the snap index.
    scene_road_network: Option<crate::model::road_network::ResolvedNetwork>,
    /// Ghost-free compromised-road keys of the active document, keyed by
    /// document revision. Grandfathers pre-existing rule violations during
    /// per-cursor-move ghost validation without rebuilding the ghost-free
    /// topology each tick.
    road_preexisting_cache: Option<(
        u32,
        u64,
        std::collections::HashSet<crate::model::road_network::RoadKey>,
    )>,
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
    move_session_original: Option<MoveSession>,
    background_tasks: BackgroundTaskState,
    pending_triangulation_loads: Vec<(
        BackgroundTaskTicket,
        PathBuf,
        mpsc::Receiver<anyhow::Result<crate::model::triangulation::LoadedTriangulation>>,
    )>,
    pending_block_model_loads: Vec<(
        BackgroundTaskTicket,
        BlockModelSource,
        mpsc::Receiver<anyhow::Result<crate::model::block_model::LoadedBlockModel>>,
    )>,
    pending_point_cloud_loads: Vec<(
        BackgroundTaskTicket,
        PathBuf,
        mpsc::Receiver<anyhow::Result<crate::model::point_cloud::LoadedPointCloud>>,
    )>,
    pending_raster_loads: Vec<(
        BackgroundTaskTicket,
        PathBuf,
        mpsc::Receiver<anyhow::Result<crate::model::raster::LoadedRasterTexture>>,
    )>,
    pub(crate) pending_file_dialogs: Vec<PendingFileDialog>,
    /// Triangulation saves/exports running on background threads; drained by
    /// `poll_saves` each frame.
    pending_saves: Vec<crate::app::commands::file::PendingSave>,
    /// Heavy compute jobs (include/cut/create) running on background threads;
    /// drained by `poll_jobs` each frame.
    pending_jobs: Vec<crate::app::jobs::BackgroundJob<'a>>,
    window_focused: bool,
}

impl<'a> App<'a> {
    pub(crate) fn new() -> Result<Self> {
        let mut app = Self {
            close_requested: false,
            fatal_shutdown: false,
            render_validation_recovery_attempts: 0,
            exit_after_pending_saves: false,
            discard_changes_on_deferred_exit: false,
            redraw_requested: false,
            next_ui_repaint_deadline: None,
            window: None,
            graphics: None,
            pending_resize: None,
            last_render_time: None,
            last_scroll_instant: None,
            last_snap_poll_instant: None,
            last_road_preview_update_instant: None,
            editor: EditorState::new(),
            workspace: Workspace::default(),
            ui_project_view_cache: RefCell::new(None),
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
            point_clouds: Vec::new(),
            point_cloud_files: Vec::new(),
            next_point_cloud_id: 0,
            raster_textures: Vec::new(),
            raster_files: Vec::new(),
            next_raster_texture_id: 0,
            empty_document: Document::new(),
            scene_document: Document::new(),
            snap_index: ObjectSnapIndex::default(),
            snap_index_dirty: false,
            scene_road_network: None,
            road_preexisting_cache: None,
            scene_document_key: None,
            history: crate::model::History::new(),
            modifiers: ModifiersState::empty(),
            drag: None,
            gizmo_drag: None,
            right_press_px: None,
            right_orbit_active: false,
            pending_topology_click: None,
            move_session_original: None,
            background_tasks: BackgroundTaskState::default(),
            pending_triangulation_loads: Vec::new(),
            pending_block_model_loads: Vec::new(),
            pending_point_cloud_loads: Vec::new(),
            pending_raster_loads: Vec::new(),
            pending_file_dialogs: Vec::new(),
            pending_saves: Vec::new(),
            pending_jobs: Vec::new(),
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
        app.editor.show_points = config.show_points;
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
        app.editor.debug_chunk_coloring = config.debug_chunk_coloring;
        app.editor.debug_clip_planes = config.debug_clip_planes;
        app.editor.plan_orbit_sensitivity = io::finite_clamped(
            config.plan_orbit_sensitivity,
            0.0001,
            0.02,
            io::default_plan_orbit_sensitivity(),
        );
        app.editor.plan_zoom_sensitivity = io::finite_clamped(
            config.plan_zoom_sensitivity,
            0.0001,
            0.05,
            io::default_plan_zoom_sensitivity(),
        );
        app.editor.plan_invert_vertical_look = config.plan_invert_vertical_look;
        app.editor.plan_invert_horizontal_look = config.plan_invert_horizontal_look;
        app.editor.plan_zoom_towards_cursor = config.plan_zoom_towards_cursor;
        app.editor.fly_field_of_view_degrees = io::finite_clamped(
            config.fly_field_of_view_degrees,
            20.0,
            120.0,
            io::default_fly_field_of_view_degrees(),
        );
        app.editor.fly_mouse_look_sensitivity = io::finite_clamped(
            config.fly_mouse_look_sensitivity,
            0.0001,
            0.02,
            io::default_fly_mouse_look_sensitivity(),
        );
        app.editor.fly_invert_vertical_look = config.fly_invert_vertical_look;
        app.editor.fly_invert_horizontal_look = config.fly_invert_horizontal_look;
        app.editor.fly_near_clip_limit = io::finite_clamped(
            config.fly_near_clip_limit,
            0.01,
            100.0,
            io::default_fly_near_clip_limit(),
        );
        app.editor.fly_max_clip_span = io::finite_clamped(
            config.fly_max_clip_span,
            100.0,
            1_000_000.0,
            io::default_fly_max_clip_span(),
        );
        app.configure_graphics_camera_preferences();

        Ok(app)
    }

    fn active_document(&self) -> &Document {
        self.workspace
            .active_document()
            .unwrap_or(&self.empty_document)
    }

    pub(crate) fn activate_project_for_object(&mut self, object_id: ObjectId) -> bool {
        let Some(index) = self.workspace.project_index_for_object(object_id) else {
            return false;
        };
        self.activate_project_index(index);
        true
    }

    pub(crate) fn activate_project_for_layer(&mut self, layer_id: LayerId) -> bool {
        let Some(index) = self.workspace.project_index_for_layer(layer_id) else {
            return false;
        };
        self.activate_project_index(index);
        true
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

    fn editing_ready(&self) -> bool {
        self.workspace.has_active_project() && !self.editor.fly_mode_enabled
    }

    fn set_active_project(&mut self, project: OpenProject) {
        let index = self.workspace.add_and_activate(project);
        self.clear_editor_transient_state();
        self.history
            .activate(self.workspace.projects[index].runtime_id);
        self.invalidate_geometry();
        self.persist_session();
    }

    fn activate_project_index(&mut self, index: usize) {
        if self.workspace.active_index == Some(index) {
            return;
        }
        let active_tool = self.editor.active_tool;
        self.clear_editor_transient_state();
        // Object interaction is allowed to retarget the current tool to a
        // different PIDB. Its project-specific preview state was cleared
        // above, but the chosen tool itself remains armed.
        self.editor.active_tool = active_tool;
        self.workspace.set_active_index(index);
        self.history
            .activate(self.workspace.projects[index].runtime_id);
        self.persist_session();
        self.invalidate_overlay();
    }

    fn clear_editor_transient_state(&mut self) {
        // Resolve document-backed drafts while their source identity is still
        // available. These helpers locate the owning project explicitly, so
        // this is also safe when a newly opened project has already become
        // active.
        if self.has_pending_move_delta() {
            self.restore_move_session_original();
        }
        if self.editor.text_editing_enabled {
            self.cancel_text_edit();
        }
        self.editor.clear_project_transients();
        self.pending_topology_click = None;
        // Clear any in-progress move session so it cannot bleed into the new PIDB.
        self.move_session_original = None;
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
            self.scene_road_network = None;
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
            self.snap_index = ObjectSnapIndex::build(&self.scene_document);
            self.snap_index_dirty = false;
        }
    }

    /// Rebuild the ghost-free resolved road network for the scene document if
    /// an edit invalidated it. Call before handing `self.scene_road_network`
    /// to a SnapToLine query so snapping never re-resolves per poll.
    fn refresh_scene_road_network(&mut self) {
        if self.scene_road_network.is_none() {
            self.scene_road_network = Some(crate::model::road_network::resolve(
                &self.scene_document,
                None,
            ));
        }
    }

    /// Ghost-free compromised-road keys for the active document, cached per
    /// `(project, document revision)` so per-cursor-move ghost validation
    /// skips the second topology build.
    fn road_preexisting_compromised(
        &mut self,
    ) -> std::collections::HashSet<crate::model::road_network::RoadKey> {
        let Some(project) = self.workspace.active_project() else {
            return Default::default();
        };
        let runtime_id = project.runtime_id;
        let document = &project.pidb.document;
        let revision = document.revision();
        if let Some((cached_runtime_id, cached_revision, keys)) = &self.road_preexisting_cache
            && *cached_runtime_id == runtime_id
            && *cached_revision == revision
        {
            return keys.clone();
        }
        let keys = crate::model::road_network::prepare(document, None).compromised_keys();
        self.road_preexisting_cache = Some((runtime_id, revision, keys.clone()));
        keys
    }

    fn invalidate_overlay(&mut self) {
        if let Some(graphics) = self.graphics.as_mut() {
            graphics.invalidate_overlay();
        }
        self.redraw_requested = true;
    }

    pub(crate) fn begin_topology_load(&mut self) -> BackgroundTaskTicket {
        let ticket = self.background_tasks.begin();
        self.update_background_task_cursor();
        self.redraw_requested = true;
        ticket
    }

    /// Move one CPU ticket through the UI-apply phase and either settle it or
    /// retain it until the renderer confirms its GPU upload is complete.
    pub(crate) fn finish_background_task(&mut self, ticket: BackgroundTaskTicket, needs_gpu: bool) {
        self.background_tasks.settle_cpu(ticket, needs_gpu);
        self.update_background_task_cursor();
    }

    pub(crate) fn cancel_background_task(&mut self, ticket: BackgroundTaskTicket) {
        self.background_tasks.cancel(ticket);
        self.update_background_task_cursor();
    }

    /// GPU-upload completion for the load pipeline: called after a render in
    /// which all renderer upload queues are empty.
    pub(crate) fn finish_topology_load(&mut self) {
        self.background_tasks.finish_gpu_uploads();
        self.update_background_task_cursor();
    }

    pub(crate) fn topology_uploads_pending(&self) -> bool {
        self.background_tasks.has_gpu_uploads()
    }

    pub(crate) fn background_tasks_pending(&self) -> bool {
        self.background_tasks.is_busy()
    }

    fn update_background_task_cursor(&self) {
        if let Some(window) = &self.window {
            window.set_cursor(if self.background_tasks.is_busy() {
                CursorIcon::Progress
            } else {
                CursorIcon::Default
            });
        }
    }

    fn fit_view_to_extents(&mut self) {
        if let Some(graphics) = self.graphics.as_mut() {
            graphics.fit_to_extents(
                &self.scene_document,
                &self.triangulations,
                &self.block_models,
                &self.point_clouds,
                &self.editor.hidden_handles,
            );
            self.redraw_requested = true;
        }
    }

    /// One definition of scene emptiness for load/apply and camera fitting.
    /// Evaluate immediately before installing a completed async result so
    /// concurrent loaders cannot all act on a stale start-time snapshot.
    pub(crate) fn scene_has_renderables(&self) -> bool {
        self.workspace.projects.iter().any(|project| {
            project
                .pidb
                .document
                .objects()
                .iter()
                .any(|object| project.loaded_layers.contains(&object.layer()))
        }) || !self.triangulations.is_empty()
            || !self.block_models.is_empty()
            || !self.point_clouds.is_empty()
            || !self.raster_textures.is_empty()
    }

    fn teardown_window(&mut self) {
        self.graphics = None;
        self.window = None;
        self.pending_resize = None;
        self.last_render_time = None;
        self.redraw_requested = false;
    }

    fn sync_slice_preview_window(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if !self.editor.slice_preview_detached
            || self
                .graphics
                .as_ref()
                .is_some_and(|graphics| graphics.slice_preview_window_id().is_some())
        {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title(format!("{} — Top-down slice preview", crate::APP_NAME))
            .with_window_icon(window_icon())
            .with_min_inner_size(winit::dpi::PhysicalSize::new(320, 240))
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 700));
        #[cfg(target_os = "linux")]
        let attributes = attributes.with_name(crate::APP_ID, crate::APP_ID);
        let result = event_loop
            .create_window(attributes)
            .map(Arc::new)
            .map_err(anyhow::Error::from)
            .and_then(|window| {
                self.graphics
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("renderer is not initialized"))?
                    .open_slice_preview(window)
            });
        if let Err(error) = result {
            log::error!("Failed to detach top-down preview: {error:#}");
            self.editor.slice_preview_detached = false;
        }
    }

    fn project_view_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.workspace.active_index.hash(&mut hasher);
        self.startup_dialog_dismissed.hash(&mut hasher);
        for project in &self.workspace.projects {
            project.runtime_id.hash(&mut hasher);
            project.path.hash(&mut hasher);
            project.pidb.metadata.name.hash(&mut hasher);
            project.has_unsaved_changes().hash(&mut hasher);
            let mut loaded_layers: Vec<_> = project.loaded_layers.iter().copied().collect();
            loaded_layers.sort_unstable_by_key(|layer| layer.0);
            loaded_layers.hash(&mut hasher);
            for layer in project.pidb.document.layers() {
                layer.id.hash(&mut hasher);
                layer.name.hash(&mut hasher);
            }
        }

        self.triangulation_files.hash(&mut hasher);
        self.triangulation_dirs.hash(&mut hasher);
        self.triangulation_dir_entries.hash(&mut hasher);
        self.active_triangulation.hash(&mut hasher);
        for triangulation in &self.triangulations {
            triangulation.id.hash(&mut hasher);
            triangulation.name.hash(&mut hasher);
            triangulation.path.hash(&mut hasher);
            (triangulation.visible
                && !self
                    .editor
                    .hidden_handles
                    .contains(&triangulation.entity_id()))
            .hash(&mut hasher);
            triangulation.is_saved.hash(&mut hasher);
            triangulation.raster_texture.hash(&mut hasher);
            triangulation.color.map(f32::to_bits).hash(&mut hasher);
        }

        self.block_model_files.hash(&mut hasher);
        self.active_block_model.hash(&mut hasher);
        for model in &self.block_models {
            model.id.hash(&mut hasher);
            model.name.hash(&mut hasher);
            model.source.hash(&mut hasher);
            model.visible.hash(&mut hasher);
            model.renderable_block_indices.len().hash(&mut hasher);
            model.model.numeric_variables().len().hash(&mut hasher);
        }

        self.point_cloud_files.hash(&mut hasher);
        for cloud in &self.point_clouds {
            cloud.id.hash(&mut hasher);
            cloud.name.hash(&mut hasher);
            cloud.path.hash(&mut hasher);
            cloud.visible.hash(&mut hasher);
            cloud.points.len().hash(&mut hasher);
        }

        self.raster_files.hash(&mut hasher);
        for raster in &self.raster_textures {
            raster.id.hash(&mut hasher);
            raster.name.hash(&mut hasher);
            raster.path.hash(&mut hasher);
            raster.source_size.hash(&mut hasher);
            raster.driver_name.hash(&mut hasher);
            raster.projection.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn project_view(&self) -> Arc<UiProjectView> {
        let key = self.project_view_key();
        if let Some((cached_key, view)) = self.ui_project_view_cache.borrow().as_ref()
            && *cached_key == key
        {
            return Arc::clone(view);
        }
        let projects: Vec<UiProjectEntry> = self
            .workspace
            .projects
            .iter()
            .enumerate()
            .map(|(index, project)| UiProjectEntry {
                runtime_id: project.runtime_id,
                name: project.pidb.metadata.name.clone(),
                dirty: project.has_unsaved_changes(),
                is_active: self.workspace.active_index == Some(index),
                path: project.path.clone(),
                layers: project
                    .pidb
                    .document
                    .layers()
                    .iter()
                    .map(|layer| UiLayerEntry {
                        id: layer.id,
                        name: layer.name.clone(),
                        is_loaded: project.loaded_layers.contains(&layer.id),
                    })
                    .collect(),
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
                    visible: tri.visible && !self.editor.hidden_handles.contains(&tri.entity_id()),
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
                                visible: tri.visible
                                    && !self.editor.hidden_handles.contains(&tri.entity_id()),
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
                    visible: tri.visible && !self.editor.hidden_handles.contains(&tri.entity_id()),
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

        let loaded_cloud_by_path: BTreeMap<PathBuf, &crate::model::point_cloud::OpenPointCloud> =
            self.point_clouds
                .iter()
                .map(|cloud| (cloud.path.clone(), cloud))
                .collect();
        let mut point_clouds = Vec::new();
        let mut seen_point_clouds = BTreeSet::new();
        for path in &self.point_cloud_files {
            if !seen_point_clouds.insert(path.clone()) {
                continue;
            }
            if let Some(cloud) = loaded_cloud_by_path.get(path) {
                point_clouds.push(UiPointCloudEntry {
                    id: Some(cloud.id),
                    name: cloud.name.clone(),
                    path: path.clone(),
                    visible: cloud.visible,
                    is_loaded: true,
                    point_count: cloud.points.len(),
                });
            } else {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("")
                    .to_owned();
                point_clouds.push(UiPointCloudEntry {
                    id: None,
                    name,
                    path: path.clone(),
                    visible: false,
                    is_loaded: false,
                    point_count: 0,
                });
            }
        }
        for cloud in &self.point_clouds {
            if !seen_point_clouds.contains(&cloud.path) {
                point_clouds.push(UiPointCloudEntry {
                    id: Some(cloud.id),
                    name: cloud.name.clone(),
                    path: cloud.path.clone(),
                    visible: cloud.visible,
                    is_loaded: true,
                    point_count: cloud.points.len(),
                });
            }
        }

        let loaded_raster_by_path: BTreeMap<PathBuf, &OpenRasterTexture> = self
            .raster_textures
            .iter()
            .map(|raster| (raster.path.clone(), raster))
            .collect();
        // Highlight only rasters a loaded triangulation is actually using;
        // dormant session assignments waiting to be restored don't count.
        let draped_raster_ids: BTreeSet<_> = self
            .triangulations
            .iter()
            .filter_map(|triangulation| triangulation.raster_texture)
            .collect();
        let mut raster_textures = Vec::new();
        let mut seen_rasters = BTreeSet::new();
        for path in &self.raster_files {
            if !seen_rasters.insert(path.clone()) {
                continue;
            }
            if let Some(raster) = loaded_raster_by_path.get(path) {
                raster_textures.push(crate::ui::state::UiRasterTextureEntry {
                    name: raster.name.clone(),
                    path: path.clone(),
                    is_loaded: true,
                    is_draped: draped_raster_ids.contains(&raster.id),
                    source_size: raster.source_size,
                    driver_name: raster.driver_name.clone(),
                    projection: raster.projection.clone(),
                });
            } else {
                raster_textures.push(crate::ui::state::UiRasterTextureEntry {
                    name: path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("")
                        .to_owned(),
                    path: path.clone(),
                    is_loaded: false,
                    is_draped: false,
                    source_size: [0, 0],
                    driver_name: String::new(),
                    projection: String::new(),
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
        let view = Arc::new(UiProjectView {
            projects,
            triangulations,
            block_models,
            point_clouds,
            raster_textures,
            has_active_project: self.workspace.has_active_project(),
            needs_startup_dialog: !self.workspace.has_active_project()
                && !self.startup_dialog_dismissed,
            active_path,
            active_triangulation_for_menu,
        });
        *self.ui_project_view_cache.borrow_mut() = Some((key, Arc::clone(&view)));
        view
    }

    /// Restore projects from a saved session. Called from `main` after startup.
    /// Non-existent or invalid paths are skipped independently. All restored
    /// PIDBs are parsed and kept open; the saved active path restores only the
    /// editing target.
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
            if self.workspace.project_index_for_path(path).is_some() {
                continue;
            }
            match pidb::load(path).and_then(|pidb| pidb::open_project(Some(path.clone()), pidb)) {
                Ok(project) => {
                    self.workspace.add_inactive(project);
                }
                Err(error) => {
                    log::warn!(
                        "Failed to reopen session project {}: {error}",
                        path.display()
                    );
                }
            }
        }
        if !had_active
            && let Some(active_path) = &session.active_path
            && let Some(index) = self.workspace.project_index_for_path(active_path)
        {
            self.workspace.set_active_index(index);
        }
        if let Some(project) = self.workspace.active_project() {
            self.history.activate(project.runtime_id);
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

        for path in &session.point_cloud_file_paths {
            if path.is_file() && !self.point_cloud_files.contains(path) {
                self.point_cloud_files.push(path.clone());
            }
        }
        for path in &session.raster_file_paths {
            if path.is_file() && !self.raster_files.contains(path) {
                self.raster_files.push(path.clone());
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
            .filter_map(|project| project.path.clone())
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
        let point_cloud_file_paths: Vec<PathBuf> = {
            let deduped: BTreeSet<_> = self.point_cloud_files.iter().cloned().collect();
            deduped.into_iter().collect()
        };
        let raster_file_paths: Vec<PathBuf> = {
            let deduped: BTreeSet<_> = self.raster_files.iter().cloned().collect();
            deduped.into_iter().collect()
        };
        let session = crate::app::io::Session {
            project_paths: paths,
            active_path,
            triangulation_paths,
            triangulation_file_paths,
            triangulation_excluded_paths,
            block_model_sources,
            point_cloud_file_paths,
            raster_file_paths,
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
        if let Err(error) = crate::os::macos::set_dock_icon() {
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
        // Background writers must be observed before honoring an exit request;
        // otherwise a completed-but-unpolled export can be terminated here.
        self.poll_saves();
        if self.fatal_shutdown {
            // The renderer is unusable, recovery copies have already been
            // written; exit as soon as atomic background writers settle so an
            // active export is not terminated mid-write.
            if self.pending_saves.is_empty() {
                self.teardown_window();
                event_loop.exit();
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(
                    Instant::now() + Duration::from_millis(16),
                ));
            }
            return;
        }
        if self.close_requested {
            self.teardown_window();
            event_loop.exit();
            return;
        }

        self.poll_file_dialogs();
        let now = Instant::now();
        if self
            .next_ui_repaint_deadline
            .is_some_and(|deadline| deadline <= now)
        {
            self.next_ui_repaint_deadline = None;
            self.redraw_requested = true;
        }
        let continuous_redraw = self
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
                if now < deadline {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
                    return;
                }
            }
            self.redraw_requested = false;
            window.request_redraw();
        }

        let task_poll_deadline = (!self.pending_file_dialogs.is_empty()
            || !self.pending_saves.is_empty()
            || !self.pending_jobs.is_empty())
        .then(|| now + Duration::from_millis(16));
        let wake_deadline = match (task_poll_deadline, self.next_ui_repaint_deadline) {
            (Some(task), Some(ui)) => Some(task.min(ui)),
            (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
            (None, None) => None,
        };
        if let Some(deadline) = wake_deadline {
            event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.teardown_window();
    }
}

pub(crate) fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("project.pidb")
        .to_string()
}
