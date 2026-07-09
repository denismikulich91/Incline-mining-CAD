//! Editor state, UI commands, and project view data types.
//!
//! `EditorState` is the central mutable state struct shared between the
//! rendering pipeline and every UI draw call.  `UiCommand` carries actions
//! back to the application core; `UiProjectView` is a flattened snapshot
//! of the project tree built each frame by the app layer.

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use glam::DVec3;
use strum::{Display, EnumIter};

use crate::{
    model::{
        FillStyle, LayerId, ObjectColor, ObjectId, RoadShape, SceneEntityId,
        block_model::{BlockModelId, BlockModelSource, ColorStop, FIRST_CUSTOM_COLOR_STOP_ID},
        formats::MeshFormat,
        point_cloud::PointCloudId,
        triangulation::TriangulationId,
    },
    userspace_log,
};

type OptionalScreenPointPx = Option<(f32, f32)>;

/// Unsaved preference values currently being edited in the Preferences tab.
///
/// When the user clicks "Save Changes" these values are applied to `EditorState`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PreferencesDraft {
    pub(crate) renderer_background_color: [f32; 4],
    pub(crate) topology_wireframes_enabled: bool,
    pub(crate) dark_mode: bool,
    pub(crate) show_console: bool,
    pub(crate) show_world_axis_gizmo: bool,
    pub(crate) snap_poll_rate: u32,
    pub(crate) frame_rate_cap: u32,
    pub(crate) resize_frame_rate_cap: u32,
    pub(crate) block_model_interaction_resolution_divisor: u32,
    pub(crate) frame_counter_enabled: bool,
    pub(crate) debug_chunk_coloring: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MoveToLayerDialog {
    pub(crate) object_ids: Vec<ObjectId>,
    pub(crate) target_layer: Option<LayerId>,
    pub(crate) copy: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MoveLayerDialog {
    pub(crate) source_project_index: usize,
    pub(crate) layer_id: LayerId,
    pub(crate) target_project_index: Option<usize>,
}

impl Default for PreferencesDraft {
    fn default() -> Self {
        Self {
            renderer_background_color: crate::app::io::default_renderer_background_color(),
            topology_wireframes_enabled: false,
            dark_mode: false,
            show_console: true,
            show_world_axis_gizmo: crate::app::io::default_show_world_axis_gizmo(),
            snap_poll_rate: crate::app::io::default_snap_poll_rate(),
            frame_rate_cap: crate::app::io::default_frame_rate_cap(),
            resize_frame_rate_cap: crate::app::io::default_resize_frame_rate_cap(),
            block_model_interaction_resolution_divisor:
                crate::app::io::default_block_model_interaction_resolution_divisor(),
            frame_counter_enabled: false,
            debug_chunk_coloring: false,
        }
    }
}

impl EditorState {
    fn clear_selection(&mut self) {
        self.selected_handles.clear();
    }

    fn replace_selection(&mut self, handle: SceneEntityId) {
        self.selected_handles.clear();
        self.selected_handles.insert(handle);
    }

    fn add_selection(&mut self, handle: SceneEntityId) {
        self.selected_handles.insert(handle);
    }

    fn toggle_selection(&mut self, handle: SceneEntityId) {
        if !self.selected_handles.remove(&handle) {
            self.selected_handles.insert(handle);
        }
    }
}

/// What the user-entered value means for an angled offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum OffsetMeasure {
    /// Direct horizontal (XY-plane) distance.
    Distance,
    /// The Z component (height).
    Height(HeightMode),
    /// Same as Distance — kept for naming clarity in the UI.
    Width,
}

/// Whether a height value is a relative delta or an absolute reduced level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum HeightMode {
    Relative,
    AbsoluteRL,
}

/// Surface/solid type for a created triangulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TriSurfaceType {
    /// Plain open surface
    Surface,
    /// Fully closed solid
    SolidClosed,
}

/// Which side of a reference triangulation to remove.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TriSurfaceCutSide {
    /// Remove geometry above the reference topology; keep geometry below it.
    CutTop,
    /// Remove geometry below the reference topology; keep geometry above it.
    CutBottom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OreFilterMode {
    GreaterOrEqual,
    LessOrEqual,
    Between,
}

/// Phase of the Create Triangulation workflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TriCreatePhase {
    MainDialog,
}

/// Source object type accepted by the Create Triangulation picker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TriCreateSource {
    Polygons,
    Roads,
}

/// A failed Create Triangulation run, retained for the failure dialog.
#[derive(Clone, Debug)]
pub(crate) struct TriCreateFailure {
    pub(crate) message: String,
    pub(crate) name: String,
    pub(crate) object_ids: Vec<ObjectId>,
    pub(crate) surface_type: TriSurfaceType,
    /// True when welding endpoints at the coarse tolerance would change the
    /// input, i.e. a retry is worth offering.
    pub(crate) weld_retry_available: bool,
}

/// Whether the Batter Berm tool is generating a pit or a stockpile.
/// Pit: each iteration goes inward and downward. Stockpile: outward and upward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum BatterBermMode {
    Pit,
    Stockpile,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BatterBermPreviewKey {
    pub(crate) target_id: ObjectId,
    pub(crate) document_revision: u64,
    pub(crate) width: f64,
    pub(crate) angle: f64,
    pub(crate) bench_height: f64,
    pub(crate) benches: u32,
    pub(crate) mode: BatterBermMode,
}

/// Which sub-mode the Relimit Line tool is operating in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RelimitMode {
    Intersect,
    AbsoluteLength,
    RelativeLength,
}

/// Which endpoint of a line the Relimit tool will move.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TrimEnd {
    Start,
    End,
}

/// Named orientations selectable from the orientation gizmo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StandardView {
    Up,
    Down,
    North,
    South,
    West,
    East,
}

/// One candidate relimit operation: move `end` of the source line to `target`.
/// The user selects between candidates by hovering near each one's `handle_px`
/// (the screen-space midpoint of the change it produces).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RelimitCandidate {
    /// Which source endpoint this operation moves.
    pub(crate) end: TrimEnd,
    /// World-space destination for that endpoint.
    pub(crate) target: DVec3,
    /// True if the line grows (extend, yellow); false if it shrinks (trim, red).
    pub(crate) is_extension: bool,
    /// Screen-space hover handle (physical px) = midpoint of moving-endpoint → target.
    pub(crate) handle_px: (f32, f32),
}

/// One segment in a Fuse-into-polygon chain.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FuseSegment {
    pub(crate) object_id: ObjectId,
    /// When true, the segment's vertices are consumed end→start instead of start→end.
    pub(crate) reversed: bool,
    /// Vertex at which a closed polygon is opened for insertion into the chain.
    pub(crate) start_index: usize,
    pub(crate) closed: bool,
    /// When true, the segment's first ordered vertex was designated as the
    /// join with the chain tail and sits (visually) on it, so commit drops it
    /// instead of keeping a doubled point / micro edge.
    pub(crate) weld_start: bool,
}

/// A transient message shown in the bottom status bar. Generic shape so any
/// background task (save, contour generation, etc.) can drive the same UI slot
/// without per-task plumbing.
#[derive(Clone, Debug)]
pub(crate) struct StatusBarMessage {
    pub(crate) text: String,
    /// `Some(f)` in `0.0..=1.0` shows `(percent%)` after the text; `None` is indeterminate.
    pub(crate) progress: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BermAngleMeasurement {
    pub(crate) angle_degrees: f64,
    pub(crate) projection: DVec3,
}

pub(crate) fn berm_angle_measurement(points: &[DVec3]) -> Option<BermAngleMeasurement> {
    let [a, b, c] = points.get(..3)? else {
        return None;
    };
    let ab_xy = b.truncate() - a.truncate();
    let ab_len_sq = ab_xy.length_squared();
    if ab_len_sq <= 1.0e-12 {
        return None;
    }

    let ac_xy = c.truncate() - a.truncate();
    let t = ac_xy.dot(ab_xy) / ab_len_sq;
    let projection_xy = a.truncate() + ab_xy * t;
    let projection_z = a.z + (b.z - a.z) * t;
    let projection = DVec3::new(projection_xy.x, projection_xy.y, projection_z);
    let horizontal = c.truncate().distance(projection_xy);
    let vertical = (c.z - projection_z).abs();
    if horizontal <= 1.0e-9 && vertical <= 1.0e-9 {
        return None;
    }

    Some(BermAngleMeasurement {
        angle_degrees: vertical.atan2(horizontal).to_degrees(),
        projection,
    })
}

/// Central mutable editor state.
///
/// Shared between the render pipeline and every UI draw call. Fields are grouped
/// into logical sections below for navigability.
pub(crate) struct EditorState {
    // Selection & visibility
    pub(crate) selected_handles: HashSet<SceneEntityId>,
    /// Entities removed from view (skipped by the renderer).
    pub(crate) hidden_handles: HashSet<SceneEntityId>,
    /// Entities frozen: still visible, but excluded from editing and snapping.
    pub(crate) frozen_handles: HashSet<SceneEntityId>,
    /// Entities dimmed toward the background colour.
    pub(crate) translucent_handles: HashSet<SceneEntityId>,
    /// Show wireframes on all topology meshes. Selected topology meshes always
    /// show a highlighted wireframe independently of this preference.
    pub(crate) topology_wireframes_enabled: bool,
    /// Use dark UI visuals and icons instead of the default light theme.
    pub(crate) dark_mode: bool,
    /// Show the console underneath the bottom toolbar.
    pub(crate) show_console: bool,
    /// Show the world-space axis gizmo in the top-right of the viewport.
    pub(crate) show_world_axis_gizmo: bool,
    /// Linear RGBA clear colour used behind the rendered scene.
    pub(crate) renderer_background_color: [f32; 4],
    /// Unsaved values currently being edited in the Preferences tab.
    pub(crate) preferences_draft: Option<PreferencesDraft>,
    pub(crate) snap_poll_rate: u32,
    pub(crate) frame_rate_cap: u32,
    pub(crate) resize_frame_rate_cap: u32,
    pub(crate) block_model_interaction_resolution_divisor: u32,
    pub(crate) frame_counter_enabled: bool,
    pub(crate) measured_fps: Option<f32>,
    /// Developer view: colour each surface chunk distinctly to visualise the
    /// Morton spatial chunking (and drive the chunk-cull stats readout).
    pub(crate) debug_chunk_coloring: bool,
    /// `(rendered, total)` surface chunks from the last frame; shown in the
    /// status bar while `debug_chunk_coloring` is on.
    pub(crate) debug_chunk_stats: Option<(u32, u32)>,
    /// Transient status-bar message from a background task (e.g. "Saving to …").
    /// `None` shows nothing; the field is updated from `poll_saves` each frame.
    pub(crate) status_message: Option<StatusBarMessage>,
    pub(crate) active_tool: ActiveTool,
    pub(crate) cursor_mode: CursorMode,
    pub(crate) tool_line_color: [f32; 4],
    pub(crate) tool_line_weight: f32,
    pub(crate) tool_hatch: ToolHatch,
    /// Active drawing layer, if any.
    pub(crate) active_layer: Option<LayerId>,
    /// Live world coordinate under the cursor (z on the active pick plane).
    pub(crate) cursor_world: Option<DVec3>,
    pub(crate) new_layer_dialog_open: bool,
    pub(crate) new_layer_name: String,
    pub(crate) new_layer_project_index: Option<usize>,
    /// Active layer rename: (project_index, layer_id, name_buffer).
    pub(crate) renaming_layer: Option<(usize, LayerId, String)>,
    /// Layer awaiting destructive deletion confirmation:
    /// (project_index, layer_id, display_name).
    pub(crate) pending_delete_layer: Option<(usize, LayerId, String)>,
    /// Vertices accumulated for an in-progress MakeLine / MakePoly stroke.
    pub(crate) pending_stroke: Vec<DVec3>,
    pub(crate) measurement_start: Option<DVec3>,
    pub(crate) measurement_end: Option<DVec3>,
    pub(crate) berm_angle_points: Vec<DVec3>,

    // Text editing
    /// Chars accumulated for an in-progress MakeText.
    pub(crate) pending_text: String,
    pub(crate) pending_text_height: f64,
    pub(crate) pending_text_rotation_degrees: f64,
    pub(crate) text_edit_dialog_px: Option<(f32, f32)>,
    /// Keep the text menu anchored while egui initializes its window state.
    pub(crate) text_edit_position_frames: u8,
    pub(crate) text_edit_focus_requested: bool,
    pub(crate) text_edit_created: bool,
    /// Dirty state of the active project before the new-text AddObject was committed.
    /// Whether the text properties popup is open.
    pub(crate) text_editing_enabled: bool,
    /// The ObjectId of the actively edited text label.
    pub(crate) editing_labels_id: Option<ObjectId>,

    // Cursor & snapping
    /// Z plane used for all placement operations (point, line, poly vertices).
    pub(crate) z_level: f64,
    /// Editable Z level value used by toolbar and Set Selection Z.
    pub(crate) z_input: f64,
    /// True when the current `cursor_world` is a snapped position (not raw ray).
    pub(crate) cursor_snapped: bool,
    /// Physical-pixel cursor position, updated on every CursorMoved event.
    pub(crate) cursor_screen_px: Option<(f32, f32)>,

    // Dialog state (position snapshots)
    /// When true, show the polygon finish dialog near the cursor.
    pub(crate) poly_finish_dialog: bool,
    /// Screen position (physical px) where the polygon finish dialog was opened.
    /// Snapshotted once so the dialog doesn't follow the cursor.
    pub(crate) poly_finish_dialog_px: Option<(f32, f32)>,
    /// When true, show the canvas right-click context menu.
    pub(crate) canvas_context_menu_open: bool,
    /// Physical-pixel position where the canvas context menu was opened.
    pub(crate) canvas_context_menu_px: Option<(f32, f32)>,
    pub(crate) move_to_layer_dialog: Option<MoveToLayerDialog>,
    pub(crate) set_selection_z_dialog: Option<crate::ui::dialogs::SetSelectionZDialog>,
    pub(crate) move_layer_dialog: Option<MoveLayerDialog>,

    // Display overrides
    pub(crate) xray_enabled: bool,
    pub(crate) vertical_exaggeration_dialog_open: bool,
    pub(crate) vertical_exaggeration: f64,
    pub(crate) vertical_exaggeration_input: f64,
    pub(crate) fly_mode_enabled: bool, // Not sure if this belongs here

    // Selection box
    /// Physical-pixel bounds of an in-progress box selection.
    pub(crate) selection_box_start_px: Option<(f32, f32)>,
    pub(crate) selection_box_current_px: Option<(f32, f32)>,

    // Offset Element tool
    pub(crate) offset_dialog_open: bool,
    pub(crate) offset_target_id: Option<ObjectId>,
    pub(crate) offset_target_ids: Vec<ObjectId>,
    pub(crate) offset_angle_degrees: f64,
    pub(crate) offset_measure: OffsetMeasure,
    pub(crate) offset_value_input: f64,
    /// Phase 2: dialog closed, waiting for canvas click to pick side.
    pub(crate) offset_awaiting_side_pick: bool,
    /// Absolute horizontal offset distance (sign determined by cursor position).
    pub(crate) offset_horiz_dist: f64,
    /// Z shift to apply to all vertices (0 for horizontal, ±height for batter).
    pub(crate) offset_z_delta: f64,
    /// When set, overrides `offset_horiz_dist`/`offset_z_delta`: project each vertex
    /// individually (by its own elevation) along `tan(angle)` so the whole result
    /// lands flat at `target_rl`. Fields are `(tan_angle, target_rl)`.
    pub(crate) offset_project_to_rl: Option<(f64, f64)>,
    /// Clamp offset vertices to the first visible triangulation they hit along
    /// the requested offset vector.
    pub(crate) offset_collide_with_triangulation: bool,
    /// Preview polygon vertices in world coordinates.
    pub(crate) offset_preview_world: Vec<DVec3>,
    /// Source vertices matching `offset_preview_world`, used for offset guide connectors.
    pub(crate) offset_source_world: Vec<DVec3>,
    /// Preview polygon vertices projected to physical-pixel screen coordinates this frame.
    pub(crate) offset_preview_screen_px: Vec<(f32, f32)>,
    /// Source vertices projected to physical-pixel screen coordinates this frame.
    pub(crate) offset_source_screen_px: Vec<(f32, f32)>,
    /// Preview screen ranges as `(start, end, closed)` so multiple offset
    /// previews are drawn independently.
    pub(crate) offset_preview_ranges: Vec<(usize, usize, bool)>,
    /// Whether the preview geometry is closed (polygon) or open (polyline).
    pub(crate) offset_preview_closed: bool,

    // Relimit Line tool
    pub(crate) relimit_dialog_open: bool,
    pub(crate) relimit_source_id: Option<ObjectId>,
    pub(crate) relimit_mode: RelimitMode,
    pub(crate) relimit_value_input: f64,
    /// Phase 0: tool active, no source selected yet — waiting for canvas click to pick source.
    pub(crate) relimit_awaiting_source_pick: bool,
    /// Phase 1: source selected, dialog closed — waiting for canvas click to pick target line.
    pub(crate) relimit_waiting_for_pick: bool,
    /// Phase 2: intersection computed, user confirms which end to move.
    pub(crate) relimit_confirming_end: bool,
    pub(crate) relimit_second_id: Option<ObjectId>,
    /// All valid relimit operations for the current source/target pair. The user
    /// hovers near each candidate's `handle_px` to select one.
    pub(crate) relimit_candidates: Vec<RelimitCandidate>,
    /// Currently active intersection (set from hover zone, used for commit).
    pub(crate) relimit_intersection_3d: Option<DVec3>,
    pub(crate) relimit_hover_end: TrimEnd,
    /// Phase 1 — hovered target line projected to physical-pixel screen coords (for yellow
    /// highlight).
    pub(crate) relimit_hover_target_id: Option<ObjectId>,
    pub(crate) relimit_hover_target_screen_px: Vec<(f32, f32)>,
    pub(crate) relimit_hover_target_closed: bool,
    /// Phase 2 — preview segment: moving endpoint → intersection, yellow=extension red=reduction.
    pub(crate) relimit_preview_from_px: Option<(f32, f32)>,
    pub(crate) relimit_preview_to_px: Option<(f32, f32)>,
    pub(crate) relimit_preview_is_extension: bool,
    /// Which end to move in AbsoluteLength / RelativeLength modes (default End).
    pub(crate) relimit_resize_end: TrimEnd,
    /// Screen position of the chosen resize endpoint sphere indicator (raw pixels).
    pub(crate) relimit_resize_end_px: Option<(f32, f32)>,

    // Fuse Into Polygon tool
    pub(crate) fuse_segments: Vec<FuseSegment>,
    /// Line that has been picked but is waiting for the user to click one of its endpoint markers.
    pub(crate) fuse_awaiting_endpoint: Option<ObjectId>,
    /// Selectable vertices for `fuse_awaiting_endpoint`. Open lines expose their
    /// two endpoints; closed polygons expose every vertex.
    pub(crate) fuse_endpoint_markers: Vec<(usize, DVec3)>,
    /// The open tail of the current chain — where the next segment will attach.
    pub(crate) fuse_chain_tail: Option<DVec3>,
    /// Opposite endpoint offered to close a single open source polyline.
    pub(crate) fuse_close_marker: Option<DVec3>,
    // Split At Points tool
    pub(crate) split_poly_id: Option<ObjectId>,
    pub(crate) split_selected_verts: [Option<usize>; 2],
    pub(crate) split_poly_verts_screen_px: Vec<(f32, f32)>,

    // Chamfer tool
    pub(crate) chamfer_radius: f64,
    pub(crate) chamfer_segments: u32,
    /// Maximum chamfer radius for the currently selected corner (f64::MAX = unlimited).
    pub(crate) chamfer_max_radius: f64,
    /// Which closed polygon is being chamfered (set on corner click).
    pub(crate) chamfer_poly_id: Option<ObjectId>,
    /// Which vertex index of `chamfer_poly_id` is the selected corner.
    pub(crate) chamfer_corner_index: Option<usize>,
    /// Preview of chamfered polygon projected to screen (raw pixels). Computed each frame by
    /// render.rs.
    pub(crate) chamfer_preview_screen_px: Vec<(f32, f32)>,
    /// Screen position of the nearest hoverable chamfer corner (raw pixels, before a corner is
    /// picked).
    pub(crate) chamfer_hover_corner_px: Option<(f32, f32)>,
    /// Screen position of the delete-tool polygon vertex marker (raw pixels).
    pub(crate) delete_hover_vertex_px: Option<(f32, f32)>,
    /// Screen position of the first chamfer corner vertex (raw pixels).
    pub(crate) chamfer_gizmo_corner_px: Option<(f32, f32)>,
    /// Screen direction of the gizmo bisector (used when radius=0 so arrow is still visible).
    pub(crate) chamfer_gizmo_bisector_px: Option<(f32, f32)>,
    /// Screen position of the radius handle (raw pixels).
    pub(crate) chamfer_gizmo_handle_px: Option<(f32, f32)>,
    pub(crate) chamfer_gizmo_hovered: bool,
    /// Normalised screen direction of the reference edge (raw pixels).
    pub(crate) chamfer_gizmo_edge_screen_dir: Option<(f32, f32)>,
    pub(crate) chamfer_gizmo_px_per_world: f64,
    pub(crate) chamfer_gizmo_drag_start_px: Option<(f32, f32)>,
    pub(crate) chamfer_gizmo_drag_start_radius: f64,

    // Move tool gizmo
    pub(crate) move_vertex_target: Option<(ObjectId, usize)>,
    pub(crate) move_gizmo_center_px: Option<(f32, f32)>,
    pub(crate) move_gizmo_x_tip_px: Option<(f32, f32)>,
    pub(crate) move_gizmo_y_tip_px: Option<(f32, f32)>,
    pub(crate) move_gizmo_z_tip_px: Option<(f32, f32)>,
    pub(crate) move_gizmo_x_px_per_world: f64,
    pub(crate) move_gizmo_y_px_per_world: f64,
    pub(crate) move_gizmo_z_px_per_world: f64,
    pub(crate) move_gizmo_hovered_axis: Option<u8>,
    pub(crate) gizmo_drag_axis_index: Option<u8>,
    pub(crate) move_panel_delta: [f64; 3],
    /// Last delta that was actually applied as a preview (to avoid redundant rebuilds).
    pub(crate) move_panel_last_preview: [f64; 3],

    /// Shared tool-highlight — draws this object in the selection colour regardless of selection.
    /// Used by Offset (selected target) and Explode (hovered object).
    pub(crate) tool_highlight_id: Option<ObjectId>,

    /// Color to use when creating or editing text (populated from the object on edit start).
    pub(crate) pending_text_color: [f32; 4],

    /// When true, show the "save before quit?" confirmation dialog.
    pub(crate) exit_confirm_open: bool,
    pub(crate) delete_confirm_open: bool,
    pub(crate) can_undo: bool,
    pub(crate) can_redo: bool,
    /// Dirty PIDB awaiting save/discard confirmation before it is removed.
    pub(crate) pending_close_project: Option<usize>,
    /// Queue of layers awaiting dirty-state unload confirmation. Each entry is
    /// `(project index, layer id, layer name)`; the front is shown in the dialog.
    /// A queue (not a single slot) so "Unload All Layers" can prompt per layer.
    pub(crate) pending_unload_queue: Vec<(usize, LayerId, String)>,
    /// Path of the triangulation folder pending "load all" confirmation.
    pub(crate) confirm_load_all_folder: Option<PathBuf>,

    // Create Triangulation workflow
    pub(crate) tri_create_open: bool,
    pub(crate) tri_create_phase: TriCreatePhase,
    pub(crate) tri_create_source: TriCreateSource,
    /// A failed Create Triangulation run, kept so the failure dialog can
    /// offer a coarse-weld retry with the same inputs.
    pub(crate) tri_create_failure: Option<TriCreateFailure>,
    /// Frozen cursor position (physical px) where the picker Area was opened.
    pub(crate) tri_create_picker_px: Option<(f32, f32)>,
    /// Objects highlighted yellow on canvas during selection hover.
    pub(crate) tri_hover_handles: HashSet<SceneEntityId>,
    /// Individually confirmed objects for triangulation.
    pub(crate) tri_selected_object_ids: Vec<ObjectId>,
    /// Confirmed layers — all their objects will be triangulated.
    pub(crate) tri_selected_layer_ids: Vec<LayerId>,
    pub(crate) tri_name_input: String,
    pub(crate) tri_surface_type: TriSurfaceType,

    // Unsaved triangulation close confirmation
    pub(crate) tri_close_unsaved: Option<TriangulationId>,

    // Cut Triangulation by Polygon
    pub(crate) tri_cut_poly_open: bool,
    pub(crate) tri_cut_poly_awaiting_pick: bool,
    pub(crate) tri_cut_poly_tri_id: Option<TriangulationId>,
    pub(crate) tri_cut_poly_object_id: Option<ObjectId>,
    pub(crate) tri_cut_poly_object_name: String,
    pub(crate) tri_cut_poly_name_input: String,

    // Cut Triangulation by Z Range
    pub(crate) tri_cut_z_open: bool,
    pub(crate) tri_cut_z_tri_id: Option<TriangulationId>,
    pub(crate) tri_cut_z_min_input: f64,
    pub(crate) tri_cut_z_max_input: f64,
    pub(crate) tri_cut_z_name_input: String,

    // Cut Triangulation by Surface
    pub(crate) tri_cut_surface_open: bool,
    pub(crate) tri_cut_surface_target_id: Option<TriangulationId>,
    pub(crate) tri_cut_surface_reference_id: Option<TriangulationId>,
    pub(crate) tri_cut_surface_side: TriSurfaceCutSide,
    pub(crate) tri_cut_surface_name_input: String,

    // Cut Topology to Pit Shell
    pub(crate) tri_cut_pitshell_open: bool,
    pub(crate) tri_cut_pitshell_topology_id: Option<TriangulationId>,
    pub(crate) tri_cut_pitshell_pitshell_id: Option<TriangulationId>,
    pub(crate) tri_cut_pitshell_name_input: String,

    // Include Pit/Stockpile Solid in Topology
    pub(crate) tri_include_solid_open: bool,
    pub(crate) tri_include_solid_topology_id: Option<TriangulationId>,
    pub(crate) tri_include_solid_shape_id: Option<TriangulationId>,
    pub(crate) tri_include_solid_name_input: String,

    // Contour Generation
    pub(crate) tri_contour_open: bool,
    pub(crate) tri_contour_tri_id: Option<TriangulationId>,
    pub(crate) tri_contour_major_interval_input: f64,
    pub(crate) tri_contour_minor_interval_input: f64,
    pub(crate) tri_contour_major_color: [f32; 4],
    pub(crate) tri_contour_minor_color: [f32; 4],
    pub(crate) tri_contour_project_index: usize,
    pub(crate) tri_contour_use_z_range: bool,
    pub(crate) tri_contour_z_min_input: f64,
    pub(crate) tri_contour_z_max_input: f64,
    pub(crate) point_cloud_tin_open: bool,
    pub(crate) point_cloud_tin_cloud_id: Option<PointCloudId>,
    pub(crate) point_cloud_tin_name_input: String,
    /// Maximum terrain TIN edge length. 0 disables edge filtering.
    pub(crate) point_cloud_tin_max_edge: f64,
    /// Input clouds larger than this are subsampled first.
    pub(crate) point_cloud_tin_max_points: u32,

    // Block Models
    pub(crate) block_model_table_pages: HashMap<BlockModelId, usize>,
    pub(crate) viewport_block_model_id: Option<BlockModelId>,
    pub(crate) block_model_variable_ranges: HashMap<(BlockModelId, String), Option<(f64, f64)>>,
    pub(crate) next_color_stop_id: u64,
    pub(crate) ore_triangulation_open: bool,
    pub(crate) ore_block_model_id: Option<BlockModelId>,
    pub(crate) ore_variable: String,
    pub(crate) ore_filter_mode: OreFilterMode,
    pub(crate) ore_min_input: f64,
    pub(crate) ore_max_input: f64,
    pub(crate) ore_name_input: String,

    // Road tool
    pub(crate) road_dialog_open: bool,
    pub(crate) road_width: f64,
    pub(crate) road_max_angle_degrees: f64,
    pub(crate) road_camber_degrees: f64,
    pub(crate) road_shape: RoadShape,
    pub(crate) road_preview_left_world: Vec<DVec3>,
    pub(crate) road_preview_right_world: Vec<DVec3>,
    /// Resolved ghost centerline (flat pockets included), empty when invalid.
    pub(crate) road_preview_center_world: Vec<DVec3>,
    pub(crate) road_preview_left_screen_px: Vec<OptionalScreenPointPx>,
    pub(crate) road_preview_right_screen_px: Vec<OptionalScreenPointPx>,
    pub(crate) road_preview_center_screen_px: Vec<OptionalScreenPointPx>,
    /// Rule the current stroke + cursor violates, refreshed with the preview.
    pub(crate) road_preview_violation: Option<crate::model::road_network::RoadRuleViolation>,
    /// Committed roads whose geometry the ghost reshapes (sorted by id). The
    /// static scene pass suppresses these; the dynamic pass draws them from
    /// `road_preview_affected_edges` instead.
    pub(crate) road_preview_affected_roads: Vec<ObjectId>,
    /// Ghost-inclusive resolved edges for the affected committed roads,
    /// refreshed by `update_road_preview` alongside the ghost preview.
    pub(crate) road_preview_affected_edges: Vec<crate::model::road_network::EdgeGeom>,

    // Batter Berm tool
    pub(crate) batter_berm_dialog_open: bool,
    pub(crate) batter_berm_target_id: Option<ObjectId>,
    pub(crate) batter_berm_width: f64,
    pub(crate) batter_berm_angle: f64,
    pub(crate) batter_berm_bench_height: f64,
    pub(crate) batter_berm_benches: u32,
    pub(crate) batter_berm_max_benches: u32,
    pub(crate) batter_berm_mode: BatterBermMode,
    /// All iteration rings in world coords: [toe_ring_0, berm_ring_0, toe_ring_1, berm_ring_1, …]
    pub(crate) batter_berm_rings_world: Vec<Vec<DVec3>>,
    pub(crate) batter_berm_source_world: Vec<DVec3>,
    pub(crate) batter_berm_guides_world: Vec<(DVec3, DVec3)>,
    /// Screen projections preserve one entry per world vertex. `None` means
    /// that vertex is outside the camera depth range, so adjacent vertices
    /// must not be joined across the gap.
    pub(crate) batter_berm_rings_screen_px: Vec<Vec<OptionalScreenPointPx>>,
    pub(crate) batter_berm_source_screen_px: Vec<OptionalScreenPointPx>,
    pub(crate) batter_berm_guides_screen_px: Vec<(OptionalScreenPointPx, OptionalScreenPointPx)>,
    pub(crate) batter_berm_preview_closed: bool,
    pub(crate) batter_berm_preview_key: Option<BatterBermPreviewKey>,

    // Bezier tool
    pub(crate) bezier_poly_id: Option<ObjectId>,
    /// Indices [first_vertex, second_vertex] of the two selected vertices in the polygon.
    pub(crate) bezier_selected_verts: [Option<usize>; 2],
    /// World-space coordinates of control point 1 (near the first selected vertex).
    pub(crate) bezier_cp1: [f64; 3],
    /// World-space coordinates of control point 2 (near the second selected vertex).
    pub(crate) bezier_cp2: [f64; 3],
    /// Number of line segments used to approximate the bezier curve.
    pub(crate) bezier_segments: u32,
    /// Screen-space positions of every vertex in the selected polygon (for white dot indicators).
    pub(crate) bezier_poly_verts_screen_px: Vec<(f32, f32)>,
    pub(crate) bezier_cp1_screen_px: Option<(f32, f32)>,
    pub(crate) bezier_cp2_screen_px: Option<(f32, f32)>,
    /// Screen-space positions of the dashed yellow preview polygon.
    pub(crate) bezier_preview_screen_px: Vec<(f32, f32)>,
    /// Which CP handle is being dragged (0 = cp1, 1 = cp2), None if not dragging.
    pub(crate) bezier_dragging_cp: Option<u8>,
    /// Which CP handle is hovered (0 = cp1, 1 = cp2), None if neither.
    pub(crate) bezier_hover_cp: Option<u8>,
    pub(crate) bezier_dialog_open: bool,
    /// Determines what preferences to show in the preferences tab
    pub(crate) active_preference_category: PreferenceCategory,
    pub(crate) show_import: bool,
    pub(crate) show_export: bool,
    /// What filetype should be selected in the import/export menu
    pub(crate) data_menu: DataMenu,
    pub(crate) import_source_menu: DataMenu,
    pub(crate) import_source_paths: Vec<PathBuf>,
    pub(crate) import_dxf_as_pidb: bool,
    pub(crate) import_dxf_project_index: Option<usize>,
    pub(crate) import_bmf_path: Option<PathBuf>,
    pub(crate) import_bdf_path: Option<PathBuf>,
    pub(crate) export_dxf_layer: bool,
    pub(crate) export_project_index: Option<usize>,
    pub(crate) export_layer: Option<(usize, LayerId)>,
    pub(crate) export_triangulation: Option<TriangulationId>,
}

impl EditorState {
    pub(crate) fn current_preferences(&self) -> PreferencesDraft {
        PreferencesDraft {
            renderer_background_color: self.renderer_background_color,
            topology_wireframes_enabled: self.topology_wireframes_enabled,
            dark_mode: self.dark_mode,
            show_console: self.show_console,
            show_world_axis_gizmo: self.show_world_axis_gizmo,
            snap_poll_rate: self.snap_poll_rate,
            frame_rate_cap: self.frame_rate_cap,
            resize_frame_rate_cap: self.resize_frame_rate_cap,
            block_model_interaction_resolution_divisor: self
                .block_model_interaction_resolution_divisor,
            frame_counter_enabled: self.frame_counter_enabled,
            debug_chunk_coloring: self.debug_chunk_coloring,
        }
    }

    pub(crate) fn reset_preferences_draft(&mut self) {
        self.preferences_draft = Some(self.current_preferences());
    }

    pub(crate) fn allocate_color_stop_id(&mut self) -> u64 {
        let id = self.next_color_stop_id;
        self.next_color_stop_id = self.next_color_stop_id.saturating_add(1);
        id
    }

    pub(crate) fn new() -> Self {
        Self {
            selected_handles: HashSet::new(),
            hidden_handles: HashSet::new(),
            frozen_handles: HashSet::new(),
            translucent_handles: HashSet::new(),
            topology_wireframes_enabled: false,
            dark_mode: false,
            show_console: true,
            show_world_axis_gizmo: crate::app::io::default_show_world_axis_gizmo(),
            renderer_background_color: crate::app::io::default_renderer_background_color(),
            preferences_draft: None,
            snap_poll_rate: crate::app::io::default_snap_poll_rate(),
            frame_rate_cap: crate::app::io::default_frame_rate_cap(),
            resize_frame_rate_cap: crate::app::io::default_resize_frame_rate_cap(),
            block_model_interaction_resolution_divisor:
                crate::app::io::default_block_model_interaction_resolution_divisor(),
            frame_counter_enabled: false,
            measured_fps: None,
            debug_chunk_coloring: false,
            debug_chunk_stats: None,
            status_message: None,
            active_tool: ActiveTool::None,
            cursor_mode: CursorMode::Select,
            tool_line_color: [1.0, 1.0, 1.0, 1.0],
            tool_line_weight: 1.0,
            tool_hatch: ToolHatch::Clear,
            active_layer: None,
            cursor_world: None,
            new_layer_dialog_open: false,
            new_layer_name: "design".to_owned(),
            new_layer_project_index: None,
            renaming_layer: None,
            pending_delete_layer: None,
            pending_stroke: Vec::new(),
            measurement_start: None,
            measurement_end: None,
            berm_angle_points: Vec::new(),
            pending_text: String::new(),
            pending_text_height: 15.0,
            pending_text_rotation_degrees: 0.0,
            text_edit_dialog_px: None,
            text_edit_position_frames: 0,
            text_edit_focus_requested: false,
            text_edit_created: false,
            text_editing_enabled: false,
            editing_labels_id: None,
            cursor_snapped: false,
            z_level: 0.0,
            z_input: 0.0,
            cursor_screen_px: None,
            poly_finish_dialog: false,
            poly_finish_dialog_px: None,
            canvas_context_menu_open: false,
            canvas_context_menu_px: None,
            move_to_layer_dialog: None,
            set_selection_z_dialog: None,
            move_layer_dialog: None,
            xray_enabled: false,
            vertical_exaggeration_dialog_open: false,
            vertical_exaggeration: 1.0,
            vertical_exaggeration_input: 1.,
            fly_mode_enabled: false,
            selection_box_start_px: None,
            selection_box_current_px: None,
            offset_dialog_open: false,
            offset_target_id: None,
            offset_target_ids: Vec::new(),
            offset_angle_degrees: 60.0,
            offset_measure: OffsetMeasure::Distance,
            offset_value_input: 0.0,
            offset_awaiting_side_pick: false,
            offset_horiz_dist: 0.0,
            offset_z_delta: 0.0,
            offset_project_to_rl: None,
            offset_collide_with_triangulation: false,
            offset_preview_world: Vec::new(),
            offset_source_world: Vec::new(),
            offset_preview_screen_px: Vec::new(),
            offset_source_screen_px: Vec::new(),
            offset_preview_ranges: Vec::new(),
            offset_preview_closed: false,
            relimit_dialog_open: false,
            relimit_source_id: None,
            relimit_mode: RelimitMode::Intersect,
            relimit_value_input: 0.0,
            relimit_awaiting_source_pick: false,
            relimit_waiting_for_pick: false,
            relimit_confirming_end: false,
            relimit_candidates: Vec::new(),
            relimit_hover_target_id: None,
            relimit_hover_target_screen_px: Vec::new(),
            relimit_hover_target_closed: false,
            relimit_preview_from_px: None,
            relimit_preview_to_px: None,
            relimit_preview_is_extension: true,
            relimit_resize_end: TrimEnd::End,
            relimit_resize_end_px: None,
            relimit_second_id: None,
            relimit_intersection_3d: None,
            relimit_hover_end: TrimEnd::End,
            fuse_segments: Vec::new(),
            fuse_awaiting_endpoint: None,
            fuse_endpoint_markers: Vec::new(),
            fuse_chain_tail: None,
            fuse_close_marker: None,
            split_poly_id: None,
            split_selected_verts: [None; 2],
            split_poly_verts_screen_px: Vec::new(),
            chamfer_radius: 1.0,
            chamfer_segments: 8,
            chamfer_max_radius: f64::MAX,
            chamfer_poly_id: None,
            chamfer_corner_index: None,
            chamfer_preview_screen_px: Vec::new(),
            chamfer_hover_corner_px: None,
            delete_hover_vertex_px: None,
            chamfer_gizmo_corner_px: None,
            chamfer_gizmo_bisector_px: None,
            chamfer_gizmo_handle_px: None,
            chamfer_gizmo_hovered: false,
            chamfer_gizmo_edge_screen_dir: None,
            chamfer_gizmo_px_per_world: 1.0,
            chamfer_gizmo_drag_start_px: None,
            chamfer_gizmo_drag_start_radius: 0.0,
            move_vertex_target: None,
            move_gizmo_center_px: None,
            move_gizmo_x_tip_px: None,
            move_gizmo_y_tip_px: None,
            move_gizmo_z_tip_px: None,
            move_gizmo_x_px_per_world: 1.0,
            move_gizmo_y_px_per_world: 1.0,
            move_gizmo_z_px_per_world: 1.0,
            move_gizmo_hovered_axis: None,
            gizmo_drag_axis_index: None,
            move_panel_delta: [0.0; 3],
            move_panel_last_preview: [f64::NAN; 3],
            tool_highlight_id: None,
            pending_text_color: [1.0, 1.0, 1.0, 1.0],
            exit_confirm_open: false,
            delete_confirm_open: false,
            can_undo: false,
            can_redo: false,
            pending_close_project: None,
            pending_unload_queue: Vec::new(),
            confirm_load_all_folder: None,
            tri_create_open: false,
            tri_create_phase: TriCreatePhase::MainDialog,
            tri_create_source: TriCreateSource::Polygons,
            tri_create_failure: None,
            tri_create_picker_px: None,
            tri_hover_handles: HashSet::new(),
            tri_selected_object_ids: Vec::new(),
            tri_selected_layer_ids: Vec::new(),
            tri_name_input: String::new(),
            tri_surface_type: TriSurfaceType::Surface,
            tri_close_unsaved: None,
            tri_cut_poly_open: false,
            tri_cut_poly_awaiting_pick: false,
            tri_cut_poly_tri_id: None,
            tri_cut_poly_object_id: None,
            tri_cut_poly_object_name: String::new(),
            tri_cut_poly_name_input: String::new(),
            tri_cut_z_open: false,
            tri_cut_z_tri_id: None,
            tri_cut_z_min_input: 0.0,
            tri_cut_z_max_input: 100.0,
            tri_cut_z_name_input: String::new(),
            tri_cut_surface_open: false,
            tri_cut_surface_target_id: None,
            tri_cut_surface_reference_id: None,
            tri_cut_surface_side: TriSurfaceCutSide::CutTop,
            tri_cut_surface_name_input: String::new(),
            tri_cut_pitshell_open: false,
            tri_cut_pitshell_topology_id: None,
            tri_cut_pitshell_pitshell_id: None,
            tri_cut_pitshell_name_input: String::new(),
            tri_include_solid_open: false,
            tri_include_solid_topology_id: None,
            tri_include_solid_shape_id: None,
            tri_include_solid_name_input: String::new(),
            tri_contour_open: false,
            tri_contour_tri_id: None,
            tri_contour_major_interval_input: 10.0,
            tri_contour_minor_interval_input: 2.0,
            tri_contour_major_color: [1.0, 0.5, 0.0, 1.0],
            tri_contour_minor_color: [0.8, 0.8, 0.8, 1.0],
            tri_contour_project_index: 0,
            tri_contour_use_z_range: false,
            tri_contour_z_min_input: 0.0,
            tri_contour_z_max_input: 100.0,
            point_cloud_tin_open: false,
            point_cloud_tin_cloud_id: None,
            point_cloud_tin_name_input: String::new(),
            point_cloud_tin_max_edge: 0.0,
            point_cloud_tin_max_points: 150_000,
            block_model_table_pages: HashMap::new(),
            viewport_block_model_id: None,
            block_model_variable_ranges: HashMap::new(),
            next_color_stop_id: FIRST_CUSTOM_COLOR_STOP_ID,
            ore_triangulation_open: false,
            ore_block_model_id: None,
            ore_variable: String::new(),
            ore_filter_mode: OreFilterMode::GreaterOrEqual,
            ore_min_input: 0.0,
            ore_max_input: 1.0,
            ore_name_input: String::new(),
            road_dialog_open: false,
            road_width: 50.0,
            road_max_angle_degrees: 15.0,
            road_camber_degrees: 1.73,
            road_shape: RoadShape::Crown,
            road_preview_left_world: Vec::new(),
            road_preview_right_world: Vec::new(),
            road_preview_center_world: Vec::new(),
            road_preview_left_screen_px: Vec::new(),
            road_preview_right_screen_px: Vec::new(),
            road_preview_center_screen_px: Vec::new(),
            road_preview_violation: None,
            road_preview_affected_roads: Vec::new(),
            road_preview_affected_edges: Vec::new(),
            batter_berm_dialog_open: false,
            batter_berm_target_id: None,
            batter_berm_width: 8.0,
            batter_berm_angle: 60.0,
            batter_berm_bench_height: 12.0,
            batter_berm_benches: 1,
            batter_berm_max_benches: 100,
            batter_berm_mode: BatterBermMode::Pit,
            batter_berm_rings_world: Vec::new(),
            batter_berm_source_world: Vec::new(),
            batter_berm_guides_world: Vec::new(),
            batter_berm_rings_screen_px: Vec::new(),
            batter_berm_source_screen_px: Vec::new(),
            batter_berm_guides_screen_px: Vec::new(),
            batter_berm_preview_closed: false,
            batter_berm_preview_key: None,
            bezier_poly_id: None,
            bezier_selected_verts: [None; 2],
            bezier_cp1: [0.0; 3],
            bezier_cp2: [0.0; 3],
            bezier_segments: 16,
            bezier_poly_verts_screen_px: Vec::new(),
            bezier_cp1_screen_px: None,
            bezier_cp2_screen_px: None,
            bezier_preview_screen_px: Vec::new(),
            bezier_dragging_cp: None,
            bezier_hover_cp: None,
            bezier_dialog_open: false,
            active_preference_category: PreferenceCategory::Interface,
            show_import: false,
            show_export: false,
            data_menu: DataMenu::None,
            import_source_menu: DataMenu::None,
            import_source_paths: Vec::new(),
            import_dxf_as_pidb: true,
            import_dxf_project_index: None,
            import_bmf_path: None,
            import_bdf_path: None,
            export_dxf_layer: false,
            export_project_index: None,
            export_layer: None,
            export_triangulation: None,
        }
    }

    /// Left-click that landed on empty space: clears the selection.
    pub(crate) fn on_canvas_click(&mut self, world: DVec3) {
        self.clear_selection();
        userspace_log!(
            "Canvas click @ {:.3}, {:.3}, {:.3}",
            world.x,
            world.y,
            world.z
        );
    }

    /// Left-click that landed on entity geometry: selects it and reports the
    /// picked world point (with the geometry's true Z).
    pub(crate) fn on_canvas_pick(
        &mut self,
        handle: SceneEntityId,
        world: DVec3,
        mode: SelectionMode,
    ) {
        self.cursor_world = Some(world);
        match mode {
            SelectionMode::Replace => self.replace_selection(handle),
            SelectionMode::Add => self.add_selection(handle),
            SelectionMode::Toggle => self.toggle_selection(handle),
        }
        userspace_log!(
            "Picked {handle:?} @ {:.3}, {:.3}, {:.3}",
            world.x,
            world.y,
            world.z
        );
    }

    /// Apply a display action to the current selection. Returns `true` when the
    /// rendered geometry must be rebuilt.
    pub(crate) fn apply_action(&mut self, action: EditorAction) -> bool {
        match action {
            EditorAction::HideSelection => {
                self.hidden_handles
                    .extend(self.selected_handles.iter().copied());
                userspace_log!("Hid {} object(s)", self.selected_handles.len());
                true
            }
            EditorAction::RevealAll => {
                self.hidden_handles.clear();
                self.frozen_handles.clear();
                self.translucent_handles.clear();
                userspace_log!("Revealed all objects");
                true
            }
            EditorAction::FreezeSelection => {
                self.frozen_handles
                    .extend(self.selected_handles.iter().copied());
                userspace_log!("Froze {} object(s)", self.selected_handles.len());
                // Frozen entities render identically; only picking is affected,
                // so no geometry rebuild is required.
                false
            }
        }
    }
}

/// How a canvas pick modifies the selection set.
#[derive(Clone, Copy)]
pub(crate) enum SelectionMode {
    Replace,
    Add,
    Toggle,
}

/// Currently active tool or `None` when no tool is engaged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActiveTool {
    None,
    MakePoint,
    MakeLine,
    MakePoly,
    MakeText,
    MakeRoad,
    DeleteElement,
    MeasureDistance,
    MeasureBermAngle,
    OffsetElement,
    RelimitLine,
    ExplodePolygon,
    FuseIntoPolygon,
    SplitAtPoints,
    Move,
    Chamfer,
    BatterBermOffset,
    Bezier,
}

/// Immediate commands applied to the current selection (or whole drawing).
#[derive(PartialEq, Clone, Copy)]
pub(crate) enum EditorAction {
    RevealAll,
    HideSelection,
    FreezeSelection,
}

/// Cursor interaction mode for canvas picks.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum CursorMode {
    Select,
    SnapToSurface,
    SnapToLine,
    SnapToPoint,
}

impl CursorMode {
    pub(crate) fn next(self) -> Self {
        match self {
            CursorMode::Select => CursorMode::SnapToSurface,
            CursorMode::SnapToSurface => CursorMode::SnapToLine,
            CursorMode::SnapToLine => CursorMode::SnapToPoint,
            CursorMode::SnapToPoint => CursorMode::Select,
        }
    }

    pub(crate) fn previous(self) -> Self {
        match self {
            CursorMode::Select => CursorMode::SnapToPoint,
            CursorMode::SnapToSurface => CursorMode::Select,
            CursorMode::SnapToLine => CursorMode::SnapToSurface,
            CursorMode::SnapToPoint => CursorMode::SnapToLine,
        }
    }
}

/// Fill pattern for closed polylines.
#[derive(PartialEq, Clone, Copy, EnumIter, Debug, Display)]
pub(crate) enum ToolHatch {
    Clear,
    Crosses,
    Slashes,
    Solid,
}

impl ToolHatch {
    pub(crate) fn to_fill_style(self) -> crate::model::FillStyle {
        match self {
            ToolHatch::Clear => crate::model::FillStyle::Clear,
            ToolHatch::Crosses => crate::model::FillStyle::Crosses,
            ToolHatch::Slashes => crate::model::FillStyle::Slashes,
            ToolHatch::Solid => crate::model::FillStyle::Solid,
        }
    }
}

/// Commands sent from the UI back to the application core.
///
/// Each variant represents an action the user triggered through the UI
/// (button clicks, menu selections, dialog confirmations).  The app layer
/// matches on these in its event loop.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum UiCommand {
    SetActiveTool(ActiveTool),
    SetFlyModeEnabled(bool),
    NewPidb,
    OpenPidb,
    CloseStartupDialog,
    SaveAllPidbs,
    ImportAsPidbPaths(DataMenu, Vec<PathBuf>),
    ImportDxfPathsInto(usize, Vec<PathBuf>),
    ImportTriangulationPaths(Vec<PathBuf>),
    ImportPointCloudPaths(Vec<PathBuf>),
    LoadPointCloud(PathBuf),
    ClosePointCloud(crate::model::point_cloud::PointCloudId),
    TogglePointCloudVisible(crate::model::point_cloud::PointCloudId),
    RemovePointCloud(PathBuf),
    RevealPointCloud(crate::model::point_cloud::PointCloudId),
    ChooseImportSourceFiles(DataMenu),
    OpenPidbPaths(Vec<PathBuf>),
    ChooseBlockModelBmf,
    ChooseBlockModelBdf,
    ConfirmImportBlockModel {
        bmf_path: PathBuf,
        bdf_path: Option<PathBuf>,
    },
    ExportPidbDxf(usize),
    ExportPidbCopy(usize),
    ExportLayerDxf(usize, LayerId),
    ExportTriangulationAs(TriangulationId, MeshFormat),
    SaveTriangulationAs(TriangulationId),
    SaveAndCloseTriangulationAs(TriangulationId),
    RevealTriangulation(TriangulationId),
    RevealBlockModel(BlockModelId),
    RequestExit,
    SaveAndExit,
    ExitWithoutSaving,
    CancelExit,
    CreateLayer {
        project_index: usize,
        name: String,
    },
    FinishPolyClose,
    CommitStrokeOpen,
    CancelOffset,
    CancelRelimit,
    ResetView,
    SetTopologyWireframes(bool),
    SetDarkMode(bool),
    SetShowConsole(bool),
    SetShowWorldAxisGizmo(bool),
    SetDebugChunkColoring(bool),
    SetStandardView(StandardView),
    ApplyPreferences(PreferencesDraft),
    OpenPreferences,
    SaveNamedPidb(usize),
    SaveNamedPidbAs(usize),
    ClosePidb(usize),
    SaveAndClosePidb(usize),
    ClosePidbForce(usize),
    RequestDeleteLayer(usize, LayerId),
    DeleteLayer(usize, LayerId),
    DuplicateLayer(usize, LayerId),
    BeginMoveLayer(usize, LayerId),
    MoveLayerToProject {
        source_project_index: usize,
        layer_id: LayerId,
        target_project_index: usize,
    },
    RenameLayer {
        project_index: usize,
        layer_id: LayerId,
        new_name: String,
    },
    BeginRenameLayer(usize, LayerId),
    /// Preview the move delta without committing (updates the live document view).
    PreviewMoveDelta(glam::DVec3),
    /// Apply a world-space delta to all selected objects.
    ApplyChamfer,
    CancelChamfer,
    ApplyBezier,
    CancelBezier,
    ApplyMoveDelta(glam::DVec3),
    CancelMoveDelta,
    LoadLayer(usize, LayerId),
    UnloadLayer(usize, LayerId),
    SelectAllObjectsInLayer(usize, LayerId),
    OpenTriangulationFolder,
    ActivateTriangulation(TriangulationId),
    ToggleTriangulationVisible(TriangulationId),
    CloseTriangulation(TriangulationId),
    /// Batch variants — produce a single history entry for multi-select changes.
    BatchSetObjectColor(Vec<ObjectId>, ObjectColor),
    BatchSetPolylineClosed(Vec<ObjectId>, bool),
    BatchSetObjectFill(Vec<ObjectId>, FillStyle),
    BatchSetPolylineLineWeight(Vec<ObjectId>, f32),
    MoveObjectsToLayer {
        project_index: usize,
        object_ids: Vec<ObjectId>,
        target_layer: LayerId,
        copy: bool,
    },
    BatchSetZValue(Vec<ObjectId>, f64),
    CommitTextEdit(ObjectId, String, f64, f64, [f32; 4]),
    CancelTextEdit,
    SetTriangulationColor(TriangulationId, [f32; 4]),
    CloseCanvasContextMenu,
    LoadTriangulation(PathBuf),
    LoadBlockModel(BlockModelSource),
    CloseBlockModel(BlockModelId),
    RemoveBlockModel(BlockModelSource),
    ToggleBlockModelVisible(BlockModelId),
    SetBlockModelColorVariable {
        id: BlockModelId,
        variable: String,
    },
    SetBlockModelColorStops {
        id: BlockModelId,
        stops: Vec<ColorStop>,
    },
    SetBlockModelHideEmptyValues {
        id: BlockModelId,
        hide: bool,
    },
    SetBlockModelDefinitionFile(BlockModelId),
    SetBlockModelSourceDefinitionFile(BlockModelSource),
    OpenBlockModelTable(BlockModelId),
    OpenCreateOreTriangulation,
    ExecuteCreateOreTriangulation {
        block_model_id: BlockModelId,
        variable: String,
        mode: OreFilterMode,
        min: f64,
        max: f64,
        name: String,
    },
    RemoveTriangulation(PathBuf),
    RemoveTriangulationFolder(PathBuf),
    RevealPidb(usize),
    RevealAllTriangulations,
    ZoomToExtents,
    /// Dialog "Apply" pressed — begin the canvas side-pick phase.
    BeginOffsetPick {
        object_ids: Vec<ObjectId>,
        /// Absolute horizontal offset distance (sign determined by cursor).
        horiz_dist: f64,
        /// Z shift to apply to all new vertices.
        z_delta: f64,
        /// When set, overrides `horiz_dist`/`z_delta`: project each vertex
        /// individually along `(tan_angle, target_rl)` so it lands flat at
        /// `target_rl` (angled batter projection to an absolute RL).
        project_to_rl: Option<(f64, f64)>,
        /// Clamp each offset vertex to the first visible triangulation hit
        /// between the source vertex and the requested offset endpoint.
        collide_with_triangulation: bool,
    },
    RelimitLineResize {
        source_id: ObjectId,
        mode: RelimitMode,
        value: f64,
    },
    /// Triggers the app to select the first valid polyline from the selection and open the offset
    /// dialog.
    OpenOffsetDialog,
    /// Triggers the app to select the first valid line from the selection and open the relimit
    /// dialog.
    OpenRelimitDialog,
    /// Triggers the app to select the first valid polyline and open the batter berm dialog.
    OpenBatterBermDialog,
    /// Dialog "Apply" pressed — commit all batter berm rings using the current panel state.
    CommitBatterBerm,
    CancelBatterBerm,
    /// Open the Create Triangulation main dialog.
    OpenCreateTriangulation,
    /// Open the road-to-triangulation dialog.
    OpenConvertRoadsToTriangulation,
    /// Replace selected road objects with their resolved polyline representation.
    ConvertSelectedRoadsToPolylines,
    /// Open the Set selections to Z value.
    OpenSetSelectionZValueDialog,
    /// Run CDT on the supplied object list and add the result as a loaded triangulation.
    ExecuteCreateTriangulation {
        name: String,
        object_ids: Vec<ObjectId>,
        surface_type: TriSurfaceType,
    },
    /// Convert selected road objects to a triangulation mesh.
    ExecuteConvertRoadsToTriangulation {
        name: String,
        object_ids: Vec<ObjectId>,
    },
    /// Retry a failed Create Triangulation with breakline endpoints welded
    /// at the coarse (cm-scale) tolerance the failure dialog offered.
    ExecuteCreateTriangulationWithWeld {
        name: String,
        object_ids: Vec<ObjectId>,
        surface_type: TriSurfaceType,
    },
    /// Open the point cloud terrain TIN dialog (Survey menu).
    OpenPointCloudTin,
    /// Reconstruct a terrain TIN from a point cloud.
    ExecutePointCloudTin {
        cloud_id: PointCloudId,
        name: String,
        max_edge: f64,
        max_points: u32,
    },
    /// Load all layers in the given project (no confirmation needed).
    LoadAllLayersInProject(usize),
    /// Unload all layers in the given project (shows dirty-check dialog if needed).
    UnloadAllLayersInProject(usize),
    /// Show the "load all in folder?" confirmation dialog for the given folder.
    ConfirmLoadAllTriangulationsInFolder(PathBuf),
    /// Actually load every triangulation file in the given folder.
    LoadAllTriangulationsInFolder(PathBuf),
    /// Close (unload) every loaded triangulation in the given folder.
    CloseAllTriangulationsInFolder(PathBuf),
    /// Save the .pidb project that contains a specific layer (the "Save Layer" menu entry).
    SaveProjectForLayer(usize, LayerId),
    /// Unload a layer without checking for unsaved changes (confirmed by the user).
    UnloadLayerConfirmed(usize, LayerId),
    /// Save the project and then unload a specific layer.
    SaveAndUnloadLayer(usize, LayerId),
    /// Close an unsaved triangulation without saving (user confirmed the discard).
    CloseTriangulationForce(TriangulationId),
    /// User confirmed deletion of all selected objects via the confirm dialog.
    ConfirmDeleteSelection,
    /// Open the "Cut Triangulation by Polygon" dialog.
    OpenCutTriangulationByPolygon,
    /// Enter polygon-pick mode for the cut-by-polygon tool.
    BeginCutPolyPick,
    /// Execute the cut: clip faces against the polygon boundary in XY.
    ExecuteCutTriangulationByPolygon {
        tri_id: TriangulationId,
        polygon_id: ObjectId,
        name: String,
    },
    /// Open the "Cut Triangulation by Z Range" dialog.
    OpenCutTriangulationByZ,
    /// Execute the Z-range cut, clipping faces at the boundary planes.
    ExecuteCutTriangulationByZ {
        tri_id: TriangulationId,
        z_min: f64,
        z_max: f64,
        name: String,
    },
    /// Open the "Cut Triangulation by Surface" dialog.
    OpenCutTriangulationBySurface,
    /// Clip one triangulation against another in the vertical direction.
    ExecuteCutTriangulationBySurface {
        target_id: TriangulationId,
        reference_id: TriangulationId,
        side: TriSurfaceCutSide,
        name: String,
    },
    /// Open the "Cut Topology to Pit Shell" dialog.
    OpenCutTopologyByPitShell,
    /// Trim a topology to the region outside a pit shell's true 3D footprint.
    ExecuteCutTopologyByPitShell {
        topology_id: TriangulationId,
        pit_shell_id: TriangulationId,
        name: String,
    },
    /// Open the "Include Pit/Stockpile Solid" dialog.
    OpenIncludeSolidInTopology,
    /// Replace the topology footprint with a pit or stockpile solid.
    ExecuteIncludeSolidInTopology {
        topology_id: TriangulationId,
        shape_id: TriangulationId,
        name: String,
    },
    /// Open the "Generate Contour Lines" dialog.
    OpenContourTriangulation,
    CommitRoad,
    CancelRoad,
    Undo,
    Redo,
    /// Execute contour generation and store lines as a new layer in the given pidb.
    ExecuteContourTriangulation {
        tri_id: TriangulationId,
        major_interval: f64,
        minor_interval: f64,
        major_color: [f32; 4],
        minor_color: [f32; 4],
        project_index: usize,
        /// Optional `(min, max)` RL band to contour instead of the full mesh.
        z_range: Option<(f64, f64)>,
    },
}

/// Output produced by a single UI frame.
pub(crate) struct UiFrameOutput {
    pub(crate) repaint: bool,
    pub(crate) geometry_dirty: bool,
    /// The pointer is actively pressed on an egui widget this frame (e.g.
    /// dragging a colour-gradient stop). Used to drop the volume raycaster to
    /// its low-quality interaction path, like camera drags and resizing.
    pub(crate) ui_pointer_active: bool,
    pub(crate) commands: Vec<UiCommand>,
}

/// One loaded layer shown in the explorer tree.
#[derive(Clone, Debug)]
pub(crate) struct UiLayerEntry {
    pub(crate) id: LayerId,
    pub(crate) name: String,
    pub(crate) is_loaded: bool,
    pub(crate) dirty: bool,
}

/// One .pidb project entry shown in the explorer tree.
#[derive(Clone, Debug)]
pub(crate) struct UiProjectEntry {
    pub(crate) name: String,
    pub(crate) dirty: bool,
    pub(crate) index: usize,
    pub(crate) is_active: bool,
    pub(crate) layers: Vec<UiLayerEntry>,
    pub(crate) path: Option<PathBuf>,
}

/// One triangulation shown in the explorer tree (individual file or within a folder group).
#[derive(Clone, Debug)]
pub(crate) struct UiPointCloudEntry {
    pub(crate) id: Option<crate::model::point_cloud::PointCloudId>,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) visible: bool,
    pub(crate) is_loaded: bool,
    pub(crate) point_count: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct UiTriangulationEntry {
    pub(crate) id: Option<TriangulationId>,
    pub(crate) name: String,
    pub(crate) visible: bool,
    pub(crate) is_active: bool,
    pub(crate) is_loaded: bool,
    pub(crate) is_saved: bool,
    pub(crate) path: PathBuf,
    /// The directory this entry was discovered from; `None` for individually-opened files.
    pub(crate) group: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub(crate) struct UiBlockModelEntry {
    pub(crate) id: Option<BlockModelId>,
    pub(crate) name: String,
    pub(crate) visible: bool,
    pub(crate) is_active: bool,
    pub(crate) is_loaded: bool,
    pub(crate) source: BlockModelSource,
    pub(crate) _block_count: usize,
    pub(crate) numeric_variable_count: usize,
}

/// Active triangulation id and face colour, as surfaced to the canvas context menu.
pub(crate) type TriangulationMenuStyle = (TriangulationId, [f32; 4]);

/// Flattened snapshot of the project tree, built each frame by the app layer.
#[derive(Clone, Debug, Default)]
pub(crate) struct UiProjectView {
    pub(crate) projects: Vec<UiProjectEntry>,
    pub(crate) triangulations: Vec<UiTriangulationEntry>,
    pub(crate) block_models: Vec<UiBlockModelEntry>,
    pub(crate) point_clouds: Vec<UiPointCloudEntry>,
    pub(crate) active_index: Option<usize>,
    pub(crate) needs_startup_dialog: bool,
    /// Full filesystem path of the currently active PIDB, if any.
    pub(crate) active_path: Option<PathBuf>,
    /// Active triangulation id and face colour, used by the context menu.
    pub(crate) active_triangulation_for_menu: Option<TriangulationMenuStyle>,
}

#[derive(PartialEq)]
pub(crate) enum PreferenceCategory {
    Interface,
    Performance,
    Developer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DataMenu {
    None,
    Dxf,
    Pidb,
    DgdIsis,
    Duf,
    Tri00t,
    Obj,
    Stl,
    Ply,
    Las,
    Xyz,
    Pcd,
    Bmf,
}
