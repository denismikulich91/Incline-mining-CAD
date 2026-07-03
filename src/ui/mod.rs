//! Top-level UI module: egui state, rendering pipeline, and the main draw_ui entry point.
//!
//! The `Gui` struct owns the egui context, winit state, and wgpu renderer.
//! `render()` processes input, calls `draw_ui()`, and feeds paint jobs back to wgpu.

pub(crate) mod dialogs;
pub(crate) mod elements;
pub(crate) mod fonts;
pub(crate) mod state;
pub(crate) mod widgets;

/// Select a themed icon by name, picking from `icons_dark/` or `icons_light/` based on `dark_mode`.
///
/// Expands to an `egui::ImageSource<'static>` suitable for `egui::Image::new(...)`.
macro_rules! themed_icon {
    ($ui:expr, $name:literal) => {{
        if $ui.visuals().dark_mode {
            egui::include_image!(concat!("../../../res/ui/icons_dark/", $name))
        } else {
            egui::include_image!(concat!("../../../res/ui/icons_light/", $name))
        }
    }};
}

/// Select an icon by name, picking from `icons/`.
///
/// Expands to an `egui::ImageSource<'static>` suitable for `egui::Image::new(...)`.
macro_rules! unthemed_icon {
    ($name:literal) => {{ egui::include_image!(concat!("../../../res/ui/icons/", $name)) }};
}

use egui_wgpu::ScreenDescriptor;
pub(crate) use themed_icon;
pub(crate) use unthemed_icon;
use winit::{event::WindowEvent, window::Window};

use crate::{
    model::{Document, block_model::OpenBlockModel, road_network::RoadRuleViolation},
    rendering::color::{color32_to_rgba, rgba_to_color32},
    ui::{
        fonts::setup_custom_fonts,
        state::{
            ActiveTool, EditorState, UiCommand, UiFrameOutput, UiProjectView, UiTriangulationEntry,
        },
        widgets::viewport::{ViewportLabel, ViewportLabelStyle},
    },
};

pub(crate) const SELECTION_COLOR_F32: [f32; 4] = [87.0 / 255.0, 163.0 / 255.0, 1.0, 1.0];
pub(crate) const SELECTION_COLOR: egui::Color32 = egui::Color32::from_rgb(87, 163, 255);

/// Owned egui GUI state: context, winit bridge, and wgpu tessellation renderer.
///
/// Created once at application startup; mutated each frame via `handle_event`
/// and `render`.
pub(crate) struct Gui {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl Gui {
    pub(crate) fn new(
        window: &Window,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let ctx = egui::Context::default();
        setup_custom_fonts(&ctx);
        egui_extras::install_image_loaders(&ctx);
        #[cfg(feature = "inspection")]
        match egui_inspection::attach_from_env(&ctx, Some("Incline".to_owned())) {
            Ok(true) => log::info!("egui_inspection: attached (EGUI_INSPECTION set)"),
            Ok(false) => {}
            Err(error) => log::warn!("egui_inspection: failed to attach: {error}"),
        }
        ctx.global_style_mut(|style| {
            style.interaction.show_tooltips_only_when_still = false;
            style.interaction.tooltip_delay = 0.0;
        });
        ctx.set_visuals(theme_visuals(false, SELECTION_COLOR));

        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        Self {
            ctx,
            state,
            renderer,
        }
    }

    pub(crate) fn handle_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
    ) -> egui_winit::EventResponse {
        self.state.on_window_event(window, event)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render(
        &mut self,
        window: &Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        editor: &mut EditorState,
        document: &mut Document,
        project: &UiProjectView,
        block_models: &[OpenBlockModel],
        screen_size: [u32; 2],
        orbit_marker: Option<(f32, f32)>,
        camera_forward: [f32; 3],
        camera_up: [f32; 3],
    ) -> UiFrameOutput {
        let selection_color = SELECTION_COLOR;
        let visuals = &self.ctx.global_style().visuals;
        if visuals.dark_mode != editor.dark_mode
            || visuals.selection.stroke.color != selection_color
        {
            self.ctx
                .set_visuals(theme_visuals(editor.dark_mode, selection_color));
        }
        let raw_input = self.state.take_egui_input(window);
        let mut geometry_dirty = false;
        let mut commands = Vec::new();
        let overlay = CanvasOverlay {
            orbit_marker,
            camera_forward,
            camera_up,
        };
        let full_output = self.ctx.run_ui(raw_input, |ui| {
            geometry_dirty |= draw_ui(
                ui,
                editor,
                document,
                project,
                block_models,
                &mut commands,
                overlay,
            );
        });

        let repaint = full_output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .is_some_and(|output| output.repaint_delay.is_zero());
        self.state
            .handle_platform_output(window, full_output.platform_output);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }

        let pixels_per_point = full_output.pixels_per_point;
        let paint_jobs = self.ctx.tessellate(full_output.shapes, pixels_per_point);
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: screen_size,
            pixels_per_point,
        };
        let extra_command_buffers =
            self.renderer
                .update_buffers(device, queue, encoder, &paint_jobs, &screen_descriptor);

        for command_buffer in extra_command_buffers {
            queue.submit(std::iter::once(command_buffer));
        }

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.renderer.render(
                &mut render_pass.forget_lifetime(),
                &paint_jobs,
                &screen_descriptor,
            );
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        UiFrameOutput {
            repaint,
            geometry_dirty,
            commands,
        }
    }
}

/// Top-level UI layout: panels, toolbars, dialogs, and canvas overlay.
///
/// Returns `true` if geometry needs to be rebuilt (e.g. selection state changed).
#[derive(Clone, Copy)]
struct CanvasOverlay {
    orbit_marker: Option<(f32, f32)>,
    camera_forward: [f32; 3],
    camera_up: [f32; 3],
}

/// Short viewport warning for the rule the road preview currently violates
/// (kept fresh by `update_road_preview`).
fn road_preview_violation_text(editor: &EditorState) -> Option<&'static str> {
    if editor.active_tool != ActiveTool::MakeRoad || editor.pending_stroke.is_empty() {
        return None;
    }
    editor
        .road_preview_violation
        .map(|violation| match violation {
            RoadRuleViolation::SegmentTooSteep { .. } => "Ramp gradient is too steep",
            RoadRuleViolation::TurnTooSharp { .. } => "Road turn is too sharp",
            RoadRuleViolation::ClearanceTooTight { .. } => {
                "No room for the flat junction approaches"
            }
            RoadRuleViolation::TurnTooCloseToJunction { .. } => {
                "Road turns too close to a junction"
            }
            RoadRuleViolation::DegenerateSegment => "Road segment is too short",
        })
}

struct ViewportToolLabel {
    text: String,
    style: ViewportLabelStyle,
}

impl ViewportToolLabel {
    fn neutral(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            style: ViewportLabelStyle::Neutral,
        }
    }

    fn important(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            style: ViewportLabelStyle::Important,
        }
    }
}

fn viewport_label_text(editor: &EditorState) -> Option<ViewportToolLabel> {
    if editor.active_tool == ActiveTool::MeasureDistance
        && let (Some(start), Some(end)) = (editor.measurement_start, editor.measurement_end)
    {
        return Some(ViewportToolLabel::neutral(format!(
            "{:.3} meters",
            start.distance(end)
        )));
    }

    match editor.active_tool {
        ActiveTool::Move if editor.selected_handles.is_empty() => Some("Select an item"),
        ActiveTool::OffsetElement if editor.offset_awaiting_side_pick => Some("Choose offset side"),
        ActiveTool::OffsetElement if editor.offset_target_id.is_none() => {
            Some("Select a line or polygon")
        }
        ActiveTool::RelimitLine if editor.relimit_confirming_end => Some("Choose relimit side"),
        ActiveTool::RelimitLine if editor.relimit_waiting_for_pick => {
            Some("Select line to relimit to")
        }
        ActiveTool::RelimitLine
            if editor.relimit_source_id.is_none() || editor.relimit_awaiting_source_pick =>
        {
            Some("Select line to relimit")
        }
        ActiveTool::FuseIntoPolygon if editor.fuse_awaiting_endpoint.is_some() => {
            Some("Select the endpoint to join")
        }
        ActiveTool::FuseIntoPolygon if !editor.fuse_segments.is_empty() => {
            Some("Select the next line to fuse")
        }
        ActiveTool::FuseIntoPolygon => Some("Select a line to fuse"),
        ActiveTool::Chamfer if editor.chamfer_corner_index.is_none() => {
            Some("Select a polygon vertex")
        }
        ActiveTool::Bezier if editor.bezier_poly_id.is_none() => Some("Select a polygon"),
        ActiveTool::Bezier if editor.bezier_selected_verts[0].is_none() => {
            Some("Click first vertex")
        }
        ActiveTool::Bezier if editor.bezier_selected_verts[1].is_none() => {
            Some("Click second vertex")
        }
        ActiveTool::ExplodePolygon => Some("Select a polygon"),
        ActiveTool::BatterBermOffset if editor.batter_berm_target_id.is_none() => {
            Some("Select a polygon")
        }
        ActiveTool::DeleteElement => Some("Select an item"),
        ActiveTool::MakeRoad => {
            return road_preview_violation_text(editor).map(ViewportToolLabel::important);
        }
        _ => None,
    }
    .map(ViewportToolLabel::neutral)
}

fn draw_ui(
    root_ui: &mut egui::Ui,
    editor: &mut EditorState,
    document: &mut Document,
    project: &UiProjectView,
    block_models: &[OpenBlockModel],
    commands: &mut Vec<UiCommand>,
    overlay: CanvasOverlay,
) -> bool {
    let mut geometry_dirty = false;

    // --- Panel layout: compute rects for all fixed panels ---
    let project_active = project.active_index.is_some();
    let editing_enabled = project.active_index.is_some() && editor.active_layer.is_some();

    let main_menu_rect =
        crate::ui::elements::main_menu::draw_main_menu(root_ui, editor, project, commands);
    crate::ui::elements::tabs::sync_block_model_table_tabs(block_models);
    let (tabs_rect, active_tab) = crate::ui::elements::tabs::draw_tabs(root_ui);
    match active_tab {
        crate::ui::elements::tabs::TabClass::Preferences => {
            crate::ui::elements::preferences::draw_preferences(root_ui, editor, commands);
            if editor.exit_confirm_open {
                crate::ui::dialogs::confirmations::draw_exit_confirm_dialog(
                    root_ui, commands, editor,
                );
            }
            return false;
        }
        crate::ui::elements::tabs::TabClass::BlockModelTable(id) => {
            crate::ui::elements::block_model::draw_block_model_table(
                root_ui,
                editor,
                block_models,
                id,
            );
            return false;
        }
        crate::ui::elements::tabs::TabClass::Workspace => {}
    }

    // --- Draw all toolbar panels ---
    let top_toolbar_rect = crate::ui::elements::toolbars::draw_top_toolbar(
        root_ui,
        editor,
        project,
        commands,
        document,
        editor.can_undo,
        editor.can_redo,
    );
    let status_bar_rect = crate::ui::elements::status_bar::draw_status_bar(root_ui, editor);
    let explorer_rect =
        crate::ui::elements::explorer::draw_explorer(root_ui, editor, project, commands);

    // Draw console if enabled
    if editor.show_console {
        crate::ui::elements::console::draw_console(root_ui);
    }
    let bottom_toolbar_rect = crate::ui::elements::toolbars::draw_bottom_toolbar(
        root_ui,
        editor,
        &mut geometry_dirty,
        commands,
    );
    let left_toolbar_rect = crate::ui::elements::toolbars::draw_left_toolbar(
        root_ui,
        editor,
        editing_enabled,
        project_active,
    );

    let right_toolbar_rect =
        crate::ui::elements::toolbars::draw_right_toolbar(root_ui, editor, commands);

    // --- Compute canvas rect (area not occupied by panels) ---
    let canvas_rect = egui::Rect::from_min_max(
        egui::pos2(
            explorer_rect.right().max(left_toolbar_rect.right()),
            main_menu_rect
                .bottom()
                .max(tabs_rect.bottom())
                .max(top_toolbar_rect.bottom()),
        ),
        egui::pos2(
            right_toolbar_rect.left(),
            status_bar_rect.top().min(bottom_toolbar_rect.top()),
        ),
    );

    if let (Some(start), Some(end)) = (
        editor.selection_box_start_px,
        editor.selection_box_current_px,
    ) {
        // Vulcan-style: left-to-right = cross select (dashed green), right-to-left = window select
        // (solid blue).
        let cross_select = end.0 > start.0;
        let box_color = if cross_select {
            egui::Color32::from_rgb(80, 220, 100) // green for cross/touch select
        } else {
            SELECTION_COLOR // theme colour for window select
        };
        let pixels_per_point = root_ui.ctx().pixels_per_point();
        let selection_rect = egui::Rect::from_two_pos(
            egui::pos2(start.0 / pixels_per_point, start.1 / pixels_per_point),
            egui::pos2(end.0 / pixels_per_point, end.1 / pixels_per_point),
        )
        .intersect(canvas_rect);
        root_ui.painter().rect_filled(
            selection_rect,
            0.0,
            egui::Color32::from_rgba_unmultiplied(box_color.r(), box_color.g(), box_color.b(), 14),
        );
        if cross_select {
            // Dashed border for cross selection
            let painter = root_ui.painter();
            let dash = 6.0;
            let gap = 4.0;
            let stroke = egui::Stroke::new(1.0, box_color);
            let r = selection_rect;
            for seg in dashed_rect_segments(r, dash, gap) {
                painter.line_segment(seg, stroke);
            }
        } else {
            root_ui.painter().rect_stroke(
                selection_rect,
                0.0,
                egui::Stroke::new(1.0, box_color),
                egui::StrokeKind::Inside,
            );
        }
    }

    // Move tool: draw 3-axis gizmo
    if editor.active_tool == ActiveTool::Move
        && !editor.selected_handles.is_empty()
        && let Some(center_px) = editor.move_gizmo_center_px
    {
        let ppp = root_ui.ctx().pixels_per_point();
        let center = egui::pos2(center_px.0 / ppp, center_px.1 / ppp);
        let painter = root_ui.painter();
        let axes = [
            (
                editor.move_gizmo_x_tip_px,
                egui::Color32::from_rgb(220, 50, 50),
                0u8,
                "X",
            ),
            (
                editor.move_gizmo_y_tip_px,
                egui::Color32::from_rgb(50, 200, 50),
                1u8,
                "Y",
            ),
            (
                editor.move_gizmo_z_tip_px,
                egui::Color32::from_rgb(50, 100, 220),
                2u8,
                "Z",
            ),
        ];
        for (tip_opt, color, axis_idx, label) in axes {
            let is_active = editor.move_gizmo_hovered_axis == Some(axis_idx)
                || editor.gizmo_drag_axis_index == Some(axis_idx);
            let draw_color = if is_active {
                egui::Color32::WHITE
            } else {
                color
            };
            let width = if is_active { 3.0 } else { 2.0 };
            if let Some(tip_px) = tip_opt {
                let tip = egui::pos2(tip_px.0 / ppp, tip_px.1 / ppp);
                painter.line_segment([center, tip], egui::Stroke::new(width, draw_color));
                painter.circle_filled(tip, 5.0, draw_color);
                painter.text(
                    egui::pos2(tip.x + 4.0, tip.y - 8.0),
                    egui::Align2::LEFT_TOP,
                    label,
                    egui::FontId::proportional(12.0),
                    draw_color,
                );
            }
        }
        painter.circle_filled(center, 4.0, egui::Color32::WHITE);
    }

    // Move tool: numeric delta panel
    if editor.active_tool == ActiveTool::Move && !editor.selected_handles.is_empty() {
        crate::ui::dialogs::editing::draw_move_panel(root_ui, editor, commands, canvas_rect);
    }

    // Chamfer tool: gizmo overlay + dock panel
    if editor.active_tool == ActiveTool::Chamfer {
        // Hover sphere: show on nearest valid corner before the user clicks one
        if let Some(hover_px) = editor.chamfer_hover_corner_px {
            let ppp = root_ui.ctx().pixels_per_point();
            let pos = egui::pos2(hover_px.0 / ppp, hover_px.1 / ppp);
            root_ui.painter().with_clip_rect(canvas_rect).circle_filled(
                pos,
                6.0,
                egui::Color32::from_rgb(255, 220, 50),
            );
        }

        // Draw the radius gizmo: cylinder+cone arrow always visible from the corner outward.
        if let Some(corner_px) = editor.chamfer_gizmo_corner_px {
            let ppp = root_ui.ctx().pixels_per_point();
            let corner = egui::pos2(corner_px.0 / ppp, corner_px.1 / ppp);
            let painter = root_ui.painter();
            let color =
                if editor.chamfer_gizmo_hovered || editor.chamfer_gizmo_drag_start_px.is_some() {
                    egui::Color32::WHITE
                } else {
                    egui::Color32::from_rgb(255, 200, 50)
                };
            const CORNER_R: f32 = 5.0;
            const STUB_LEN: f32 = 40.0;
            const SHAFT_W: f32 = 3.5;
            const HEAD_LEN: f32 = 12.0;
            const HEAD_W: f32 = 7.0;
            const HANDLE_R: f32 = 6.0;

            if let Some(handle_px) = editor.chamfer_gizmo_handle_px {
                let handle = egui::pos2(handle_px.0 / ppp, handle_px.1 / ppp);
                let raw_vec = handle - corner;
                // Use the edge stub direction when radius ≈ 0 so the arrow is still visible.
                let dir = if raw_vec.length() > 4.0 {
                    raw_vec.normalized()
                } else if let Some(ed) = editor.chamfer_gizmo_bisector_px {
                    egui::vec2(ed.0, ed.1).normalized()
                } else {
                    egui::vec2(1.0, 0.0)
                };
                let tip = if raw_vec.length() > STUB_LEN {
                    handle
                } else {
                    corner + dir * STUB_LEN
                };
                // Shaft starts just past the corner circle so it doesn't overlap it.
                let shaft_start = corner + dir * (CORNER_R + 1.0);
                let shaft_end = tip - dir * HEAD_LEN;
                if (shaft_end - shaft_start).length() > 1.0 {
                    painter
                        .line_segment([shaft_start, shaft_end], egui::Stroke::new(SHAFT_W, color));
                }
                let perp = egui::vec2(-dir.y, dir.x);
                painter.add(egui::Shape::convex_polygon(
                    vec![tip, shaft_end + perp * HEAD_W, shaft_end - perp * HEAD_W],
                    color,
                    egui::Stroke::NONE,
                ));
                // Handle circle only when radius > 0 and handle is away from corner.
                if raw_vec.length() > 4.0 {
                    painter.circle_filled(handle, HANDLE_R, color);
                }
            } else if let Some(ed) = editor.chamfer_gizmo_bisector_px {
                let dir = egui::vec2(ed.0, ed.1).normalized();
                let shaft_start = corner + dir * (CORNER_R + 1.0);
                let tip = corner + dir * STUB_LEN;
                let shaft_end = tip - dir * HEAD_LEN;
                painter.line_segment([shaft_start, shaft_end], egui::Stroke::new(SHAFT_W, color));
                let perp = egui::vec2(-dir.y, dir.x);
                painter.add(egui::Shape::convex_polygon(
                    vec![tip, shaft_end + perp * HEAD_W, shaft_end - perp * HEAD_W],
                    color,
                    egui::Stroke::NONE,
                ));
            }
            painter.circle_filled(corner, CORNER_R, color);
        }

        // Draw the chamfer preview polyline
        if !editor.chamfer_preview_screen_px.is_empty() {
            let ppp = root_ui.ctx().pixels_per_point();
            let pts: Vec<egui::Pos2> = editor
                .chamfer_preview_screen_px
                .iter()
                .map(|(x, y)| egui::pos2(x / ppp, y / ppp))
                .collect();
            let painter = root_ui.painter().with_clip_rect(canvas_rect);
            let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 220, 0));
            let n = pts.len();
            for i in 0..n {
                painter.line_segment([pts[i], pts[(i + 1) % n]], stroke);
            }
        }

        crate::ui::dialogs::editing::draw_chamfer_panel(root_ui, editor, commands, canvas_rect);
    }

    if editor.active_tool == ActiveTool::DeleteElement
        && let Some(hover_px) = editor.delete_hover_vertex_px
    {
        let ppp = root_ui.ctx().pixels_per_point();
        let pos = egui::pos2(hover_px.0 / ppp, hover_px.1 / ppp);
        let painter = root_ui.painter().with_clip_rect(canvas_rect);
        let color = SELECTION_COLOR;
        painter.circle_filled(pos, 6.0, color);
        painter.circle_stroke(pos, 7.5, egui::Stroke::new(1.5, egui::Color32::WHITE));
    }

    // Bezier tool: vertex dots, control handles, and dashed preview
    if editor.active_tool == ActiveTool::Bezier {
        let ppp = root_ui.ctx().pixels_per_point();
        let painter = root_ui.painter().with_clip_rect(canvas_rect);

        // All polygon vertices as white dots
        for &(x, y) in &editor.bezier_poly_verts_screen_px {
            painter.circle_filled(egui::pos2(x / ppp, y / ppp), 5.0, egui::Color32::WHITE);
        }

        // Selected vertices in selection colour
        for slot in 0..2usize {
            if let Some(vi) = editor.bezier_selected_verts[slot]
                && let Some(&(x, y)) = editor.bezier_poly_verts_screen_px.get(vi)
            {
                painter.circle_filled(egui::pos2(x / ppp, y / ppp), 7.0, SELECTION_COLOR);
            }
        }

        // Dashed yellow preview polygon
        if !editor.bezier_preview_screen_px.is_empty() {
            let pts: Vec<egui::Pos2> = editor
                .bezier_preview_screen_px
                .iter()
                .map(|(x, y)| egui::pos2(x / ppp, y / ppp))
                .collect();
            let dash_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 220, 0));
            let n = pts.len();
            for i in 0..n {
                let next = (i + 1) % n;
                for seg in dashed_line_segments(pts[i], pts[next], 6.0, 4.0) {
                    painter.line_segment(seg, dash_stroke);
                }
            }
        }

        // Control point gizmos (only when both vertices are selected)
        if let (Some(cp1_px), Some(cp2_px)) =
            (editor.bezier_cp1_screen_px, editor.bezier_cp2_screen_px)
        {
            // Tangent lines from anchor vertices to control points
            if let Some(vi) = editor.bezier_selected_verts[0]
                && let Some(&v_px) = editor.bezier_poly_verts_screen_px.get(vi)
            {
                painter.line_segment(
                    [
                        egui::pos2(v_px.0 / ppp, v_px.1 / ppp),
                        egui::pos2(cp1_px.0 / ppp, cp1_px.1 / ppp),
                    ],
                    egui::Stroke::new(
                        1.5,
                        egui::Color32::from_rgba_unmultiplied(255, 150, 50, 200),
                    ),
                );
            }
            if let Some(vj) = editor.bezier_selected_verts[1]
                && let Some(&v_px) = editor.bezier_poly_verts_screen_px.get(vj)
            {
                painter.line_segment(
                    [
                        egui::pos2(v_px.0 / ppp, v_px.1 / ppp),
                        egui::pos2(cp2_px.0 / ppp, cp2_px.1 / ppp),
                    ],
                    egui::Stroke::new(
                        1.5,
                        egui::Color32::from_rgba_unmultiplied(50, 150, 255, 200),
                    ),
                );
            }

            const HANDLE_R: f32 = 6.0;
            let cp1_active =
                editor.bezier_hover_cp == Some(0) || editor.bezier_dragging_cp == Some(0);
            let cp2_active =
                editor.bezier_hover_cp == Some(1) || editor.bezier_dragging_cp == Some(1);
            let cp1_color = if cp1_active {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_rgb(255, 150, 50)
            };
            let cp2_color = if cp2_active {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_rgb(50, 150, 255)
            };

            let cp1_pos = egui::pos2(cp1_px.0 / ppp, cp1_px.1 / ppp);
            let cp2_pos = egui::pos2(cp2_px.0 / ppp, cp2_px.1 / ppp);
            painter.circle_stroke(cp1_pos, HANDLE_R, egui::Stroke::new(2.0, cp1_color));
            painter.circle_filled(cp1_pos, 3.5, cp1_color);
            painter.circle_stroke(cp2_pos, HANDLE_R, egui::Stroke::new(2.0, cp2_color));
            painter.circle_filled(cp2_pos, 3.5, cp2_color);
        }

        if editor.bezier_dialog_open {
            crate::ui::dialogs::editing::draw_bezier_panel(root_ui, editor, commands, canvas_rect);
        }
    }

    // --- Startup & global dialogs ---
    if project.needs_startup_dialog {
        crate::ui::dialogs::editing::draw_select_pidb_dialog(root_ui, commands);
    }
    crate::ui::dialogs::files::draw_file_operation_dialog(root_ui, editor, project, commands);
    crate::ui::dialogs::files::draw_vertical_exaggeration_dialog(root_ui, editor, canvas_rect);
    crate::ui::dialogs::editing::draw_move_to_layer_dialog(root_ui, editor, project, commands);
    crate::ui::dialogs::editing::draw_set_selection_z_dialog(root_ui, editor, commands);
    crate::ui::dialogs::editing::draw_move_layer_dialog(root_ui, editor, project, commands);

    // --- Canvas right-click context menu ---
    if editor.canvas_context_menu_open
        && let Some((px, py)) = editor.canvas_context_menu_px
    {
        crate::ui::dialogs::editing::draw_right_click_context(
            root_ui,
            editor,
            project,
            commands,
            &mut geometry_dirty,
            document,
            px,
            py,
        );
    }

    // --- Tool-specific dialogs ---

    // Create Layer
    if editor.active_tool == ActiveTool::NewLayer {
        crate::ui::dialogs::editing::draw_create_layer_dialog(
            root_ui,
            commands,
            editor,
            project,
            canvas_rect,
        );
    }

    // Rename Layer
    if editor.renaming_layer.is_some() {
        crate::ui::dialogs::editing::draw_rename_layer_dialog(root_ui, commands, editor);
    }

    // Offset Element
    if editor.active_tool == ActiveTool::OffsetElement
        && !editor.offset_dialog_open
        && !editor.offset_awaiting_side_pick
    {
        commands.push(UiCommand::OpenOffsetDialog);
    }
    if editor.offset_dialog_open {
        crate::ui::dialogs::editing::draw_offset_dialog(root_ui, commands, editor, canvas_rect);
    }
    if editor.offset_awaiting_side_pick && !editor.offset_preview_screen_px.is_empty() {
        let ppp = root_ui.ctx().pixels_per_point();
        let pts: Vec<egui::Pos2> = editor
            .offset_preview_screen_px
            .iter()
            .map(|(x, y)| egui::pos2(x / ppp, y / ppp))
            .collect();
        let src_pts: Vec<egui::Pos2> = editor
            .offset_source_screen_px
            .iter()
            .map(|(x, y)| egui::pos2(x / ppp, y / ppp))
            .collect();
        let painter = root_ui.painter().with_clip_rect(canvas_rect);
        let yellow = egui::Color32::from_rgb(255, 220, 0);
        let guide = egui::Stroke::new(
            2.0,
            egui::Color32::from_rgba_unmultiplied(255, 230, 40, 220),
        );
        for (from, to) in src_pts.iter().zip(pts.iter()) {
            for seg in dashed_line_segments(*from, *to, 6.0, 4.0) {
                painter.line_segment(seg, guide);
            }
        }
        let stroke = egui::Stroke::new(2.0, yellow);
        let n = pts.len();
        for i in 0..n {
            let next = if editor.offset_preview_closed {
                (i + 1) % n
            } else {
                i + 1
            };
            if next < n {
                painter.line_segment([pts[i], pts[next]], stroke);
            }
        }
    }

    // Road tool
    if editor.active_tool == ActiveTool::MakeRoad && !editor.road_dialog_open {
        editor.road_dialog_open = true;
    }
    if editor.road_dialog_open {
        crate::ui::dialogs::editing::draw_road_dialog(root_ui, commands, editor, canvas_rect);
    }
    if let Some(label) = viewport_label_text(editor) {
        ViewportLabel::new("viewport_tool_label", label.text, canvas_rect)
            .style(label.style)
            .show(root_ui.ctx());
    }
    // Batter Berm
    if editor.active_tool == ActiveTool::BatterBermOffset && !editor.batter_berm_dialog_open {
        commands.push(UiCommand::OpenBatterBermDialog);
    }
    if editor.batter_berm_dialog_open {
        crate::ui::dialogs::editing::draw_batter_berm_dialog(
            root_ui,
            commands,
            editor,
            canvas_rect,
        );
    }
    // Relimit Line
    if editor.active_tool == ActiveTool::RelimitLine
        && !editor.relimit_dialog_open
        && !editor.relimit_awaiting_source_pick
        && !editor.relimit_waiting_for_pick
        && !editor.relimit_confirming_end
    {
        commands.push(UiCommand::OpenRelimitDialog);
    }
    if editor.relimit_dialog_open {
        crate::ui::dialogs::editing::draw_relimit_dialog(root_ui, commands, editor, canvas_rect);
    }
    // Sphere on the chosen resize endpoint (absolute / relative modes)
    if let Some(ep_px) = editor.relimit_resize_end_px {
        let ppp = root_ui.ctx().pixels_per_point();
        let pos = egui::pos2(ep_px.0 / ppp, ep_px.1 / ppp);
        root_ui.painter().with_clip_rect(canvas_rect).circle_filled(
            pos,
            7.0,
            egui::Color32::from_rgb(255, 220, 0),
        );
    }
    if editor.relimit_waiting_for_pick && !editor.relimit_hover_target_screen_px.is_empty() {
        let ppp = root_ui.ctx().pixels_per_point();
        let pts: Vec<egui::Pos2> = editor
            .relimit_hover_target_screen_px
            .iter()
            .map(|(x, y)| egui::pos2(x / ppp, y / ppp))
            .collect();
        let painter = root_ui.painter().with_clip_rect(canvas_rect);
        let stroke = egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 220, 0));
        let n = pts.len();
        for i in 0..n.saturating_sub(1) {
            painter.line_segment([pts[i], pts[i + 1]], stroke);
        }
        if editor.relimit_hover_target_closed && n >= 2 {
            painter.line_segment([pts[n - 1], pts[0]], stroke);
        }
    }
    if editor.relimit_confirming_end
        && let (Some(from), Some(to)) =
            (editor.relimit_preview_from_px, editor.relimit_preview_to_px)
    {
        let ppp = root_ui.ctx().pixels_per_point();
        let color = if editor.relimit_preview_is_extension {
            egui::Color32::from_rgb(255, 220, 0) // yellow = growing
        } else {
            egui::Color32::from_rgb(220, 60, 60) // red = shrinking
        };
        root_ui.painter().with_clip_rect(canvas_rect).line_segment(
            [
                egui::pos2(from.0 / ppp, from.1 / ppp),
                egui::pos2(to.0 / ppp, to.1 / ppp),
            ],
            egui::Stroke::new(2.5, color),
        );
    }

    // Text editing
    if editor.text_editing_enabled {
        crate::ui::dialogs::editing::draw_text_edit_dialog(
            root_ui,
            commands,
            editor,
            &mut geometry_dirty,
            canvas_rect,
        );
    }

    // Polygon finish (MakePoly)
    if editor.poly_finish_dialog
        && let Some((px, py)) = editor.poly_finish_dialog_px
    {
        crate::ui::dialogs::editing::draw_finish_polygon_dialog(root_ui, commands, editor, px, py);
    }

    // --- Canvas overlays ---

    // Orbit marker (clipped to the 3D viewport)
    if let Some((ox, oy)) = overlay.orbit_marker {
        crate::ui::elements::cursors::draw_orbit_marker(root_ui, ox, oy, canvas_rect);
    }

    if editor.show_world_axis_gizmo {
        crate::ui::elements::cursors::draw_orientation_gizmo(
            root_ui,
            canvas_rect,
            overlay.camera_forward,
            overlay.camera_up,
        );
    }
    if editor.show_view_cube {
        crate::ui::elements::cursors::draw_view_cube(
            root_ui,
            canvas_rect,
            overlay.camera_forward,
            overlay.camera_up,
            commands,
        );
    }

    // Colour-scale legend for the first visible, grade-coloured block model.
    // Deliberately independent of `editor.selected_handles`: a stray canvas
    // click clearing the selection (e.g. while interacting with the
    // legend's own dropdown) must not make the legend itself disappear.
    let legend_model = block_models
        .iter()
        .find(|model| model.visible && model.active_numeric_variable.is_some());
    if let Some(model) = legend_model {
        // A variable with no usable range (e.g. every rendered block is the
        // sentinel/default value) still needs its dropdown shown — losing
        // the whole widget here would strand the user on a variable they
        // can't switch away from through the UI.
        let range = crate::ui::elements::block_model::active_color_scale(model)
            .map(|(_, min, max)| (min, max));
        crate::ui::widgets::viewport::ColorScaleLegend::new(
            ("block_model_color_scale_legend", model.id),
            model,
            range,
            canvas_rect,
        )
        .show(root_ui.ctx(), commands);
    }

    // Exit confirmation
    if editor.exit_confirm_open {
        crate::ui::dialogs::confirmations::draw_exit_confirm_dialog(root_ui, commands, editor);
    }

    // Delete selection confirmation
    if editor.delete_confirm_open {
        crate::ui::dialogs::confirmations::draw_delete_confirm_dialog(root_ui, commands, editor);
    }

    // Delete layer confirmation
    if editor.pending_delete_layer.is_some() {
        crate::ui::dialogs::confirmations::draw_delete_layer_confirm_dialog(
            root_ui, commands, editor,
        );
    }

    // Dirty PIDB close confirmation
    if editor.pending_close_project.is_some() {
        crate::ui::dialogs::confirmations::draw_close_project_dialog(
            root_ui, commands, editor, project,
        );
    }

    // Dirty-layer unload confirmation
    if !editor.pending_unload_queue.is_empty() {
        crate::ui::dialogs::confirmations::draw_pending_unload_layer_dialog(
            root_ui, commands, editor,
        );
    }

    // Load-all triangulations folder confirmation
    if editor.confirm_load_all_folder.is_some() {
        crate::ui::dialogs::confirmations::draw_confirm_load_all_folder_dialog(
            root_ui, commands, editor,
        );
    }

    // Close unsaved triangulation confirmation
    if editor.tri_close_unsaved.is_some() {
        crate::ui::dialogs::confirmations::draw_close_unsaved_tri_dialog(root_ui, commands, editor);
    }

    // Create Triangulation (always mark geometry dirty while open)
    if editor.tri_create_open {
        geometry_dirty = true;
        crate::ui::dialogs::triangulation::draw_tri_create_main_dialog(
            root_ui, editor, document, commands,
        );
    }

    if editor.tri_create_failure.is_some() {
        crate::ui::dialogs::triangulation::draw_tri_create_failure_dialog(
            root_ui, editor, commands,
        );
    }

    if editor.tri_cut_poly_open {
        crate::ui::dialogs::triangulation::draw_cut_poly_dialog(
            root_ui, editor, document, project, commands,
        );
    }

    if editor.tri_cut_z_open {
        crate::ui::dialogs::triangulation::draw_cut_z_dialog(root_ui, editor, project, commands);
    }

    if editor.tri_cut_surface_open {
        crate::ui::dialogs::triangulation::draw_cut_surface_dialog(
            root_ui, editor, project, commands,
        );
    }

    if editor.tri_cut_pitshell_open {
        crate::ui::dialogs::triangulation::draw_cut_topology_to_pit_shell_dialog(
            root_ui, editor, project, commands,
        );
    }

    if editor.tri_include_solid_open {
        crate::ui::dialogs::triangulation::draw_include_solid_dialog(
            root_ui, editor, project, commands,
        );
    }

    if editor.tri_contour_open {
        crate::ui::dialogs::triangulation::draw_contour_dialog(root_ui, editor, project, commands);
    }
    if editor.ore_triangulation_open {
        crate::ui::elements::block_model::draw_ore_triangulation_dialog(
            root_ui,
            editor,
            block_models,
            commands,
        );
    }

    geometry_dirty
}

/// Generate dashed line segments along the four edges of a rect.
/// Returns pairs of `[Pos2; 2]` suitable for `painter.line_segment`.
fn dashed_rect_segments(r: egui::Rect, dash: f32, gap: f32) -> Vec<[egui::Pos2; 2]> {
    let mut segs = Vec::new();
    let corners = [
        r.left_top(),
        r.right_top(),
        r.right_bottom(),
        r.left_bottom(),
    ];
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let total = (b - a).length();
        let step = dash + gap;
        let mut t = 0.0f32;
        while t < total {
            let t_end = (t + dash).min(total);
            let frac_a = t / total;
            let frac_b = t_end / total;
            segs.push([a + (b - a) * frac_a, a + (b - a) * frac_b]);
            t += step;
        }
    }
    segs
}

fn dashed_line_segments(
    start: egui::Pos2,
    end: egui::Pos2,
    dash: f32,
    gap: f32,
) -> Vec<[egui::Pos2; 2]> {
    let delta = end - start;
    let length = delta.length();
    if length <= 0.0 {
        return Vec::new();
    }
    let dir = delta / length;
    let mut segments = Vec::new();
    let mut t = 0.0;
    while t < length {
        let a = start + dir * t;
        let b = start + dir * (t + dash).min(length);
        segments.push([a, b]);
        t += dash + gap;
    }
    segments
}

/// Build an `egui::Visuals` set with selection styling applied to the given theme.
fn theme_visuals(dark_mode: bool, selection_color: egui::Color32) -> egui::Visuals {
    let mut visuals = if dark_mode {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    visuals.selection.bg_fill = selection_color.gamma_multiply(0.35);
    visuals.selection.stroke.color = selection_color;
    visuals
}
