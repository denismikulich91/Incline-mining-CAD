//! Four toolbar panels: top (layer/Z/line/fill settings), left (drawing tools),
//! right (viewport controls), bottom (selection actions + cursor mode).

use crate::ui::{
    EditorState, UiProjectView, color32_to_rgba, rgba_to_color32,
    state::{ActiveTool, CursorMode, EditorAction, UiCommand},
    themed_icon, unthemed_icon,
    widgets::toolbar::{ColorSquarePicker, HatchPicker, ToolbarButton},
};

/// Draw the top toolbar (save, undo/redo, layer combo, Z level, line/fill colors, weight, hatch).
pub(crate) fn draw_top_toolbar(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
    document: &mut crate::ui::Document,
    can_undo: bool,
    can_redo: bool,
) -> egui::Rect {
    egui::Panel::top("top_tools_strip")
        .resizable(false)
        .default_size(34.0)
        .show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                let has_dirty = project.projects.iter().any(|p| p.dirty);
                let save = ui.add_enabled(
                    has_dirty,
                    ToolbarButton::new(
                        egui::Image::new(unthemed_icon!("save_floppy.svg")),
                        "Save all layers",
                    ),
                );
                if save.clicked() {
                    commands.push(UiCommand::SaveAllPidbs);
                }

                let undo_btn = ui.add_enabled(
                    can_undo,
                    ToolbarButton::new(egui::Image::new(themed_icon!(ui, "undo.svg")), "Undo"),
                );
                if undo_btn.clicked() {
                    commands.push(UiCommand::Undo);
                }
                let redo_btn = ui.add_enabled(
                    can_redo,
                    ToolbarButton::new(egui::Image::new(themed_icon!(ui, "redo.svg")), "Redo"),
                );
                if redo_btn.clicked() {
                    commands.push(UiCommand::Redo);
                }
                ui.separator();
                ui.label("Layer: ");
                let selected_layer = editor
                    .active_layer
                    .and_then(|id| document.layer(id))
                    .map(|layer| layer.name.as_str())
                    .unwrap_or("None");
                const MAX_LAYER_DISPLAY: usize = 22;
                let layer_display: String = if selected_layer.chars().count() > MAX_LAYER_DISPLAY {
                    format!(
                        "{}…",
                        selected_layer
                            .chars()
                            .take(MAX_LAYER_DISPLAY - 1)
                            .collect::<String>()
                    )
                } else {
                    selected_layer.to_string()
                };
                egui::ComboBox::from_id_salt("layer_combo_box")
                    .selected_text(layer_display)
                    .width(200.)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut editor.active_layer, None, "None");
                        for layer in document.layers() {
                            ui.selectable_value(
                                &mut editor.active_layer,
                                Some(layer.id),
                                &layer.name,
                            );
                        }
                    });

                ui.label("Z: ");
                let z_resp =
                    ui.add(egui::TextEdit::singleline(&mut editor.z_input).desired_width(80.));
                if z_resp.changed()
                    && let Ok(z) = editor.z_input.parse::<f64>()
                    && z.is_finite()
                {
                    editor.z_level = z;
                }

                shifted_up(ui, 3.0, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;

                        let mut line_c32 = rgba_to_color32(editor.tool_line_color);

                        if ColorSquarePicker::new(&mut line_c32).show(ui).changed() {
                            editor.tool_line_color = color32_to_rgba(line_c32);
                            editor.tool_fill_color = color32_to_rgba(line_c32);
                        }

                        HatchPicker::new(
                            &mut editor.tool_hatch,
                            rgba_to_color32(editor.tool_fill_color),
                        )
                        .show(ui);
                    });
                });
            });
        })
        .response
        .rect
}

/// Draw the left toolbar (New layer, MakePoint, MakeLine, MakePoly, MakeText, Move, OffsetElement,
/// RelimitLine, FuseIntoPolygon, ExplodePolygon, DeleteElement).
pub(crate) fn draw_left_toolbar(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    editing_enabled: bool,
    project_active: bool,
) -> egui::Rect {
    egui::Panel::left("left_tools_strip")
        .resizable(false)
        .default_size(32.0)
        .show(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_enabled_ui(project_active, |ui| {
                    tool_button(
                        ui,
                        egui::Image::new(unthemed_icon!("new_layer.svg")),
                        "New layer",
                        editor,
                        ActiveTool::NewLayer,
                    );

                    ui.add_space(6.0);
                    ui.separator();
                });

                ui.add_enabled_ui(editing_enabled, |ui| {
                    ui.add_space(6.0);

                    tool_button(
                        ui,
                        egui::Image::new(themed_icon!(ui, "make_point.svg")),
                        "Make Point",
                        editor,
                        ActiveTool::MakePoint,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(themed_icon!(ui, "make_line.svg")),
                        "Make Line",
                        editor,
                        ActiveTool::MakeLine,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(themed_icon!(ui, "make_poly.svg")),
                        "Make Polygon",
                        editor,
                        ActiveTool::MakePoly,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(unthemed_icon!("make_text.svg")),
                        "Make Text",
                        editor,
                        ActiveTool::MakeText,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(unthemed_icon!("make_road.svg")),
                        "Make Roads",
                        editor,
                        ActiveTool::MakeRoad,
                    );

                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(6.0);

                    tool_button(
                        ui,
                        egui::Image::new(themed_icon!(ui, "pan.svg")),
                        "Move",
                        editor,
                        ActiveTool::Move,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(themed_icon!(ui, "offset.svg")),
                        "Offset",
                        editor,
                        ActiveTool::OffsetElement,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(unthemed_icon!("pit_design.svg")),
                        "Auto-Bench",
                        editor,
                        ActiveTool::BatterBermOffset,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(themed_icon!(ui, "rebase.svg")),
                        "Relimit Line",
                        editor,
                        ActiveTool::RelimitLine,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(themed_icon!(ui, "fuse.svg")),
                        "Fuse Lines Into Polygon",
                        editor,
                        ActiveTool::FuseIntoPolygon,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(themed_icon!(ui, "chamfer.svg")),
                        "Chamfer Polygon Corners",
                        editor,
                        ActiveTool::Chamfer,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(themed_icon!(ui, "bezier.svg")),
                        "Bezier Polygon",
                        editor,
                        ActiveTool::Bezier,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(unthemed_icon!("boom.svg")),
                        "Explode Polygon to Lines",
                        editor,
                        ActiveTool::ExplodePolygon,
                    );

                    tool_button(
                        ui,
                        egui::Image::new(unthemed_icon!("delete_element.svg")),
                        "Delete",
                        editor,
                        ActiveTool::DeleteElement,
                    );
                });
            });
        })
        .response
        .rect
}

/// Draw the right toolbar (Reset view, Zoom to extents, Vertical exaggeration, X-Ray).
pub(crate) fn draw_right_toolbar(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    commands: &mut Vec<UiCommand>,
) -> egui::Rect {
    egui::Panel::right("right_tools_strip")
        .resizable(false)
        .default_size(32.0)
        .show(ui, |ui| {
            ui.vertical_centered(|ui| {
                // Reset view button
                let response = ui.add(ToolbarButton::new(
                    egui::Image::new(unthemed_icon!("reset_view.svg")),
                    "Reset view",
                ));
                if response.clicked() {
                    commands.push(UiCommand::ResetView);
                }

                // Zoom to bounds button
                let response = ui.add(ToolbarButton::new(
                    egui::Image::new(unthemed_icon!("globe.svg")),
                    "Zoom to extents",
                ));
                if response.clicked() {
                    commands.push(UiCommand::ZoomToExtents);
                }

                // Open Vertical Exaggeration menu button
                let response = ui.add(
                    ToolbarButton::new(
                        egui::Image::new(unthemed_icon!("exagerated_topology.svg")),
                        format!(
                            "Vertical Exaggeration ({:.2}×)",
                            editor.vertical_exaggeration
                        ),
                    )
                    .selected(editor.vertical_exaggeration != 1.0),
                );
                if response.clicked() {
                    editor.vertical_exaggeration_input =
                        format!("{}", editor.vertical_exaggeration);
                    editor.vertical_exaggeration_dialog_open = true;
                }

                // Enable x-ray vision
                let response = ui.add(
                    ToolbarButton::new(
                        egui::Image::new(unthemed_icon!("xray.svg")),
                        if editor.xray_enabled {
                            "Disable X-Ray Vision"
                        } else {
                            "Enable X-Ray Vision"
                        },
                    )
                    .selected(editor.xray_enabled),
                );
                if response.clicked() {
                    editor.xray_enabled = !editor.xray_enabled;
                }

                // Toggle fly mode
                let response = ui.add(
                    ToolbarButton::new(
                        egui::Image::new(unthemed_icon!("aeroplane.svg")),
                        if editor.fly_mode_enabled {
                            "Disable Flying Mode"
                        } else {
                            "Enable Flying Mode"
                        },
                    )
                    .selected(editor.fly_mode_enabled),
                );
                if response.clicked() {
                    editor.fly_mode_enabled = !editor.fly_mode_enabled;
                }
            });
        })
        .response
        .rect
}

/// Draw the bottom toolbar (reveal/hide/freeze selection, cursor mode, measure distance).
///
/// Updates `geometry_dirty` based on which selection actions were triggered.
pub(crate) fn draw_bottom_toolbar(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    geometry_dirty: &mut bool,
    commands: &mut Vec<UiCommand>,
) -> egui::Rect {
    egui::Panel::bottom("bottom_tools_strip")
        .resizable(false)
        .default_size(32.0)
        .show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                let reveal_all = editor_action_button(
                    ui,
                    egui::Image::new(unthemed_icon!("reveal_all.svg")),
                    "Reveal all elements",
                    editor,
                    EditorAction::RevealAll,
                );
                if reveal_all {
                    commands.push(UiCommand::RevealAllTriangulations);
                }
                *geometry_dirty |= reveal_all;

                *geometry_dirty |= editor_action_button(
                    ui,
                    egui::Image::new(themed_icon!(ui, "hide_selection.svg")),
                    "Hide selection",
                    editor,
                    EditorAction::HideSelection,
                );

                *geometry_dirty |= editor_action_button(
                    ui,
                    egui::Image::new(unthemed_icon!("freeze_selection.svg")),
                    "Freeze selection",
                    editor,
                    EditorAction::FreezeSelection,
                );

                ui.separator();

                cursor_mode_button(
                    ui,
                    egui::Image::new(themed_icon!(ui, "select.svg")),
                    "Cursor: Regular",
                    editor,
                    CursorMode::Select,
                );

                cursor_mode_button(
                    ui,
                    egui::Image::new(themed_icon!(ui, "snap_to_surface.svg")),
                    "Cursor: Snap to surface",
                    editor,
                    CursorMode::SnapToSurface,
                );

                cursor_mode_button(
                    ui,
                    egui::Image::new(themed_icon!(ui, "snap_to_line.svg")),
                    "Cursor: Snap to line",
                    editor,
                    CursorMode::SnapToLine,
                );

                cursor_mode_button(
                    ui,
                    egui::Image::new(themed_icon!(ui, "snap_to_point.svg")),
                    "Cursor: Snap to point",
                    editor,
                    CursorMode::SnapToPoint,
                );

                ui.separator();

                tool_button(
                    ui,
                    egui::Image::new(themed_icon!(ui, "mesure_distance.svg")),
                    "Measure distance",
                    editor,
                    ActiveTool::MeasureDistance,
                );
            });
        })
        .response
        .rect
}

/// Draw a single tool button; toggles `editor.active_tool` on click.
pub(crate) fn tool_button(
    ui: &mut egui::Ui,
    icon: egui::Image<'static>,
    tooltip: &str,
    editor: &mut EditorState,
    tool: ActiveTool,
) -> egui::Response {
    let selected = editor.active_tool == tool;
    let response = ui.add(ToolbarButton::new(icon, tooltip).selected(selected));

    if response.clicked() {
        let previous_tool = editor.active_tool;
        if selected {
            editor.active_tool = ActiveTool::None;
        } else {
            editor.active_tool = tool;
        }

        // Cancel in-progress tool state when switching away from tools with panels/previews.
        if previous_tool == ActiveTool::OffsetElement
            && editor.active_tool != ActiveTool::OffsetElement
        {
            editor.offset_dialog_open = false;
            editor.offset_target_id = None;
            editor.offset_awaiting_side_pick = false;
            editor.offset_preview_world.clear();
            editor.offset_source_world.clear();
        }
        if previous_tool == ActiveTool::RelimitLine && editor.active_tool != ActiveTool::RelimitLine
        {
            editor.relimit_dialog_open = false;
            editor.relimit_source_id = None;
            editor.relimit_awaiting_source_pick = false;
            editor.relimit_waiting_for_pick = false;
            editor.relimit_confirming_end = false;
        }
        if previous_tool == ActiveTool::BatterBermOffset
            && editor.active_tool != ActiveTool::BatterBermOffset
        {
            editor.batter_berm_dialog_open = false;
            editor.batter_berm_target_id = None;
            editor.batter_berm_rings_world.clear();
            editor.batter_berm_source_world.clear();
            editor.batter_berm_guides_world.clear();
            editor.batter_berm_rings_screen_px.clear();
            editor.batter_berm_source_screen_px.clear();
            editor.batter_berm_guides_screen_px.clear();
            editor.tool_highlight_id = None;
        }
        if previous_tool == ActiveTool::MakeRoad && editor.active_tool != ActiveTool::MakeRoad {
            editor.road_dialog_open = false;
            editor.pending_stroke.clear();
            editor.road_preview_left_world.clear();
            editor.road_preview_right_world.clear();
            editor.road_preview_left_screen_px.clear();
            editor.road_preview_right_screen_px.clear();
            editor.road_preview_center_screen_px.clear();
        }
        if previous_tool == ActiveTool::Bezier && editor.active_tool != ActiveTool::Bezier {
            editor.bezier_poly_id = None;
            editor.bezier_selected_verts = [None; 2];
            editor.bezier_cp1 = [0.0; 3];
            editor.bezier_cp2 = [0.0; 3];
            editor.bezier_poly_verts_screen_px.clear();
            editor.bezier_cp1_screen_px = None;
            editor.bezier_cp2_screen_px = None;
            editor.bezier_preview_screen_px.clear();
            editor.bezier_dragging_cp = None;
            editor.bezier_hover_cp = None;
            editor.bezier_dialog_open = false;
        }

        if editor.active_tool == ActiveTool::NewLayer {
            editor.new_layer_name = "design".to_owned();
        }
        if tool == ActiveTool::MeasureDistance {
            editor.measurement_start = None;
            editor.measurement_end = None;
        }
    }

    response
}

/// Draw a cursor mode button; sets `editor.cursor_mode` on click.
pub(crate) fn cursor_mode_button(
    ui: &mut egui::Ui,
    icon: egui::Image<'static>,
    tooltip: &str,
    editor: &mut EditorState,
    mode: CursorMode,
) -> egui::Response {
    let selected = editor.cursor_mode == mode;
    let response = ui.add(ToolbarButton::new(icon, tooltip).selected(selected));

    if response.clicked() {
        editor.cursor_mode = mode;
    }

    response
}

/// Draw a selection action button.  Returns `true` if the action was applied.
pub(crate) fn editor_action_button(
    ui: &mut egui::Ui,
    icon: egui::Image<'static>,
    tooltip: &str,
    editor: &mut EditorState,
    action: EditorAction,
) -> bool {
    let response = ui.add(ToolbarButton::new(icon, tooltip));

    response.clicked() && editor.apply_action(action)
}

fn shifted_up(ui: &mut egui::Ui, amount: f32, add_contents: impl FnOnce(&mut egui::Ui)) {
    let rect = ui.available_rect_before_wrap();
    let shifted_rect = rect.translate(egui::vec2(0.0, -amount));

    ui.scope_builder(egui::UiBuilder::new().max_rect(shifted_rect), |ui| {
        add_contents(ui)
    });
}
