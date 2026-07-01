//! Object editing and viewport tool dialogs.

use crate::{
    model::{Document, FillStyle, ObjectColor, ObjectId, RoadShape},
    rendering::color::{color32_to_rgba, rgba_to_color32},
    ui::{
        state::{
            ActiveTool, BatterBermMode, EditorState, HeightMode, OffsetMeasure, OffsetProjection,
            RelimitMode, TrimEnd, UiCommand, UiProjectView,
        },
        widgets::menu::{
            DragableMenu, MenuField, MenuFieldColor32, MenuFieldCombo, MenuFieldF32, MenuFieldF64,
            MenuFieldRgba, MenuFieldText, MenuFieldU32,
        },
        widgets::viewport::ViewportDockPanel,
    },
};

fn fill_style_label(style: FillStyle) -> &'static str {
    match style {
        FillStyle::Clear => "Clear",
        FillStyle::Crosses => "Crosses",
        FillStyle::Slashes => "Slashes",
        FillStyle::Solid => "Solid",
    }
}

/// Draw the canvas right-click context menu for selected objects and triangulations.
///
/// Provides controls for line/fill colour, polyline open/closed toggle,
/// fill type, and line weight.  Updates `geometry_dirty` when changes are made.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_right_click_context(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
    geometry_dirty: &mut bool,
    document: &Document,
    px: f32,
    py: f32,
) {
    let ppp = ui.ctx().pixels_per_point();
    let pos = egui::pos2(px / ppp + 4.0, py / ppp + 4.0);
    let mut open = true;
    DragableMenu::new("Properties")
        .open(&mut open)
        .default_pos(pos)
        .min_width(150.0)
        .show(ui.ctx(), |ui| {
            ui.set_max_width(220.);
            // Gather selected document objects
            let selected_obj_ids: Vec<ObjectId> = editor
                .selected_handles
                .iter()
                .filter_map(|&h| match h {
                    crate::model::SceneEntityId::Object(id) => Some(id),
                    crate::model::SceneEntityId::Triangulation(_)
                    | crate::model::SceneEntityId::BlockModel(_) => None,
                })
                .collect();

            let selected_polys: Vec<ObjectId> = selected_obj_ids
                .iter()
                .copied()
                .filter(|&id| {
                    matches!(
                        document.get_object(id),
                        Some(crate::model::Object::Polyline { .. })
                    )
                })
                .collect();

            let has_doc_objects = !selected_obj_ids.is_empty();
            let has_polys = !selected_polys.is_empty();
            let has_tri = project.active_triangulation_for_menu.is_some()
                && editor
                    .selected_handles
                    .iter()
                    .any(|&h| matches!(h, crate::model::SceneEntityId::Triangulation(_)));

            // --- Color picker ---
            if has_doc_objects || has_tri {
                if has_doc_objects {
                    let first_line_color = selected_obj_ids
                        .first()
                        .and_then(|&id| document.get_object(id))
                        .map(|obj| document.object_rgba(obj))
                        .unwrap_or([0.0; 4]);
                    let mut color32 = rgba_to_color32(first_line_color);
                    let color_resp = MenuFieldColor32::new("Line Color", &mut color32).show(ui);
                    if color_resp.drag_stopped() || (color_resp.changed() && !color_resp.dragged())
                    {
                        let rgba = color32_to_rgba(color32);
                        let new_color = ObjectColor::Fixed(rgba);
                        commands.push(UiCommand::BatchSetObjectColor(
                            selected_obj_ids.clone(),
                            new_color,
                        ));
                        *geometry_dirty = true;
                    }
                }

                if has_tri
                    && let Some((tri_id, mut face_color)) = project.active_triangulation_for_menu
                {
                    let mut color32 = rgba_to_color32(face_color);
                    if MenuFieldColor32::new("Face Color", &mut color32)
                        .show(ui)
                        .changed()
                    {
                        face_color = color32_to_rgba(color32);
                        commands.push(UiCommand::SetTriangulationColor(tri_id, face_color));
                        *geometry_dirty = true;
                    }
                }

                ui.separator();
            }

            // --- Polyline-specific controls ---
            if has_polys {
                let first_poly = selected_polys
                    .first()
                    .and_then(|&id| document.get_object(id));
                let (first_closed, first_fill, first_line_weight) =
                    if let Some(crate::model::Object::Polyline {
                        closed,
                        fill,
                        line_weight,
                        ..
                    }) = first_poly
                    {
                        (*closed, *fill, *line_weight)
                    } else {
                        (false, FillStyle::Clear, 0.0)
                    };

                // Open / Closed toggle
                let mut is_closed = first_closed;
                MenuField::new("Shape").show(ui, |ui, _| {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut is_closed, false, "⟵⟶ Open");
                        ui.selectable_value(&mut is_closed, true, "⬛ Closed");
                    })
                    .response
                });
                if is_closed != first_closed {
                    commands.push(UiCommand::BatchSetPolylineClosed(
                        selected_polys.clone(),
                        is_closed,
                    ));
                    *geometry_dirty = true;
                }

                ui.separator();

                // Fill dropdown
                let mut fill = first_fill;
                let old_fill = fill;
                MenuFieldCombo::new(
                    "ctx_fill_combo",
                    "Fill",
                    &mut fill,
                    fill_style_label(first_fill),
                    [
                        FillStyle::Clear,
                        FillStyle::Crosses,
                        FillStyle::Slashes,
                        FillStyle::Solid,
                    ]
                    .map(|style| (style, fill_style_label(style).into())),
                )
                .width(139.0)
                .show(ui);
                if fill != old_fill {
                    commands.push(UiCommand::BatchSetObjectFill(selected_polys.clone(), fill));
                    *geometry_dirty = true;
                }

                let mut lw = first_line_weight;
                let lw_resp = MenuFieldF32::new("Line Weight", &mut lw, 0.1..=20.0)
                    .speed(0.1)
                    .show(ui);
                if lw_resp.drag_stopped() || (lw_resp.changed() && !lw_resp.dragged()) {
                    commands.push(UiCommand::BatchSetPolylineLineWeight(
                        selected_polys.clone(),
                        lw,
                    ));
                    *geometry_dirty = true;
                }

                ui.separator();
            }

            if ui.button("✖  Close").clicked() {
                commands.push(UiCommand::CloseCanvasContextMenu);
            }
        });
    if !open {
        commands.push(UiCommand::CloseCanvasContextMenu);
    }
}

/// Draw the startup dialog prompting the user to open or create a .pidb file.
pub(crate) fn draw_select_pidb_dialog(ui: &mut egui::Ui, commands: &mut Vec<UiCommand>) {
    DragableMenu::new("Open or create a PIDB")
        .min_width(300.0)
        .show(ui.ctx(), |ui| {
            ui.label("An active PIDB file and layer are required before editing.");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("New PIDB").clicked() {
                    commands.push(UiCommand::NewPidb);
                }
                if ui.button("Open PIDB").clicked() {
                    commands.push(UiCommand::OpenPidb);
                }
            });
        });
}

/// Draw the "Create a new layer" dialog (opened when NewLayer tool is active).
pub(crate) fn draw_create_layer_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
    project: &UiProjectView,
    viewport_rect: egui::Rect,
) {
    if editor
        .new_layer_project_index
        .is_none_or(|index| !project.projects.iter().any(|entry| entry.index == index))
    {
        editor.new_layer_project_index = project
            .active_index
            .or_else(|| project.projects.first().map(|entry| entry.index));
    }

    ViewportDockPanel::new("create_layer_panel", "Create a new layer", viewport_rect)
        .min_width(220.0)
        .show(ui.ctx(), |ui| {
            let selected_label = editor
                .new_layer_project_index
                .and_then(|index| project.projects.iter().find(|entry| entry.index == index))
                .map(|entry| entry.name.as_str())
                .unwrap_or("Choose a .pidb");
            MenuFieldCombo::new(
                "new_layer_project",
                "Save to",
                &mut editor.new_layer_project_index,
                selected_label,
                project
                    .projects
                    .iter()
                    .map(|entry| (Some(entry.index), entry.name.clone().into())),
            )
            .show(ui);
            let can_save = editor.new_layer_project_index.is_some()
                && !editor.new_layer_name.trim().is_empty();
            MenuFieldText::new("Layer name", &mut editor.new_layer_name)
                .hint_text("Required")
                .show(ui);
            let submitted = ui.input(|input| input.key_pressed(egui::Key::Enter));
            ui.horizontal(|ui| {
                if (submitted
                    || ui
                        .add_enabled(can_save, egui::Button::new("Create Layer"))
                        .clicked())
                    && can_save
                    && let Some(project_index) = editor.new_layer_project_index
                {
                    commands.push(UiCommand::CreateLayer {
                        project_index,
                        name: editor.new_layer_name.trim().to_string(),
                    });
                    editor.new_layer_project_index = None;
                    editor.active_tool = ActiveTool::None;
                }
                if ui.button("Cancel").clicked() {
                    editor.new_layer_project_index = None;
                    editor.active_tool = ActiveTool::None;
                }
            });
        });
}

/// Draw the rename layer floating dialog.
pub(crate) fn draw_rename_layer_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
) {
    let Some((project_index, layer_id, _)) = editor.renaming_layer else {
        return;
    };
    // Work on a local copy of the name buffer to avoid borrow conflicts inside the closure.
    let mut name_buf = editor
        .renaming_layer
        .as_ref()
        .map(|(_, _, n)| n.clone())
        .unwrap_or_default();
    let mut close = false;
    let mut rename_to: Option<String> = None;
    let mut open = true;
    DragableMenu::new("Rename Layer")
        .open(&mut open)
        .show(ui.ctx(), |ui| {
            ui.set_max_width(230.);
            let response = MenuFieldText::new("New name", &mut name_buf)
                .width(160.0)
                .hint_text("Required")
                .show(ui);
            ui.horizontal(|ui| {
                let can_rename = !name_buf.trim().is_empty();
                let submitted =
                    response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if (submitted
                    || ui
                        .add_enabled(can_rename, egui::Button::new("Rename"))
                        .clicked())
                    && can_rename
                {
                    rename_to = Some(name_buf.trim().to_string());
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
        });
    // Write the edited buffer back.
    if let Some((_, _, ref mut buf)) = editor.renaming_layer {
        *buf = name_buf;
    }
    if let Some(new_name) = rename_to {
        commands.push(UiCommand::RenameLayer {
            project_index,
            layer_id,
            new_name,
        });
    } else if close || !open {
        editor.renaming_layer = None;
    }
}

/// Draw the text-editing properties popup (height, rotation, colour, content).
pub(crate) fn draw_text_edit_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
    geometry_dirty: &mut bool,
    viewport_rect: egui::Rect,
) {
    let Some(object_id) = editor.editing_labels_id else {
        return;
    };
    ViewportDockPanel::new("text_edit_panel", "Editing text", viewport_rect)
        .min_width(260.0)
        .show(ui.ctx(), |ui| {
            let response = MenuFieldText::new("Text", &mut editor.pending_text)
                .width(240.0)
                .hint_text("Text")
                .show(ui);
            if response.changed() {
                *geometry_dirty = true;
            }
            if editor.text_edit_focus_requested {
                response.request_focus();
                editor.text_edit_focus_requested = false;
            }
            *geometry_dirty |=
                MenuFieldF64::new("Height", &mut editor.pending_text_height, 0.001..=1.0e9)
                    .speed(0.25)
                    .max_decimals(3)
                    .show(ui)
                    .changed();
            *geometry_dirty |= MenuFieldF64::new(
                "Rotation",
                &mut editor.pending_text_rotation_degrees,
                f64::MIN..=f64::MAX,
            )
            .speed(1.0)
            .suffix("°")
            .show(ui)
            .changed();
            *geometry_dirty |= MenuFieldRgba::new("Color", &mut editor.pending_text_color)
                .show(ui)
                .changed();

            let apply_from_enter =
                response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            let cancel_from_escape = ui.input(|input| input.key_pressed(egui::Key::Escape));
            ui.horizontal(|ui| {
                let apply = ui.button("Apply").clicked() || apply_from_enter;
                let cancel = ui
                    .button(if editor.text_edit_created {
                        "Discard"
                    } else {
                        "Cancel"
                    })
                    .clicked()
                    || cancel_from_escape;
                if apply {
                    commands.push(UiCommand::CommitTextEdit(
                        object_id,
                        editor.pending_text.clone(),
                        editor.pending_text_height,
                        editor.pending_text_rotation_degrees,
                        editor.pending_text_color,
                    ));
                    editor.text_editing_enabled = false;
                } else if cancel {
                    commands.push(UiCommand::CancelTextEdit);
                    editor.text_editing_enabled = false;
                }
            });
        });
    editor.text_edit_position_frames = editor.text_edit_position_frames.saturating_sub(1);
}

/// Draw the polygon finish dialog (Close / Leave open / Cancel) near the cursor.
pub(crate) fn draw_finish_polygon_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
    px: f32,
    py: f32,
) {
    let ppp = ui.ctx().pixels_per_point();
    let pos = egui::pos2(px / ppp + 14.0, py / ppp + 14.0);
    let mut open = true;
    DragableMenu::new("Finish polygon")
        .open(&mut open)
        .default_pos(pos)
        .min_width(130.0)
        .show(ui.ctx(), |ui| {
            ui.set_max_width(280.0);
            if ui.button("⬛ Close polygon").clicked() {
                commands.push(UiCommand::FinishPolyClose);
                editor.poly_finish_dialog = false;
            }
            if ui.button("╌ Leave open").clicked() {
                commands.push(UiCommand::FinishPolyOpen);
                editor.poly_finish_dialog = false;
            }
            ui.add_space(2.0);
            ui.separator();
            if ui.button("✖ Cancel").clicked() {
                editor.poly_finish_dialog = false;
                editor.poly_finish_dialog_px = None;
            }
        });
    if !open {
        editor.poly_finish_dialog = false;
        editor.poly_finish_dialog_px = None;
    }
}

/// Draw the Offset Element dialog (horizontal berm or angled batter projection).
pub(crate) fn draw_offset_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
    viewport_rect: egui::Rect,
) {
    let Some(object_id) = editor.offset_target_id else {
        return;
    };

    ViewportDockPanel::new("offset_element_panel", "Offset Element", viewport_rect)
        .min_width(350.0)
        .show(ui.ctx(), |ui| {
            // Projection row
            MenuField::new("Projection").show(ui, |ui, row_height| {
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut editor.offset_projection,
                        OffsetProjection::Horizontal,
                        "Horizontal (berm)",
                    );
                    let angle_active =
                        matches!(editor.offset_projection, OffsetProjection::Angled(_));
                    if ui
                        .add(egui::Button::selectable(angle_active, "Angle (batter)"))
                        .clicked()
                        && !angle_active
                    {
                        let deg = editor.offset_angle_input.parse::<f64>().unwrap_or(60.0);
                        editor.offset_projection = OffsetProjection::Angled(deg);
                    }
                    if let OffsetProjection::Angled(ref mut deg) = editor.offset_projection {
                        let resp = ui.add_sized(
                            [48.0, row_height],
                            egui::TextEdit::singleline(&mut editor.offset_angle_input)
                                .hint_text("60"),
                        );
                        if resp.changed()
                            && let Ok(v) = editor.offset_angle_input.parse::<f64>()
                        {
                            *deg = v;
                        }
                        ui.label("°");
                    }
                })
                .response
            });

            ui.add_space(4.0);

            // Measure row (only meaningful for Angled)
            let angle_mode = matches!(editor.offset_projection, OffsetProjection::Angled(_));
            if angle_mode {
                MenuField::new("Measure").show(ui, |ui, _| {
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut editor.offset_measure,
                            OffsetMeasure::Distance,
                            "Distance",
                        );
                        ui.selectable_value(
                            &mut editor.offset_measure,
                            OffsetMeasure::Width,
                            "Width",
                        );
                        let height_active =
                            matches!(editor.offset_measure, OffsetMeasure::Height(_));
                        if ui
                            .add(egui::Button::selectable(height_active, "Height"))
                            .clicked()
                            && !height_active
                        {
                            editor.offset_measure = OffsetMeasure::Height(HeightMode::Relative);
                        }
                    })
                    .response
                });
                if let OffsetMeasure::Height(ref mut mode) = editor.offset_measure {
                    MenuField::new("Height mode").show(ui, |ui, _| {
                        ui.horizontal(|ui| {
                            ui.selectable_value(mode, HeightMode::Relative, "Relative (+/-)");
                            ui.selectable_value(mode, HeightMode::AbsoluteRL, "Absolute RL");
                        })
                        .response
                    });
                }
            } else {
                editor.offset_measure = OffsetMeasure::Distance;
            }

            ui.add_space(4.0);

            // Value label adapts to context
            let value_label = match (&editor.offset_projection, &editor.offset_measure) {
                (OffsetProjection::Horizontal, _) => "Offset distance (m)",
                (OffsetProjection::Angled(_), OffsetMeasure::Distance) => {
                    "Distance along slope (m)"
                }
                (OffsetProjection::Angled(_), OffsetMeasure::Width) => "Horizontal distance (m)",
                (OffsetProjection::Angled(_), OffsetMeasure::Height(HeightMode::Relative)) => {
                    "Height change (m)"
                }
                (OffsetProjection::Angled(_), OffsetMeasure::Height(HeightMode::AbsoluteRL)) => {
                    "Target RL (m)"
                }
            };
            MenuFieldText::new(value_label, &mut editor.offset_value_input)
                .hint_text("0.0")
                .show(ui);

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("✖ Cancel").clicked() {
                    editor.offset_dialog_open = false;
                    editor.offset_target_id = None;
                    editor.active_tool = ActiveTool::None;
                }
                if ui.button("Apply — pick side →").clicked()
                    && let Ok(raw_value) = editor.offset_value_input.parse::<f64>()
                    && raw_value.is_finite()
                {
                    // Compute horiz_dist and z_delta from projection + measure + value.
                    let (horiz_dist, z_delta, project_to_rl) = match editor.offset_projection {
                        OffsetProjection::Horizontal => (raw_value, 0.0, None),
                        OffsetProjection::Angled(deg) => {
                            let rad = deg.to_radians();
                            let tan = rad.tan();
                            match editor.offset_measure {
                                OffsetMeasure::Distance => {
                                    // distance along slope
                                    let h = raw_value * rad.sin();
                                    let horiz = raw_value * rad.cos();
                                    (horiz, h, None)
                                }
                                OffsetMeasure::Width => (raw_value, raw_value * tan, None),
                                OffsetMeasure::Height(HeightMode::Relative) => {
                                    if tan.abs() < 1e-9 {
                                        (raw_value, 0.0, None)
                                    } else {
                                        (raw_value / tan, raw_value, None)
                                    }
                                }
                                OffsetMeasure::Height(HeightMode::AbsoluteRL) => {
                                    // Project each vertex individually along the batter
                                    // angle so the whole string lands flat at the target
                                    // RL, rather than uniformly shifting the string.
                                    (0.0, 0.0, Some((tan, raw_value)))
                                }
                            }
                        }
                    };

                    commands.push(UiCommand::BeginOffsetPick {
                        object_id,
                        horiz_dist,
                        z_delta,
                        project_to_rl,
                    });
                }
            });
        });
}

/// Draw the Bench Batter-Berm Generator dialog.
pub(crate) fn draw_batter_berm_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
    viewport_rect: egui::Rect,
) {
    if editor.batter_berm_target_id.is_none() {
        return;
    }

    ViewportDockPanel::new(
        "batter_berm_panel",
        "Bench Batter-Berm Generator",
        viewport_rect,
    )
    .min_width(310.0)
    .show(ui.ctx(), |ui| {
        MenuFieldF64::new("Berm width (m)", &mut editor.batter_berm_width, 0.1..=500.0)
            .speed(0.1)
            .max_decimals(2)
            .show(ui);
        MenuFieldF64::new(
            "Batter angle (\u{b0})",
            &mut editor.batter_berm_angle,
            1.0..=89.0,
        )
        .speed(0.5)
        .max_decimals(1)
        .show(ui);
        MenuFieldF64::new(
            "Bench height (m)",
            &mut editor.batter_berm_bench_height,
            0.1..=500.0,
        )
        .speed(0.1)
        .max_decimals(2)
        .show(ui);
        let max_benches = editor.batter_berm_max_benches.max(1);
        MenuFieldU32::new("Benches", &mut editor.batter_berm_benches, 1..=max_benches)
            .speed(0.1)
            .show(ui);

        MenuField::new("Type").show(ui, |ui, _| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut editor.batter_berm_mode, BatterBermMode::Pit, "Pit");
                ui.selectable_value(
                    &mut editor.batter_berm_mode,
                    BatterBermMode::Stockpile,
                    "Stockpile",
                );
            })
            .response
        });

        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if ui.button("✖ Cancel").clicked() {
                commands.push(UiCommand::CancelBatterBerm);
            }
            if ui
                .add_enabled(
                    !editor.batter_berm_rings_world.is_empty(),
                    egui::Button::new("✔ Apply"),
                )
                .clicked()
            {
                commands.push(UiCommand::CommitBatterBerm);
            }
        });
    });
}

/// Draw the Relimit Line dialog (intersect, absolute length, or relative length modes).
pub(crate) fn draw_relimit_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
    viewport_rect: egui::Rect,
) {
    ViewportDockPanel::new("relimit_line_panel", "Relimit Line", viewport_rect)
        .min_width(300.0)
        .show(ui.ctx(), |ui| {
            // Mode tabs
            let previous_mode = editor.relimit_mode;
            MenuField::new("Mode").show(ui, |ui, _| {
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut editor.relimit_mode,
                        RelimitMode::Intersect,
                        "Intersect",
                    );
                    ui.selectable_value(
                        &mut editor.relimit_mode,
                        RelimitMode::AbsoluteLength,
                        "Absolute length",
                    );
                    ui.selectable_value(
                        &mut editor.relimit_mode,
                        RelimitMode::RelativeLength,
                        "Relative (+/-)",
                    );
                })
                .response
            });
            if editor.relimit_mode == RelimitMode::Intersect
                && previous_mode != RelimitMode::Intersect
            {
                editor.relimit_waiting_for_pick = true;
                editor.relimit_confirming_end = false;
            }

            ui.add_space(4.0);

            match editor.relimit_mode {
                RelimitMode::Intersect => {
                    if editor.relimit_waiting_for_pick {
                        ui.label("Click the line to intersect with…");
                    } else if editor.relimit_confirming_end {
                        ui.label("Hover to choose which end to move, then click to confirm.");
                        ui.colored_label(
                            egui::Color32::from_rgb(220, 180, 0),
                            match editor.relimit_hover_end {
                                TrimEnd::Start => "Moving: Start endpoint",
                                TrimEnd::End => "Moving: End endpoint",
                            },
                        );
                    }
                }
                RelimitMode::AbsoluteLength | RelimitMode::RelativeLength => {
                    let label = if matches!(editor.relimit_mode, RelimitMode::AbsoluteLength) {
                        "New length (m)"
                    } else {
                        "Delta length (m, use + or -)"
                    };
                    MenuFieldText::new(label, &mut editor.relimit_value_input)
                        .hint_text("0.0")
                        .show(ui);
                    MenuField::new("Move which end").show(ui, |ui, _| {
                        ui.horizontal(|ui| {
                            ui.selectable_value(
                                &mut editor.relimit_resize_end,
                                TrimEnd::Start,
                                "Start",
                            );
                            ui.selectable_value(
                                &mut editor.relimit_resize_end,
                                TrimEnd::End,
                                "End",
                            );
                        })
                        .response
                    });
                }
            }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("✖ Cancel").clicked() {
                    editor.relimit_dialog_open = false;
                    editor.relimit_awaiting_source_pick = false;
                    editor.relimit_source_id = None;
                    editor.relimit_waiting_for_pick = false;
                    editor.relimit_confirming_end = false;
                    editor.relimit_intersection_3d = None;
                    editor.relimit_candidates.clear();
                    editor.relimit_hover_target_id = None;
                    editor.relimit_hover_target_screen_px.clear();
                    editor.relimit_preview_from_px = None;
                    editor.relimit_preview_to_px = None;
                    editor.active_tool = ActiveTool::None;
                }
                match editor.relimit_mode {
                    RelimitMode::Intersect => {
                        if ui.button("Apply — pick target →").clicked() {
                            editor.relimit_dialog_open = false;
                            editor.relimit_waiting_for_pick = true;
                        }
                    }
                    RelimitMode::AbsoluteLength | RelimitMode::RelativeLength => {
                        if ui.button("Apply").clicked()
                            && let (Ok(value), Some(source_id)) = (
                                editor.relimit_value_input.parse::<f64>(),
                                editor.relimit_source_id,
                            )
                            && value.is_finite()
                        {
                            commands.push(UiCommand::RelimitLineResize {
                                source_id,
                                mode: editor.relimit_mode,
                                value,
                            });
                            editor.relimit_dialog_open = false;
                        }
                    }
                }
            });
        });
}

/// Draw the floating Move tool panel with dX/dY/dZ inputs and an Apply button.
pub(crate) fn draw_move_panel(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    commands: &mut Vec<UiCommand>,
    viewport_rect: egui::Rect,
) {
    ViewportDockPanel::new("move_panel", "Move", viewport_rect)
        .min_width(210.0)
        .show(ui.ctx(), |ui| {
            let dx_resp =
                MenuFieldF64::new("dX", &mut editor.move_panel_delta[0], f64::MIN..=f64::MAX)
                    .speed(0.1)
                    .show(ui);
            let dy_resp =
                MenuFieldF64::new("dY", &mut editor.move_panel_delta[1], f64::MIN..=f64::MAX)
                    .speed(0.1)
                    .show(ui);
            let dz_resp =
                MenuFieldF64::new("dZ", &mut editor.move_panel_delta[2], f64::MIN..=f64::MAX)
                    .speed(0.1)
                    .show(ui);
            if dx_resp.changed() || dy_resp.changed() || dz_resp.changed() {
                commands.push(UiCommand::PreviewMoveDelta(glam::DVec3::new(
                    editor.move_panel_delta[0],
                    editor.move_panel_delta[1],
                    editor.move_panel_delta[2],
                )));
            }

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("Apply").clicked() {
                    commands.push(UiCommand::ApplyMoveDelta(glam::DVec3::new(
                        editor.move_panel_delta[0],
                        editor.move_panel_delta[1],
                        editor.move_panel_delta[2],
                    )));
                    editor.active_tool = ActiveTool::None;
                }
                if ui.button("Cancel").clicked() {
                    commands.push(UiCommand::CancelMoveDelta);
                    editor.active_tool = ActiveTool::None;
                }
            });
        });
}

/// Chamfer tool viewport dock: segments input + Apply / Cancel.
pub(crate) fn draw_chamfer_panel(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    commands: &mut Vec<UiCommand>,
    viewport_rect: egui::Rect,
) {
    let corner_picked = editor.chamfer_poly_id.is_some() && editor.chamfer_corner_index.is_some();

    ViewportDockPanel::new("chamfer_panel", "Chamfer", viewport_rect)
        .min_width(200.0)
        .show(ui.ctx(), |ui| {
            if !corner_picked {
                ui.label("Click a corner on a closed polygon.");
            } else {
                let mut seg = editor.chamfer_segments as i32;
                MenuField::new("Segments").show(ui, |ui, _| {
                    ui.add(egui::DragValue::new(&mut seg).range(1..=64).speed(0.1))
                });
                editor.chamfer_segments = seg.clamp(1, 64) as u32;

                let mut r = editor.chamfer_radius;
                let max_r = if editor.chamfer_max_radius.is_finite() {
                    editor.chamfer_max_radius
                } else {
                    f64::MAX
                };
                MenuFieldF64::new("Radius", &mut r, 0.0..=max_r)
                    .speed(0.05)
                    .show(ui);
                editor.chamfer_radius = r.clamp(0.0, max_r);
            }

            ui.add_space(4.0);
            let apply_from_enter = ui.input(|input| input.key_pressed(egui::Key::Enter));
            ui.horizontal(|ui| {
                // Grey out when the displayed value is "0.00" (2 dp) — matches user perception.
                let can_apply = corner_picked && (editor.chamfer_radius * 100.0).round() > 0.0;
                if (apply_from_enter
                    || ui
                        .add_enabled(can_apply, egui::Button::new("Apply"))
                        .clicked())
                    && can_apply
                {
                    commands.push(UiCommand::ApplyChamfer);
                }
                if ui.button("Cancel").clicked() {
                    commands.push(UiCommand::CancelChamfer);
                }
            });
        });
}

/// Bezier curve editor: vertex selection status, control point inputs, and Apply/Cancel.
pub(crate) fn draw_bezier_panel(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    commands: &mut Vec<UiCommand>,
    viewport_rect: egui::Rect,
) {
    let both_selected =
        editor.bezier_selected_verts[0].is_some() && editor.bezier_selected_verts[1].is_some();

    ViewportDockPanel::new("bezier_panel", "Bezier Curve", viewport_rect)
        .min_width(260.0)
        .show(ui.ctx(), |ui| {
            if editor.bezier_poly_id.is_none() {
                ui.label("Click a closed polygon to begin.");
            } else if !both_selected {
                match editor.bezier_selected_verts[0] {
                    None => {
                        ui.label("Click a vertex to start the edge.");
                    }
                    Some(_) => {
                        ui.label("Click an adjacent vertex to finish the edge.");
                    }
                }
            } else {
                // Segments
                let mut seg = editor.bezier_segments as i32;
                MenuField::new("Segments").show(ui, |ui, _| {
                    ui.add(egui::DragValue::new(&mut seg).range(2..=64).speed(0.1))
                });
                editor.bezier_segments = seg.clamp(2, 64) as u32;

                ui.add_space(4.0);

                // Control point 1
                ui.label("Control Point 1:");
                MenuFieldF64::new("  X", &mut editor.bezier_cp1[0], f64::MIN..=f64::MAX)
                    .speed(0.1)
                    .max_decimals(3)
                    .show(ui);
                MenuFieldF64::new("  Y", &mut editor.bezier_cp1[1], f64::MIN..=f64::MAX)
                    .speed(0.1)
                    .max_decimals(3)
                    .show(ui);
                MenuFieldF64::new("  Z", &mut editor.bezier_cp1[2], f64::MIN..=f64::MAX)
                    .speed(0.1)
                    .max_decimals(3)
                    .show(ui);

                ui.add_space(4.0);

                // Control point 2
                ui.label("Control Point 2:");
                MenuFieldF64::new("  X", &mut editor.bezier_cp2[0], f64::MIN..=f64::MAX)
                    .speed(0.1)
                    .max_decimals(3)
                    .show(ui);
                MenuFieldF64::new("  Y", &mut editor.bezier_cp2[1], f64::MIN..=f64::MAX)
                    .speed(0.1)
                    .max_decimals(3)
                    .show(ui);
                MenuFieldF64::new("  Z", &mut editor.bezier_cp2[2], f64::MIN..=f64::MAX)
                    .speed(0.1)
                    .max_decimals(3)
                    .show(ui);
            }

            ui.add_space(4.0);
            let apply_from_enter = ui.input(|input| input.key_pressed(egui::Key::Enter));
            ui.horizontal(|ui| {
                let can_apply = both_selected;
                if (apply_from_enter
                    || ui
                        .add_enabled(can_apply, egui::Button::new("Apply"))
                        .clicked())
                    && can_apply
                {
                    commands.push(UiCommand::ApplyBezier);
                }
                if ui.button("Cancel").clicked() {
                    commands.push(UiCommand::CancelBezier);
                }
            });
        });
}

pub(crate) fn draw_road_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
    viewport_rect: egui::Rect,
) {
    ViewportDockPanel::new("road_panel", "Create Road", viewport_rect)
        .min_width(280.0)
        .show(ui.ctx(), |ui| {
            MenuFieldF64::new("Width (m)", &mut editor.road_width, 0.5..=500.0)
                .speed(0.1)
                .max_decimals(2)
                .show(ui);
            MenuFieldF64::new(
                "Max angle (\u{b0})",
                &mut editor.road_max_angle_degrees,
                0.0..=89.9,
            )
            .speed(0.1)
            .max_decimals(2)
            .show(ui);

            // Camber input with % / ° toggle
            MenuField::new("Camber").show(ui, |ui, _| {
                ui.horizontal(|ui| {
                    if editor.road_camber_is_percent {
                        let mut pct = editor.road_camber_degrees.to_radians().tan() * 100.0;
                        let drag = egui::DragValue::new(&mut pct)
                            .speed(0.1)
                            .range(0.0..=30.0)
                            .max_decimals(2);
                        if ui.add(drag).changed() {
                            editor.road_camber_degrees = pct.atan2(100.0_f64).to_degrees();
                        }
                        ui.label("%");
                    } else {
                        let drag = egui::DragValue::new(&mut editor.road_camber_degrees)
                            .speed(0.1)
                            .range(0.0..=30.0)
                            .max_decimals(2);
                        ui.add(drag);
                        ui.label("\u{b0}");
                    }
                    let toggle_label = if editor.road_camber_is_percent {
                        "Switch to \u{b0}"
                    } else {
                        "Switch to %"
                    };
                    if ui.small_button(toggle_label).clicked() {
                        editor.road_camber_is_percent = !editor.road_camber_is_percent;
                    }
                    ui.response()
                })
                .response
            });

            // Shape selector
            MenuField::new("Shape").show(ui, |ui, _| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut editor.road_shape, RoadShape::Crown, "^ Crown");
                    ui.selectable_value(
                        &mut editor.road_shape,
                        RoadShape::CrossFallRight,
                        "/ Right",
                    );
                    ui.selectable_value(
                        &mut editor.road_shape,
                        RoadShape::CrossFallLeft,
                        "\\ Left",
                    );
                    ui.response()
                })
                .response
            });

            ui.separator();
            let n = editor.pending_stroke.len();
            if n == 0 {
                ui.label("Click on the canvas to place road points.");
            } else {
                ui.label(format!(
                    "{n} point(s) placed — Enter or right-click to finish"
                ));
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("✖ Cancel").clicked() {
                    commands.push(UiCommand::CancelRoad);
                }
                let can_finish = n >= 2;
                if ui
                    .add_enabled(can_finish, egui::Button::new("✔ Finish Road"))
                    .clicked()
                {
                    commands.push(UiCommand::CommitRoad);
                }
            });
        });
}
