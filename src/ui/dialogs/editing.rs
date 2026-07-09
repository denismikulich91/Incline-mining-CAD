//! Object editing and viewport tool dialogs.

use crate::{
    model::{Document, FillStyle, ObjectColor, ObjectId, RoadShape},
    rendering::color::{color32_to_rgba, rgba_to_color32},
    ui::{
        state::{
            ActiveTool, BatterBermMode, EditorState, HeightMode, MoveToLayerDialog, OffsetMeasure,
            RelimitMode, TrimEnd, UiCommand, UiProjectView,
        },
        themed_icon, unthemed_icon,
        widgets::{
            menu::{
                DragableMenu, MenuField, MenuFieldBool, MenuFieldColor32, MenuFieldCombo,
                MenuFieldF32, MenuFieldF64, MenuFieldRgba, MenuFieldText, MenuFieldU32,
            },
            viewport::ViewportDockPanel,
        },
    },
    userspace_warn,
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
                    let color_resp = MenuFieldColor32::new("Line Colour", &mut color32).show(ui);
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
                    let color_resp = MenuFieldColor32::new("Face Colour", &mut color32).show(ui);
                    if color_resp.drag_stopped() || (color_resp.changed() && !color_resp.dragged())
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
                        ui.selectable_value(&mut is_closed, false, "Open");
                        ui.selectable_value(&mut is_closed, true, "Closed");
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

            if has_doc_objects {
                if ui.button("Move to Layer...").clicked() {
                    let target_layer = project
                        .active_index
                        .and_then(|index| {
                            project.projects.iter().find(|entry| entry.index == index)
                        })
                        .and_then(|entry| entry.layers.first())
                        .map(|layer| layer.id);
                    editor.move_to_layer_dialog = Some(MoveToLayerDialog {
                        object_ids: selected_obj_ids.clone(),
                        target_layer,
                        copy: false,
                    });
                    commands.push(UiCommand::CloseCanvasContextMenu);
                }

                ui.separator();
            }

            if ui.button("Close").clicked() {
                commands.push(UiCommand::CloseCanvasContextMenu);
            }
        });
    if !open {
        commands.push(UiCommand::CloseCanvasContextMenu);
    }
}

pub(crate) fn draw_move_to_layer_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    let Some(active_project_index) = project.active_index else {
        editor.move_to_layer_dialog = None;
        return;
    };
    let Some(active_project) = project
        .projects
        .iter()
        .find(|entry| entry.index == active_project_index)
    else {
        editor.move_to_layer_dialog = None;
        return;
    };
    let Some(dialog) = editor.move_to_layer_dialog.as_mut() else {
        return;
    };

    if dialog
        .target_layer
        .is_none_or(|id| !active_project.layers.iter().any(|layer| layer.id == id))
    {
        dialog.target_layer = active_project.layers.first().map(|layer| layer.id);
    }

    let selected_label = dialog
        .target_layer
        .and_then(|id| active_project.layers.iter().find(|layer| layer.id == id))
        .map(|layer| layer.name.as_str())
        .unwrap_or("Choose a layer");
    let layer_options = active_project
        .layers
        .iter()
        .map(|layer| (Some(layer.id), layer.name.clone().into()));
    let can_apply = dialog.target_layer.is_some() && !dialog.object_ids.is_empty();
    let object_count = dialog.object_ids.len();
    let mut close = false;
    let mut apply = false;
    let mut open = true;

    DragableMenu::new("Move to Layer")
        .open(&mut open)
        .min_width(260.0)
        .show(ui.ctx(), |ui| {
            ui.set_max_width(280.0);
            MenuField::new("Action").show(ui, |ui, _| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut dialog.copy, false, "Move");
                    ui.selectable_value(&mut dialog.copy, true, "Copy");
                })
                .response
            });
            MenuFieldCombo::new(
                "move_to_layer_target",
                "Layer",
                &mut dialog.target_layer,
                selected_label,
                layer_options,
            )
            .width(180.0)
            .show(ui);
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let action_label = if dialog.copy { "Copy" } else { "Move" };
                if ui
                    .add_enabled(can_apply, egui::Button::new(action_label))
                    .clicked()
                {
                    apply = true;
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
            ui.label(format!("{object_count} object(s) selected"));
        });

    if apply {
        if let Some(target_layer) = dialog.target_layer {
            commands.push(UiCommand::MoveObjectsToLayer {
                project_index: active_project_index,
                object_ids: dialog.object_ids.clone(),
                target_layer,
                copy: dialog.copy,
            });
        }
        editor.move_to_layer_dialog = None;
    } else if close || !open {
        editor.move_to_layer_dialog = None;
    }
}

pub(crate) fn draw_set_selection_z_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    commands: &mut Vec<UiCommand>,
) {
    let Some(dialog) = editor.set_selection_z_dialog.as_mut() else {
        return;
    };

    let object_count = dialog.object_ids.len();
    let can_apply = dialog.z_input.is_finite() && object_count > 0;
    let mut close = false;
    let mut apply = false;
    let mut open = true;

    DragableMenu::new("Set Selection Z")
        .open(&mut open)
        .min_width(260.0)
        .show(ui.ctx(), |ui| {
            ui.set_max_width(280.0);
            let response = MenuFieldF64::new("Z value", &mut dialog.z_input, f64::MIN..=f64::MAX)
                .width(120.0)
                .show(ui);
            if !dialog.z_input.is_finite() {
                ui.colored_label(
                    egui::Color32::from_rgb(200, 70, 70),
                    "Enter a valid Z value.",
                );
            }
            ui.add_space(4.0);
            let submitted =
                response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            ui.horizontal(|ui| {
                if (submitted
                    || ui
                        .add_enabled(can_apply, egui::Button::new("Apply"))
                        .clicked())
                    && can_apply
                {
                    apply = true;
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
            ui.label(format!("{object_count} object(s) selected"));
        });

    if apply {
        commands.push(UiCommand::BatchSetZValue(
            dialog.object_ids.clone(),
            dialog.z_input,
        ));
        editor.z_input = dialog.z_input;
        editor.z_level = dialog.z_input;
        editor.set_selection_z_dialog = None;
    } else if close || !open {
        editor.set_selection_z_dialog = None;
    }
}

pub(crate) fn draw_move_layer_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    let Some(dialog) = editor.move_layer_dialog.as_mut() else {
        return;
    };
    let source_name = project
        .projects
        .iter()
        .find(|entry| entry.index == dialog.source_project_index)
        .and_then(|entry| {
            entry
                .layers
                .iter()
                .find(|layer| layer.id == dialog.layer_id)
                .map(|layer| layer.name.as_str())
        })
        .unwrap_or("Layer");

    if dialog.target_project_index.is_none_or(|target| {
        target == dialog.source_project_index
            || !project.projects.iter().any(|entry| entry.index == target)
    }) {
        dialog.target_project_index = project
            .projects
            .iter()
            .find(|entry| entry.index != dialog.source_project_index)
            .map(|entry| entry.index);
    }

    let selected_label = dialog
        .target_project_index
        .and_then(|index| project.projects.iter().find(|entry| entry.index == index))
        .map(|entry| entry.name.as_str())
        .unwrap_or("Choose a PIDB");
    let pidb_options = project
        .projects
        .iter()
        .filter(|entry| entry.index != dialog.source_project_index)
        .map(|entry| (Some(entry.index), entry.name.clone().into()));
    let can_apply = dialog.target_project_index.is_some();
    let mut close = false;
    let mut apply = false;
    let mut open = true;

    DragableMenu::new("Move Layer")
        .open(&mut open)
        .min_width(280.0)
        .show(ui.ctx(), |ui| {
            ui.set_max_width(300.0);
            ui.label(source_name);
            MenuFieldCombo::new(
                "move_layer_target_project",
                "PIDB",
                &mut dialog.target_project_index,
                selected_label,
                pidb_options,
            )
            .width(200.0)
            .show(ui);
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(can_apply, egui::Button::new("Move"))
                    .clicked()
                {
                    apply = true;
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
        });

    if apply {
        if let Some(target_project_index) = dialog.target_project_index {
            commands.push(UiCommand::MoveLayerToProject {
                source_project_index: dialog.source_project_index,
                layer_id: dialog.layer_id,
                target_project_index,
            });
        }
        editor.move_layer_dialog = None;
    } else if close || !open {
        editor.move_layer_dialog = None;
    }
}

/// Draw the startup dialog prompting the user to open or create a .pidb file.
pub(crate) fn draw_select_pidb_dialog(
    ui: &mut egui::Ui,
    editor: &mut crate::ui::state::EditorState,
    commands: &mut Vec<UiCommand>,
) {
    const PANEL_SIZE: f32 = 500.0;
    const COLUMN_WIDTH: f32 = 190.0;
    const ROW_HEIGHT: f32 = 22.0;

    let panel_size = PANEL_SIZE.min(ui.ctx().content_rect().height() * 0.8);

    egui::Area::new(egui::Id::new("select_pidb_dialog"))
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .order(egui::Order::Foreground)
        .show(ui.ctx(), |ui| {
            egui::Frame::new()
                .fill(ui.visuals().window_fill())
                .stroke(ui.visuals().window_stroke())
                .corner_radius(egui::CornerRadius::ZERO)
                .inner_margin(egui::Margin::ZERO)
                .show(ui, |ui| {
                    ui.set_width(panel_size);
                    ui.set_height(panel_size * 0.7);
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 16.0);

                    ui.add(egui::Image::new(unthemed_icon!("splash.svg")).shrink_to_fit());

                    ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
                        ui.add_space(30.0);
                        select_pidb_action_column(ui, "Project", COLUMN_WIDTH, |ui| {
                            if select_pidb_action_row(
                                ui,
                                egui::Image::new(themed_icon!(ui, "new_pidb.svg")),
                                "New PIDB",
                                COLUMN_WIDTH,
                                ROW_HEIGHT,
                            )
                            .clicked()
                            {
                                commands.push(UiCommand::NewPidb);
                            }
                            if select_pidb_action_row(
                                ui,
                                egui::Image::new(themed_icon!(ui, "convert_to_pidb.svg")),
                                "Import",
                                COLUMN_WIDTH,
                                ROW_HEIGHT,
                            )
                            .clicked()
                            {
                                editor.show_export = false;
                                editor.show_import = true;
                                commands.push(UiCommand::CloseStartupDialog);
                            }
                            if select_pidb_action_row(
                                ui,
                                egui::Image::new(themed_icon!(ui, "load_pidb.svg")),
                                "Load PIDB",
                                COLUMN_WIDTH,
                                ROW_HEIGHT,
                            )
                            .clicked()
                            {
                                commands.push(UiCommand::OpenPidb);
                            }
                            if select_pidb_action_row(
                                ui,
                                egui::Image::new(themed_icon!(ui, "exit.svg")),
                                "Close",
                                COLUMN_WIDTH,
                                ROW_HEIGHT,
                            )
                            .clicked()
                            {
                                commands.push(UiCommand::CloseStartupDialog);
                            }
                        });

                        ui.add_space(panel_size - (COLUMN_WIDTH * 2.0) - 60.0);

                        select_pidb_action_column(ui, "Application", COLUMN_WIDTH, |ui| {
                            if select_pidb_action_row(
                                ui,
                                egui::Image::new(themed_icon!(ui, "preferences.svg")),
                                "Preferences",
                                COLUMN_WIDTH,
                                ROW_HEIGHT,
                            )
                            .clicked()
                            {
                                commands.push(UiCommand::OpenPreferences);
                            }
                            if select_pidb_action_row(
                                ui,
                                egui::Image::new(themed_icon!(ui, "github.svg")),
                                "GitHub",
                                COLUMN_WIDTH,
                                ROW_HEIGHT,
                            )
                            .clicked()
                            {
                                ui.ctx().open_url(egui::OpenUrl::new_tab(
                                    "https://github.com/Incline-Developers/Incline",
                                ));
                            }
                            if select_pidb_action_row(
                                ui,
                                egui::Image::new(unthemed_icon!("heart.svg")),
                                "Donate",
                                COLUMN_WIDTH,
                                ROW_HEIGHT,
                            )
                            .clicked()
                            {
                                userspace_warn!("Donate not implemented");
                            }
                        });
                    });
                    egui::Panel::bottom("meta_splash")
                        .show_separator_line(false)
                        .show(ui, |ui| {
                            ui.horizontal_centered(|ui| {
                                ui.label("© 2026 Leo Timmins and Lucas Timmins");
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.label(format!(
                                            "{}: {}",
                                            crate::APP_NAME,
                                            crate::APP_RELEASE
                                        ))
                                    },
                                );
                            });
                        });
                });
        });
}

fn select_pidb_action_column(
    ui: &mut egui::Ui,
    heading: &'static str,
    width: f32,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    ui.vertical(|ui| {
        ui.set_width(width);
        ui.label(
            egui::RichText::new(heading)
                .size(12.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);
        add_contents(ui);
    });
}

fn select_pidb_action_row(
    ui: &mut egui::Ui,
    icon: egui::Image<'static>,
    label: &'static str,
    width: f32,
    height: f32,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click());
    if response.hovered() {
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(2),
            ui.visuals().widgets.hovered.bg_fill,
        );
    }

    let icon_size = egui::vec2(22.0, 22.0);
    let icon_rect = egui::Rect::from_min_size(
        egui::pos2(rect.left() + 2.0, rect.center().y - icon_size.y / 2.0),
        icon_size,
    );
    icon.fit_to_exact_size(icon_size).paint_at(ui, icon_rect);

    ui.painter().text(
        egui::pos2(rect.left() + 30.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(13.0),
        ui.visuals().text_color(),
    );

    response
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
            let name_response = MenuFieldText::new("Layer name", &mut editor.new_layer_name)
                .hint_text("Required")
                .show(ui);
            let submitted =
                name_response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
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
                    editor.new_layer_dialog_open = false;
                }
                if ui.button("Cancel").clicked() {
                    editor.new_layer_project_index = None;
                    editor.new_layer_dialog_open = false;
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
    ViewportDockPanel::new("text_edit_panel", "Edit Text", viewport_rect)
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
                    .suffix("m")
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
            *geometry_dirty |= MenuFieldRgba::new("Colour", &mut editor.pending_text_color)
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
    viewport_rect: egui::Rect,
) {
    ViewportDockPanel::new("finish_poly_dialog", "Finish Polygon", viewport_rect).show(ui, |ui| {
        MenuField::new("Shape").show(ui, |ui, _| {
            ui.horizontal(|ui| {
                if ui.button("Closed").clicked() {
                    commands.push(UiCommand::FinishPolyClose);
                    editor.poly_finish_dialog = false;
                }
                if ui.button("Open").clicked() {
                    commands.push(UiCommand::CommitStrokeOpen);
                    editor.poly_finish_dialog = false;
                }
            });
        });
    });
}

// I dont like this - we need to clean this up later - too many options and sub menus
/// Draw the Offset Element dialog.
pub(crate) fn draw_offset_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
    viewport_rect: egui::Rect,
) {
    if editor.offset_target_ids.is_empty() {
        return;
    }

    ViewportDockPanel::new("offset_element_panel", "Offset Element", viewport_rect)
        .min_width(350.0)
        .show(ui.ctx(), |ui| {
            let response =
                MenuFieldF64::new("Angle", &mut editor.offset_angle_degrees, -90.0..=90.0)
                    .width(70.0)
                    .speed(0.5)
                    .suffix("°")
                    .show(ui);
            if response.changed() {
                editor.offset_angle_degrees = editor.offset_angle_degrees.clamp(-90.0, 90.0);
            }

            ui.add_space(4.0);

            MenuField::new("Measure").show(ui, |ui, _| {
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut editor.offset_measure,
                        OffsetMeasure::Distance,
                        "Distance",
                    );
                    ui.selectable_value(&mut editor.offset_measure, OffsetMeasure::Width, "Width");
                    let height_active = matches!(editor.offset_measure, OffsetMeasure::Height(_));
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

            ui.add_space(4.0);

            // Value label adapts to context
            let value_label = match editor.offset_measure {
                OffsetMeasure::Distance => "Distance along slope",
                OffsetMeasure::Width => "Horizontal distance",
                OffsetMeasure::Height(HeightMode::Relative) => "Height change",
                OffsetMeasure::Height(HeightMode::AbsoluteRL) => "Target RL",
            };
            MenuFieldF64::new(value_label, &mut editor.offset_value_input, 0.0..=f64::MAX)
                .speed(0.1)
                .suffix("m")
                .show(ui);

            MenuFieldBool::new(
                "Collide with Triangulation",
                &mut editor.offset_collide_with_triangulation,
            )
            .show(ui);

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    commands.push(UiCommand::CancelOffset);
                }
                let can_pick_side = matches!(
                    editor.offset_measure,
                    OffsetMeasure::Height(HeightMode::AbsoluteRL)
                ) || editor.offset_value_input.abs() > 1e-9;
                let pick_side_clicked = ui
                    .add_enabled(can_pick_side, egui::Button::new("Pick Side"))
                    .clicked();
                let enter_pressed = ui.input(|input| input.key_pressed(egui::Key::Enter));
                if can_pick_side && (pick_side_clicked || enter_pressed) {
                    queue_begin_offset_pick(commands, editor);
                }
            });
        });
}

fn queue_begin_offset_pick(commands: &mut Vec<UiCommand>, editor: &EditorState) {
    let rad = editor.offset_angle_degrees.to_radians();
    let tan = rad.tan();
    let (horiz_dist, z_delta, project_to_rl) = match editor.offset_measure {
        OffsetMeasure::Distance => {
            let h = editor.offset_value_input * rad.sin();
            let horiz = editor.offset_value_input * rad.cos();
            (horiz, h, None)
        }
        OffsetMeasure::Width => (
            editor.offset_value_input,
            editor.offset_value_input * tan,
            None,
        ),
        OffsetMeasure::Height(HeightMode::Relative) => {
            if tan.abs() < 1e-9 {
                (editor.offset_value_input, 0.0, None)
            } else {
                (
                    editor.offset_value_input / tan,
                    editor.offset_value_input,
                    None,
                )
            }
        }
        OffsetMeasure::Height(HeightMode::AbsoluteRL) => {
            // Project each vertex individually along the batter angle so the
            // whole string lands flat at the target RL.
            (0.0, 0.0, Some((tan, editor.offset_value_input)))
        }
    };

    commands.push(UiCommand::BeginOffsetPick {
        object_ids: editor.offset_target_ids.clone(),
        horiz_dist,
        z_delta,
        project_to_rl,
        collide_with_triangulation: editor.offset_collide_with_triangulation,
    });
}

/// Draw the Generate Batter-Berms dialog.
pub(crate) fn draw_batter_berm_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
    viewport_rect: egui::Rect,
) {
    if editor.batter_berm_target_id.is_none() {
        return;
    }

    ViewportDockPanel::new("batter_berm_panel", "Generate Batter-Berms", viewport_rect)
        .min_width(310.0)
        .show(ui.ctx(), |ui| {
            MenuFieldF64::new("Berm width", &mut editor.batter_berm_width, 0.1..=500.0)
                .speed(0.1)
                .max_decimals(2)
                .suffix("m")
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
                "Bench height",
                &mut editor.batter_berm_bench_height,
                0.1..=500.0,
            )
            .speed(0.1)
            .max_decimals(2)
            .suffix("m")
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
                if ui.button("Cancel").clicked() {
                    commands.push(UiCommand::CancelBatterBerm);
                }
                if ui
                    .add_enabled(
                        !editor.batter_berm_rings_world.is_empty(),
                        egui::Button::new("Apply"),
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
                    MenuFieldF64::new(label, &mut editor.relimit_value_input, f64::MIN..=f64::MAX)
                        .speed(0.1)
                        .suffix("m")
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
                if ui.button("Cancel").clicked() {
                    commands.push(UiCommand::CancelRelimit);
                }
                match editor.relimit_mode {
                    RelimitMode::Intersect => {
                        if ui.button("Apply and Pick Target").clicked() {
                            editor.relimit_dialog_open = false;
                            editor.relimit_waiting_for_pick = true;
                        }
                    }
                    RelimitMode::AbsoluteLength | RelimitMode::RelativeLength => {
                        if ui.button("Apply").clicked()
                            && editor.relimit_value_input.is_finite()
                            && let Some(source_id) = editor.relimit_source_id
                        {
                            commands.push(UiCommand::RelimitLineResize {
                                source_id,
                                mode: editor.relimit_mode,
                                value: editor.relimit_value_input,
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
                MenuFieldU32::new("Segments", &mut editor.chamfer_segments, 1..=64)
                    .speed(0.1)
                    .show(ui);

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
            ui.horizontal(|ui| {
                // Grey out when the displayed value is "0.00" (2 dp) — matches user perception.
                let can_apply = corner_picked && (editor.chamfer_radius * 100.0).round() > 0.0;
                if ui
                    .add_enabled(can_apply, egui::Button::new("Apply"))
                    .clicked()
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
                MenuFieldU32::new("Segments", &mut editor.bezier_segments, 2..=64)
                    .speed(0.1)
                    .show(ui);

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
            ui.horizontal(|ui| {
                let can_apply = both_selected;
                if ui
                    .add_enabled(can_apply, egui::Button::new("Apply"))
                    .clicked()
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
            MenuFieldF64::new("Width", &mut editor.road_width, 10.0..=100.0)
                .speed(0.1)
                .max_decimals(2)
                .suffix("m")
                .show(ui);

            MenuFieldF64::new(
                "Max turning angle",
                &mut editor.road_max_angle_degrees,
                0.0..=89.9,
            )
            .speed(0.1)
            .max_decimals(2)
            .suffix("\u{b0}")
            .show(ui);

            // Camber input (DEG)
            MenuFieldF64::new("Camber", &mut editor.road_camber_degrees, 0.0..=30.0)
                .max_decimals(2)
                .speed(0.05)
                .suffix("\u{b0}")
                .show(ui);

            // Shape selector
            MenuField::new("Shape").show(ui, |ui, _| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut editor.road_shape, RoadShape::Crown, "Crown");
                    ui.selectable_value(
                        &mut editor.road_shape,
                        RoadShape::CrossFallRight,
                        "Slant Right",
                    );
                    ui.selectable_value(
                        &mut editor.road_shape,
                        RoadShape::CrossFallLeft,
                        "Slant Left",
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
                if ui.button("Cancel").clicked() {
                    commands.push(UiCommand::CancelRoad);
                }
                let can_finish = n >= 2;
                if ui
                    .add_enabled(can_finish, egui::Button::new("Finish Road"))
                    .clicked()
                {
                    commands.push(UiCommand::CommitRoad);
                }
            });
        });
}
