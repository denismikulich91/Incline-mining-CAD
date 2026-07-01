//! Triangulation creation and processing dialogs.

use crate::{
    model::{Document, Object, ObjectId, SceneEntityId, triangulation::TriangulationId},
    rendering::color::{color32_to_rgba, rgba_to_color32},
    ui::{
        state::{
            EditorState, TriCreatePhase, TriSurfaceCutSide, TriSurfaceType, UiCommand,
            UiProjectView,
        },
        widgets::menu::{DragableMenu, MenuField, MenuFieldColor32, MenuFieldCombo, MenuFieldText},
    },
};

/// Reset the Create Triangulation workflow state.
fn tri_reset_state(editor: &mut EditorState) {
    editor.tri_create_open = false;
    editor.tri_create_phase = TriCreatePhase::MainDialog;
    editor.tri_create_picker_px = None;
    editor.tri_hover_handles.clear();
    editor.tri_selected_object_ids.clear();
    editor.tri_selected_layer_ids.clear();
    editor.tri_name_input.clear();
    editor.tri_surface_type = TriSurfaceType::Surface;
    editor.selected_handles.clear();
}

/// Returns a human-readable label for an object (type + layer name).
fn object_label(obj: &Object, document: &Document) -> String {
    let layer_name = document
        .layer(obj.layer())
        .map(|l| l.name.as_str())
        .unwrap_or("?");
    match obj {
        Object::Point { .. } => format!("Point on '{layer_name}'"),
        Object::Polyline { closed: true, .. } => format!("Polygon on '{layer_name}'"),
        Object::Polyline { closed: false, .. } => format!("String on '{layer_name}'"),
        Object::Text { content, .. } => format!("Text \"{content}\" on '{layer_name}'"),
        Object::Road { .. } => format!("Road on '{layer_name}'"),
    }
}

fn tri_surface_type_label(surface_type: TriSurfaceType) -> &'static str {
    match surface_type {
        TriSurfaceType::Surface => "Open surface",
        TriSurfaceType::SolidClosed => "Solid – fully closed",
    }
}

/// Movable main dialog for the Create Triangulation workflow.
pub(crate) fn draw_tri_create_main_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    document: &Document,
    commands: &mut Vec<UiCommand>,
) {
    if !editor.tri_create_open || editor.tri_create_phase != TriCreatePhase::MainDialog {
        return;
    }

    let mut open = true;
    DragableMenu::new("Create Triangulation")
        .open(&mut open)
        .min_width(300.0)
        .show(ui.ctx(), |ui| {
            ui.colored_label(
                egui::Color32::GRAY,
                "Click objects in the viewport to select/deselect. Drag to box-select.",
            );
            ui.add_space(4.0);

            // --- Selection list ---
            let mut remove_object: Option<ObjectId> = None;
            let mut hover_handles: std::collections::HashSet<SceneEntityId> =
                std::collections::HashSet::new();

            egui::ScrollArea::vertical()
                .max_height(160.0)
                .id_salt("tri_sel_list")
                .show(ui, |ui| {
                    for &oid in &editor.tri_selected_object_ids {
                        let label = document
                            .get_object(oid)
                            .map(|o| object_label(o, document))
                            .unwrap_or_else(|| format!("Object #{}", oid.0));
                        ui.horizontal(|ui| {
                            let resp = ui.selectable_label(false, &label);
                            if resp.hovered() {
                                hover_handles.insert(SceneEntityId::Object(oid));
                            }
                            if ui
                                .button(egui::RichText::new("✖").color(egui::Color32::RED))
                                .clicked()
                            {
                                remove_object = Some(oid);
                            }
                        });
                    }
                });

            editor.tri_hover_handles = hover_handles;

            if let Some(oid) = remove_object {
                editor.tri_selected_object_ids.retain(|&o| o != oid);
                editor.selected_handles.remove(&SceneEntityId::Object(oid));
            }

            let has_selection = !editor.tri_selected_object_ids.is_empty();
            if !has_selection {
                ui.colored_label(egui::Color32::GRAY, "No objects selected yet.");
            }

            ui.add_space(4.0);
            ui.separator();

            // --- Surface / solid type ---
            let surface_type_label = tri_surface_type_label(editor.tri_surface_type);
            MenuFieldCombo::new(
                "tri_surface_type",
                "Triangulation type",
                &mut editor.tri_surface_type,
                surface_type_label,
                [TriSurfaceType::Surface, TriSurfaceType::SolidClosed].map(|surface_type| {
                    (surface_type, tri_surface_type_label(surface_type).into())
                }),
            )
            .width(210.0)
            .show(ui);

            ui.separator();

            // --- Name + Triangulate ---
            MenuFieldText::new("Name", &mut editor.tri_name_input)
                .width(180.0)
                .hint_text("triangulation name")
                .show(ui);
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let ready = has_selection && !editor.tri_name_input.trim().is_empty();
                if ui
                    .add_enabled(ready, egui::Button::new("Triangulate"))
                    .clicked()
                {
                    let object_ids: Vec<ObjectId> = editor.tri_selected_object_ids.clone();
                    let name = editor.tri_name_input.trim().to_owned();
                    let surface_type = editor.tri_surface_type;
                    tri_reset_state(editor);
                    commands.push(UiCommand::ExecuteCreateTriangulation {
                        name,
                        object_ids,
                        surface_type,
                    });
                }
                if ui.button("Cancel").clicked() {
                    tri_reset_state(editor);
                }
            });
        });

    if !open {
        tri_reset_state(editor);
    }
}

pub(crate) fn draw_cut_poly_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    _document: &Document,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    if !editor.tri_cut_poly_open {
        return;
    }

    // While awaiting a viewport pick, show a small floating prompt instead of the full dialog.
    if editor.tri_cut_poly_awaiting_pick {
        let mut open = true;
        DragableMenu::new("Pick Polygon")
            .open(&mut open)
            .min_width(240.0)
            .show(ui.ctx(), |ui| {
                ui.label("Click a closed polygon in the viewport.");
                ui.add_space(4.0);
                if ui.button("Cancel Pick").clicked() {
                    editor.tri_cut_poly_awaiting_pick = false;
                }
            });
        if !open {
            editor.tri_cut_poly_awaiting_pick = false;
            editor.tri_cut_poly_open = false;
        }
        return;
    }

    let mut open = true;
    DragableMenu::new("Cut Triangulation by Polygon")
        .open(&mut open)
        .min_width(300.0)
        .show(ui.ctx(), |ui| {
            let loaded: Vec<(TriangulationId, &str)> = project
                .triangulations
                .iter()
                .filter_map(|t| t.id.map(|id| (id, t.name.as_str())))
                .collect();
            let tri_label = editor
                .tri_cut_poly_tri_id
                .and_then(|id| loaded.iter().find(|(lid, _)| *lid == id).map(|(_, n)| *n))
                .unwrap_or("— select —");
            let old_tri_id = editor.tri_cut_poly_tri_id;
            MenuFieldCombo::new(
                "cut_poly_tri",
                "Triangulation",
                &mut editor.tri_cut_poly_tri_id,
                tri_label,
                loaded.iter().map(|(id, name)| (Some(*id), (*name).into())),
            )
            .width(220.0)
            .show(ui);
            if editor.tri_cut_poly_tri_id != old_tri_id
                && editor.tri_cut_poly_name_input.trim().is_empty()
                && let Some(name) = editor.tri_cut_poly_tri_id.and_then(|id| {
                    loaded
                        .iter()
                        .find(|(loaded_id, _)| *loaded_id == id)
                        .map(|(_, name)| *name)
                })
            {
                let path = std::path::Path::new(name);
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(name);
                let ext = path.extension().and_then(|e| e.to_str());
                editor.tri_cut_poly_name_input = if let Some(ext) = ext {
                    format!("{stem}_cut.{ext}")
                } else {
                    format!("{stem}_cut")
                };
            }

            ui.add_space(4.0);

            // Polygon picker — viewport click, not a list
            MenuField::new("Cutting polygon").show(ui, |ui, _| {
                let poly_label = if editor.tri_cut_poly_object_id.is_some() {
                    editor.tri_cut_poly_object_name.as_str()
                } else {
                    "— none picked —"
                };
                ui.horizontal(|ui| {
                    ui.add(egui::Label::new(poly_label).truncate());
                    if ui.button("Pick …").clicked() {
                        commands.push(UiCommand::BeginCutPolyPick);
                    }
                })
                .response
            });

            ui.add_space(4.0);

            // Output name
            MenuFieldText::new("Output name", &mut editor.tri_cut_poly_name_input)
                .width(220.0)
                .hint_text("e.g. mysurf_cut")
                .show(ui);

            ui.add_space(6.0);
            ui.separator();

            let can_run = editor.tri_cut_poly_tri_id.is_some()
                && editor.tri_cut_poly_object_id.is_some()
                && !editor.tri_cut_poly_name_input.trim().is_empty();

            ui.horizontal(|ui| {
                if ui.add_enabled(can_run, egui::Button::new("Cut")).clicked()
                    && let (Some(tri_id), Some(poly_id)) =
                        (editor.tri_cut_poly_tri_id, editor.tri_cut_poly_object_id)
                {
                    commands.push(UiCommand::ExecuteCutTriangulationByPolygon {
                        tri_id,
                        polygon_id: poly_id,
                        name: editor.tri_cut_poly_name_input.trim().to_owned(),
                    });
                }
                if ui.button("Cancel").clicked() {
                    editor.tri_cut_poly_open = false;
                    editor.tool_highlight_id = None;
                }
            });
        });
    if !open {
        editor.tri_cut_poly_open = false;
        editor.tool_highlight_id = None;
    }
}

pub(crate) fn draw_cut_z_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    if !editor.tri_cut_z_open {
        return;
    }
    let mut open = true;
    DragableMenu::new("Cut Triangulation by Z Range")
        .open(&mut open)
        .min_width(280.0)
        .show(ui.ctx(), |ui| {
            let loaded: Vec<(TriangulationId, &str)> = project
                .triangulations
                .iter()
                .filter_map(|t| t.id.map(|id| (id, t.name.as_str())))
                .collect();
            let tri_label = editor
                .tri_cut_z_tri_id
                .and_then(|id| loaded.iter().find(|(lid, _)| *lid == id).map(|(_, n)| *n))
                .unwrap_or("— select —");
            let old_tri_id = editor.tri_cut_z_tri_id;
            MenuFieldCombo::new(
                "cut_z_tri",
                "Triangulation",
                &mut editor.tri_cut_z_tri_id,
                tri_label,
                loaded.iter().map(|(id, name)| (Some(*id), (*name).into())),
            )
            .width(220.0)
            .show(ui);
            if editor.tri_cut_z_tri_id != old_tri_id
                && editor.tri_cut_z_name_input.trim().is_empty()
                && let Some(name) = editor.tri_cut_z_tri_id.and_then(|id| {
                    loaded
                        .iter()
                        .find(|(loaded_id, _)| *loaded_id == id)
                        .map(|(_, name)| *name)
                })
            {
                let path = std::path::Path::new(name);
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(name);
                let ext = path.extension().and_then(|e| e.to_str());
                editor.tri_cut_z_name_input = if let Some(ext) = ext {
                    format!("{stem}_slice.{ext}")
                } else {
                    format!("{stem}_slice")
                };
            }

            ui.add_space(4.0);

            MenuFieldText::new("Z min", &mut editor.tri_cut_z_min_input)
                .width(80.0)
                .show(ui);
            MenuFieldText::new("Z max", &mut editor.tri_cut_z_max_input)
                .width(80.0)
                .show(ui);

            ui.add_space(4.0);

            MenuFieldText::new("Output name", &mut editor.tri_cut_z_name_input)
                .width(220.0)
                .hint_text("e.g. mysurf_slice")
                .show(ui);

            ui.add_space(6.0);
            ui.separator();

            let z_min = editor.tri_cut_z_min_input.trim().parse::<f64>().ok();
            let z_max = editor.tri_cut_z_max_input.trim().parse::<f64>().ok();
            let valid_z_range = z_min.zip(z_max).is_some_and(|(min, max)| min < max);
            let can_run = editor.tri_cut_z_tri_id.is_some()
                && valid_z_range
                && !editor.tri_cut_z_name_input.trim().is_empty();

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(can_run, egui::Button::new("Slice"))
                    .clicked()
                    && let Some(tri_id) = editor.tri_cut_z_tri_id
                {
                    commands.push(UiCommand::ExecuteCutTriangulationByZ {
                        tri_id,
                        z_min: z_min.expect("Slice button enabled only with parsed Z min"),
                        z_max: z_max.expect("Slice button enabled only with parsed Z max"),
                        name: editor.tri_cut_z_name_input.trim().to_owned(),
                    });
                }
                if ui.button("Cancel").clicked() {
                    editor.tri_cut_z_open = false;
                }
            });
        });
    if !open {
        editor.tri_cut_z_open = false;
    }
}

pub(crate) fn draw_cut_surface_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    if !editor.tri_cut_surface_open {
        return;
    }

    let mut open = true;
    DragableMenu::new("Cut Triangulation by Surface")
        .open(&mut open)
        .min_width(320.0)
        .show(ui.ctx(), |ui| {
            let loaded: Vec<(TriangulationId, &str)> = project
                .triangulations
                .iter()
                .filter_map(|entry| entry.id.map(|id| (id, entry.name.as_str())))
                .collect();

            let target_label = editor
                .tri_cut_surface_target_id
                .and_then(|id| {
                    loaded
                        .iter()
                        .find(|(loaded_id, _)| *loaded_id == id)
                        .map(|(_, name)| *name)
                })
                .unwrap_or("— select —");
            let old_target = editor.tri_cut_surface_target_id;
            MenuFieldCombo::new(
                "cut_surface_target",
                "Cut object",
                &mut editor.tri_cut_surface_target_id,
                target_label,
                loaded.iter().map(|(id, name)| (Some(*id), (*name).into())),
            )
            .width(220.0)
            .show(ui);

            if editor.tri_cut_surface_target_id != old_target {
                if editor.tri_cut_surface_reference_id == editor.tri_cut_surface_target_id {
                    editor.tri_cut_surface_reference_id = None;
                }
                if let Some(name) = editor.tri_cut_surface_target_id.and_then(|id| {
                    loaded
                        .iter()
                        .find(|(loaded_id, _)| *loaded_id == id)
                        .map(|(_, name)| *name)
                }) {
                    let path = std::path::Path::new(name);
                    let stem = path
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or(name);
                    let ext = path.extension().and_then(|value| value.to_str());
                    editor.tri_cut_surface_name_input = if let Some(ext) = ext {
                        format!("{stem}_surface_cut.{ext}")
                    } else {
                        format!("{stem}_surface_cut")
                    };
                }
            }

            let reference_label = editor
                .tri_cut_surface_reference_id
                .and_then(|id| {
                    loaded
                        .iter()
                        .find(|(loaded_id, _)| *loaded_id == id)
                        .map(|(_, name)| *name)
                })
                .unwrap_or("— select —");
            let target_id = editor.tri_cut_surface_target_id;
            MenuFieldCombo::new(
                "cut_surface_reference",
                "Reference topology",
                &mut editor.tri_cut_surface_reference_id,
                reference_label,
                loaded
                    .iter()
                    .filter(|(id, _)| Some(*id) != target_id)
                    .map(|(id, name)| (Some(*id), (*name).into())),
            )
            .width(220.0)
            .show(ui);

            MenuField::new("Operation").show(ui, |ui, _| {
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut editor.tri_cut_surface_side,
                        TriSurfaceCutSide::CutTop,
                        "Cut top",
                    );
                    ui.selectable_value(
                        &mut editor.tri_cut_surface_side,
                        TriSurfaceCutSide::CutBottom,
                        "Cut bottom",
                    );
                })
                .response
            });

            ui.colored_label(
                egui::Color32::GRAY,
                match editor.tri_cut_surface_side {
                    TriSurfaceCutSide::CutTop => {
                        "Keeps the cut object at or below the reference topology."
                    }
                    TriSurfaceCutSide::CutBottom => {
                        "Keeps the cut object at or above the reference topology."
                    }
                },
            );

            MenuFieldText::new("Output name", &mut editor.tri_cut_surface_name_input)
                .width(220.0)
                .hint_text("e.g. design_surface_cut")
                .show(ui);

            ui.add_space(6.0);
            ui.separator();

            let can_run = editor.tri_cut_surface_target_id.is_some()
                && editor.tri_cut_surface_reference_id.is_some()
                && editor.tri_cut_surface_target_id != editor.tri_cut_surface_reference_id
                && !editor.tri_cut_surface_name_input.trim().is_empty();
            ui.horizontal(|ui| {
                if ui.add_enabled(can_run, egui::Button::new("Cut")).clicked()
                    && let (Some(target_id), Some(reference_id)) = (
                        editor.tri_cut_surface_target_id,
                        editor.tri_cut_surface_reference_id,
                    )
                {
                    commands.push(UiCommand::ExecuteCutTriangulationBySurface {
                        target_id,
                        reference_id,
                        side: editor.tri_cut_surface_side,
                        name: editor.tri_cut_surface_name_input.trim().to_owned(),
                    });
                }
                if ui.button("Cancel").clicked() {
                    editor.tri_cut_surface_open = false;
                }
            });
        });

    if !open {
        editor.tri_cut_surface_open = false;
    }
}

pub(crate) fn draw_cut_topology_to_pit_shell_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    if !editor.tri_cut_pitshell_open {
        return;
    }

    let mut open = true;
    DragableMenu::new("Cut Topology to Pit Shell")
        .open(&mut open)
        .min_width(320.0)
        .show(ui.ctx(), |ui| {
            let loaded: Vec<(TriangulationId, &str)> = project
                .triangulations
                .iter()
                .filter_map(|entry| entry.id.map(|id| (id, entry.name.as_str())))
                .collect();

            let topology_label = editor
                .tri_cut_pitshell_topology_id
                .and_then(|id| loaded.iter().find(|(lid, _)| *lid == id).map(|(_, n)| *n))
                .unwrap_or("— select —");
            let old_topology_id = editor.tri_cut_pitshell_topology_id;
            MenuFieldCombo::new(
                "cut_pitshell_topology",
                "Topology",
                &mut editor.tri_cut_pitshell_topology_id,
                topology_label,
                loaded
                    .iter()
                    .filter(|(id, _)| Some(*id) != editor.tri_cut_pitshell_pitshell_id)
                    .map(|(id, name)| (Some(*id), (*name).into())),
            )
            .width(220.0)
            .show(ui);

            if editor.tri_cut_pitshell_topology_id != old_topology_id
                && editor.tri_cut_pitshell_name_input.trim().is_empty()
                && let Some(name) = editor
                    .tri_cut_pitshell_topology_id
                    .and_then(|id| loaded.iter().find(|(lid, _)| *lid == id).map(|(_, n)| *n))
            {
                let path = std::path::Path::new(name);
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(name);
                let ext = path.extension().and_then(|e| e.to_str());
                editor.tri_cut_pitshell_name_input = if let Some(ext) = ext {
                    format!("{stem}_cut.{ext}")
                } else {
                    format!("{stem}_cut")
                };
            }

            let pitshell_label = editor
                .tri_cut_pitshell_pitshell_id
                .and_then(|id| loaded.iter().find(|(lid, _)| *lid == id).map(|(_, n)| *n))
                .unwrap_or("— select —");
            MenuFieldCombo::new(
                "cut_pitshell_shell",
                "Pit shell",
                &mut editor.tri_cut_pitshell_pitshell_id,
                pitshell_label,
                loaded
                    .iter()
                    .filter(|(id, _)| Some(*id) != editor.tri_cut_pitshell_topology_id)
                    .map(|(id, name)| (Some(*id), (*name).into())),
            )
            .width(220.0)
            .show(ui);

            ui.add_space(4.0);
            ui.colored_label(
                egui::Color32::GRAY,
                "Removes the topology where the pit shell excavates below it so the shell \
                 fills the hole. The seam follows the true 3D contact line between the \
                 surfaces; topology under parts of the shell that stand above the ground \
                 is kept.",
            );

            MenuFieldText::new("Output name", &mut editor.tri_cut_pitshell_name_input)
                .width(220.0)
                .hint_text("e.g. topo_cut")
                .show(ui);

            ui.add_space(6.0);
            ui.separator();

            let can_run = editor.tri_cut_pitshell_topology_id.is_some()
                && editor.tri_cut_pitshell_pitshell_id.is_some()
                && !editor.tri_cut_pitshell_name_input.trim().is_empty();

            ui.horizontal(|ui| {
                if ui.add_enabled(can_run, egui::Button::new("Cut")).clicked()
                    && let (Some(topology_id), Some(pit_shell_id)) = (
                        editor.tri_cut_pitshell_topology_id,
                        editor.tri_cut_pitshell_pitshell_id,
                    )
                {
                    commands.push(UiCommand::ExecuteCutTopologyByPitShell {
                        topology_id,
                        pit_shell_id,
                        name: editor.tri_cut_pitshell_name_input.trim().to_owned(),
                    });
                }
                if ui.button("Cancel").clicked() {
                    editor.tri_cut_pitshell_open = false;
                    editor.tool_highlight_id = None;
                }
            });
        });

    if !open {
        editor.tri_cut_pitshell_open = false;
        editor.tool_highlight_id = None;
    }
}

pub(crate) fn draw_include_solid_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    if !editor.tri_include_solid_open {
        return;
    }

    let mut open = true;
    DragableMenu::new("Include Pit/Stockpile Solid")
        .open(&mut open)
        .min_width(320.0)
        .show(ui.ctx(), |ui| {
            let loaded: Vec<(TriangulationId, &str)> = project
                .triangulations
                .iter()
                .filter_map(|entry| entry.id.map(|id| (id, entry.name.as_str())))
                .collect();

            let topology_label = editor
                .tri_include_solid_topology_id
                .and_then(|id| {
                    loaded
                        .iter()
                        .find(|(loaded_id, _)| *loaded_id == id)
                        .map(|(_, name)| *name)
                })
                .unwrap_or("— select —");
            let old_topology = editor.tri_include_solid_topology_id;
            MenuFieldCombo::new(
                "include_solid_topology",
                "Topology",
                &mut editor.tri_include_solid_topology_id,
                topology_label,
                loaded.iter().map(|(id, name)| (Some(*id), (*name).into())),
            )
            .width(220.0)
            .show(ui);

            if editor.tri_include_solid_topology_id != old_topology {
                if editor.tri_include_solid_shape_id == editor.tri_include_solid_topology_id {
                    editor.tri_include_solid_shape_id = None;
                }
                if let Some(name) = editor.tri_include_solid_topology_id.and_then(|id| {
                    loaded
                        .iter()
                        .find(|(loaded_id, _)| *loaded_id == id)
                        .map(|(_, name)| *name)
                }) {
                    let stem = std::path::Path::new(name)
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or(name);
                    editor.tri_include_solid_name_input = format!("{stem}_with_shape");
                }
            }

            let shape_label = editor
                .tri_include_solid_shape_id
                .and_then(|id| {
                    loaded
                        .iter()
                        .find(|(loaded_id, _)| *loaded_id == id)
                        .map(|(_, name)| *name)
                })
                .unwrap_or("— select —");
            let topology_id = editor.tri_include_solid_topology_id;
            MenuFieldCombo::new(
                "include_solid_shape",
                "Pit/stockpile solid",
                &mut editor.tri_include_solid_shape_id,
                shape_label,
                loaded
                    .iter()
                    .filter(|(id, _)| Some(*id) != topology_id)
                    .map(|(id, name)| (Some(*id), (*name).into())),
            )
            .width(220.0)
            .show(ui);

            MenuFieldText::new("Output name", &mut editor.tri_include_solid_name_input)
                .width(220.0)
                .hint_text("e.g. topo_with_pit")
                .show(ui);

            ui.add_space(6.0);
            ui.separator();

            let can_run = editor.tri_include_solid_topology_id.is_some()
                && editor.tri_include_solid_shape_id.is_some()
                && editor.tri_include_solid_topology_id != editor.tri_include_solid_shape_id
                && !editor.tri_include_solid_name_input.trim().is_empty();
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(can_run, egui::Button::new("Create"))
                    .clicked()
                    && let (Some(topology_id), Some(shape_id)) = (
                        editor.tri_include_solid_topology_id,
                        editor.tri_include_solid_shape_id,
                    )
                {
                    commands.push(UiCommand::ExecuteIncludeSolidInTopology {
                        topology_id,
                        shape_id,
                        name: editor.tri_include_solid_name_input.trim().to_owned(),
                    });
                }
                if ui.button("Cancel").clicked() {
                    editor.tri_include_solid_open = false;
                }
            });
        });

    if !open {
        editor.tri_include_solid_open = false;
    }
}

pub(crate) fn draw_contour_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    if !editor.tri_contour_open {
        return;
    }
    let mut open = true;
    DragableMenu::new("Generate Contour Lines")
        .open(&mut open)
        .min_width(300.0)
        .show(ui.ctx(), |ui| {
            let loaded: Vec<(TriangulationId, &str)> = project
                .triangulations
                .iter()
                .filter_map(|t| t.id.map(|id| (id, t.name.as_str())))
                .collect();
            let tri_label = editor
                .tri_contour_tri_id
                .and_then(|id| loaded.iter().find(|(lid, _)| *lid == id).map(|(_, n)| *n))
                .unwrap_or("— select —");
            MenuFieldCombo::new(
                "contour_tri",
                "Triangulation",
                &mut editor.tri_contour_tri_id,
                tri_label,
                loaded.iter().map(|(id, name)| (Some(*id), (*name).into())),
            )
            .width(220.0)
            .show(ui);

            ui.add_space(4.0);

            MenuFieldText::new(
                "Minor interval",
                &mut editor.tri_contour_minor_interval_input,
            )
            .width(70.0)
            .show(ui);
            MenuFieldText::new(
                "Major interval",
                &mut editor.tri_contour_major_interval_input,
            )
            .width(70.0)
            .show(ui);

            ui.add_space(4.0);

            let mut minor_color = rgba_to_color32(editor.tri_contour_minor_color);
            if MenuFieldColor32::new("Minor color", &mut minor_color)
                .show(ui)
                .changed()
            {
                editor.tri_contour_minor_color = color32_to_rgba(minor_color);
            }
            let mut major_color = rgba_to_color32(editor.tri_contour_major_color);
            if MenuFieldColor32::new("Major color", &mut major_color)
                .show(ui)
                .changed()
            {
                editor.tri_contour_major_color = color32_to_rgba(major_color);
            }

            ui.add_space(4.0);

            let pidb_label = project
                .projects
                .get(editor.tri_contour_project_index)
                .map(|p| p.name.as_str())
                .unwrap_or("— select —");
            MenuFieldCombo::new(
                "contour_pidb",
                "Store in",
                &mut editor.tri_contour_project_index,
                pidb_label,
                project
                    .projects
                    .iter()
                    .map(|project| (project.index, project.name.clone().into())),
            )
            .width(220.0)
            .show(ui);

            ui.add_space(6.0);
            ui.separator();

            let minor_interval = editor
                .tri_contour_minor_interval_input
                .trim()
                .parse::<f64>()
                .ok();
            let major_interval = editor
                .tri_contour_major_interval_input
                .trim()
                .parse::<f64>()
                .ok();
            let valid_intervals = minor_interval
                .zip(major_interval)
                .is_some_and(|(minor, major)| minor >= 1e-6 && major >= 1e-6 && major >= minor);
            let can_run = editor.tri_contour_tri_id.is_some()
                && valid_intervals
                && !project.projects.is_empty();

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(can_run, egui::Button::new("Generate"))
                    .clicked()
                    && let Some(tri_id) = editor.tri_contour_tri_id
                {
                    commands.push(UiCommand::ExecuteContourTriangulation {
                        tri_id,
                        major_interval: major_interval
                            .expect("Generate button enabled only with parsed major interval"),
                        minor_interval: minor_interval
                            .expect("Generate button enabled only with parsed minor interval"),
                        major_color: editor.tri_contour_major_color,
                        minor_color: editor.tri_contour_minor_color,
                        project_index: editor.tri_contour_project_index,
                    });
                }
                if ui.button("Cancel").clicked() {
                    editor.tri_contour_open = false;
                }
            });
        });
    if !open {
        editor.tri_contour_open = false;
    }
}
