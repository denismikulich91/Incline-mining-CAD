//! File, project, layer, viewport, and properties dialogs.

use crate::ui::{
    state::{EditorState, FileOperationKind, UiCommand, UiProjectView},
    widgets::{
        menu::{DragableMenu, MenuField, MenuFieldCombo, MenuFieldF64},
        viewport::ViewportDockPanel,
    },
};

/// Draw the file import/export dialog.
pub(crate) fn draw_file_operation_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    let Some(dialog) = editor.file_operation_dialog.as_mut() else {
        return;
    };
    let title = match dialog.kind {
        FileOperationKind::ImportLayers => "Import as Layers",
        FileOperationKind::ImportPidb => "Import as .pidb",
        FileOperationKind::ImportTriangulation => "Import as Triangulation",
        FileOperationKind::ImportBlockModel => "Import Block Model",
        FileOperationKind::ExportLayer => "Export Layer",
        FileOperationKind::ExportPidb => "Export .pidb",
        FileOperationKind::ExportTriangulation => "Export Triangulation",
    };

    let mut open = true;
    DragableMenu::new(title)
        .open(&mut open)
        .min_width(390.0)
        .show(ui.ctx(), |ui| {
            match dialog.kind {
                FileOperationKind::ImportLayers => {
                    project_combo(ui, "Destination .pidb", project, &mut dialog.project_index);
                }
                FileOperationKind::ImportPidb => {
                    MenuField::new("Source type")
                        .show(ui, |ui, _| ui.label("Vulcan design database (dgd.isis)"));
                    MenuField::new("Source file").show(ui, |ui, _| {
                        let path = dialog
                            .source_path
                            .as_ref()
                            .and_then(|path| path.file_name())
                            .and_then(|name| name.to_str())
                            .unwrap_or("No file chosen");
                        ui.horizontal(|ui| {
                            ui.label(path);
                            if ui.button("Choose File...").clicked() {
                                commands.push(UiCommand::ChooseDgdIsisSource);
                            }
                        })
                        .response
                    });
                }
                FileOperationKind::ImportTriangulation => {}
                FileOperationKind::ImportBlockModel => {
                    MenuField::new("Model file").show(ui, |ui, _| {
                        let path = dialog
                            .bmf_path
                            .as_ref()
                            .and_then(|path| path.file_name())
                            .and_then(|name| name.to_str())
                            .unwrap_or("No .bmf chosen");
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new(path).truncate());
                            if ui.button("Choose .bmf...").clicked() {
                                commands.push(UiCommand::ChooseBlockModelBmf);
                            }
                        })
                        .response
                    });
                    MenuField::new("Definition file").show(ui, |ui, _| {
                        let path = dialog
                            .bdf_path
                            .as_ref()
                            .and_then(|path| path.file_name())
                            .and_then(|name| name.to_str())
                            .unwrap_or("No .bdf chosen");
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new(path).truncate());
                            if ui.button("Choose .bdf...").clicked() {
                                commands.push(UiCommand::ChooseBlockModelBdf);
                            }
                        })
                        .response
                    });
                }
                FileOperationKind::ExportLayer => {
                    layer_combo(ui, "Layer", project, &mut dialog.layer);
                }
                FileOperationKind::ExportPidb => {
                    project_combo(ui, ".pidb", project, &mut dialog.project_index);
                }
                FileOperationKind::ExportTriangulation => {
                    triangulation_combo(ui, "Triangulation", project, &mut dialog.triangulation);
                }
            }

            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    commands.push(UiCommand::CloseFileOperation);
                }
                let (action_label, enabled) = match dialog.kind {
                    FileOperationKind::ImportLayers => {
                        ("Import...", dialog.project_index.is_some())
                    }
                    FileOperationKind::ImportPidb => ("Import...", dialog.source_path.is_some()),
                    FileOperationKind::ImportTriangulation => ("Import...", true),
                    FileOperationKind::ImportBlockModel => ("Import", dialog.bmf_path.is_some()),
                    FileOperationKind::ExportLayer => ("Export...", dialog.layer.is_some()),
                    FileOperationKind::ExportPidb => ("Export...", dialog.project_index.is_some()),
                    FileOperationKind::ExportTriangulation => {
                        ("Export...", dialog.triangulation.is_some())
                    }
                };
                if ui
                    .add_enabled(enabled, egui::Button::new(action_label))
                    .clicked()
                {
                    match dialog.kind {
                        FileOperationKind::ImportLayers => {
                            commands.push(UiCommand::ImportDxfInto(dialog.project_index.unwrap()));
                        }
                        FileOperationKind::ImportPidb => {
                            commands.push(UiCommand::ConfirmImportDgdIsis);
                        }
                        FileOperationKind::ImportTriangulation => {
                            commands.push(UiCommand::ImportTriangulation);
                        }
                        FileOperationKind::ImportBlockModel => {
                            commands.push(UiCommand::ConfirmImportBlockModel {
                                bmf_path: dialog.bmf_path.clone().unwrap(),
                                bdf_path: dialog.bdf_path.clone(),
                            });
                        }
                        FileOperationKind::ExportLayer => {
                            let (project_index, layer) = dialog.layer.unwrap();
                            commands.push(UiCommand::ExportLayerDxf(project_index, layer));
                        }
                        FileOperationKind::ExportPidb => {
                            commands.push(UiCommand::ExportPidbDxf(dialog.project_index.unwrap()));
                        }
                        FileOperationKind::ExportTriangulation => {
                            commands.push(UiCommand::ExportTriangulation(
                                dialog.triangulation.unwrap(),
                            ));
                        }
                    }
                    commands.push(UiCommand::CloseFileOperation);
                }
            });
        });
    if !open {
        commands.push(UiCommand::CloseFileOperation);
    }
}

fn project_combo(
    ui: &mut egui::Ui,
    field_label: &str,
    project: &UiProjectView,
    selected: &mut Option<usize>,
) {
    let label = selected
        .and_then(|index| project.projects.iter().find(|entry| entry.index == index))
        .map(|entry| entry.name.as_str())
        .unwrap_or("Choose a .pidb");
    MenuFieldCombo::new(
        ("file_operation_project", field_label),
        field_label,
        selected,
        label,
        project
            .projects
            .iter()
            .map(|entry| (Some(entry.index), entry.name.clone().into())),
    )
    .width(220.0)
    .show(ui);
}

fn layer_combo(
    ui: &mut egui::Ui,
    field_label: &str,
    project: &UiProjectView,
    selected: &mut Option<(usize, crate::model::LayerId)>,
) {
    let selected_label = selected.and_then(|selected| {
        project.projects.iter().find_map(|entry| {
            entry
                .layers
                .iter()
                .find(|layer| layer.is_loaded && (entry.index, layer.id) == selected)
                .map(|layer| format!("{} / {}", entry.name, layer.name))
        })
    });
    let options = project.projects.iter().flat_map(|entry| {
        entry
            .layers
            .iter()
            .filter(|layer| layer.is_loaded)
            .map(|layer| {
                (
                    Some((entry.index, layer.id)),
                    format!("{} / {}", entry.name, layer.name).into(),
                )
            })
    });
    MenuFieldCombo::new(
        "file_operation_layer",
        field_label,
        selected,
        selected_label.unwrap_or_else(|| "Choose a loaded layer".to_string()),
        options,
    )
    .width(220.0)
    .show(ui);
}

fn triangulation_combo(
    ui: &mut egui::Ui,
    field_label: &str,
    project: &UiProjectView,
    selected: &mut Option<crate::model::triangulation::TriangulationId>,
) {
    let label = selected
        .and_then(|id| {
            project
                .triangulations
                .iter()
                .find(|entry| entry.id == Some(id))
        })
        .map(|entry| entry.name.as_str())
        .unwrap_or("Choose a loaded triangulation");
    MenuFieldCombo::new(
        "file_operation_triangulation",
        field_label,
        selected,
        label,
        project
            .triangulations
            .iter()
            .filter(|entry| entry.is_loaded)
            .map(|entry| (entry.id, entry.name.clone().into())),
    )
    .width(220.0)
    .show(ui);
}

/// Draw the Vertical Exaggeration adjustment dialog.
pub(crate) fn draw_vertical_exaggeration_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    viewport_rect: egui::Rect,
) {
    if !editor.vertical_exaggeration_dialog_open {
        return;
    }
    ViewportDockPanel::new(
        "vertical_exaggeration_panel",
        "Vertical Exaggeration",
        viewport_rect,
    )
    .min_width(330.0)
    .show(ui.ctx(), |ui| {
        ui.label("Scales Z distances visually without changing stored coordinates.");
        ui.add_space(8.0);
        let response = MenuFieldF64::new(
            "Z scale ratio",
            &mut editor.vertical_exaggeration_input,
            0.1..=20.,
        )
        .max_decimals(1)
        .width(100.)
        .speed(0.1)
        .suffix("x")
        .show(ui);
        ui.add_space(10.0);
        let apply_from_enter =
            response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
        let cancel_from_escape = ui.input(|input| input.key_pressed(egui::Key::Escape));
        ui.horizontal(|ui| {
            if ui.button("Cancel").clicked() || cancel_from_escape {
                editor.vertical_exaggeration_dialog_open = false;
            }
            if ui.button("Reset to 1×").clicked() {
                editor.vertical_exaggeration = 1.0;
                editor.vertical_exaggeration_input = 1.0;
                editor.vertical_exaggeration_dialog_open = false;
            }
            let apply_clicked = ui.add(egui::Button::new("Apply")).clicked();
            if apply_from_enter || apply_clicked {
                editor.vertical_exaggeration = editor.vertical_exaggeration_input;
                editor.vertical_exaggeration_dialog_open = false;
            }
        });
    });
}
