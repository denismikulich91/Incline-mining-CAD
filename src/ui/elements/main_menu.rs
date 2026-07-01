//! Top application menu bar (File, Design, View, Analyse, Open Pit, Triangulation).

use crate::ui::{EditorState, UiCommand, UiProjectView, state::FileOperationKind};

/// Draw the top menu bar panel.
///
/// Returns the panel's bounding rect.
pub(crate) fn draw_main_menu(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) -> egui::Rect {
    egui::Panel::top("main_menu")
        .show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    let dirty_count = project.projects.iter().filter(|entry| entry.dirty).count();
                    if ui
                        .add_enabled(dirty_count > 0, egui::Button::new("Save All"))
                        .clicked()
                    {
                        commands.push(UiCommand::SaveAllPidbs);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("New .pidb").clicked() {
                        commands.push(UiCommand::NewPidb);
                        ui.close();
                    }
                    if ui.button("Open .pidb(s)").clicked() {
                        commands.push(UiCommand::OpenPidb);
                        ui.close();
                    }
                    ui.separator();
                    draw_import_menu(ui, commands);
                    draw_export_menu(ui, commands);
                    if ui.button("Add Triangulation Folder").clicked() {
                        commands.push(UiCommand::OpenTriangulationFolder);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Preferences ...").clicked() {
                        commands.push(UiCommand::OpenPreferences);
                        ui.close();
                    }
                    if ui.button(format!("Exit {}", crate::APP_NAME)).clicked() {
                        commands.push(UiCommand::RequestExit);
                        ui.close();
                    }
                });

                ui.menu_button("View", |ui| {
                    let mut dark_mode = editor.dark_mode;
                    if ui.checkbox(&mut dark_mode, "Dark Mode").changed() {
                        commands.push(UiCommand::SetDarkMode(dark_mode));
                    }
                    ui.separator();

                    let mut show_console = editor.show_console;
                    if ui.checkbox(&mut show_console, "Show Console").changed() {
                        commands.push(UiCommand::SetShowConsole(show_console));
                    }

                    let mut wireframes_enabled = editor.topology_wireframes_enabled;
                    if ui
                        .checkbox(&mut wireframes_enabled, "Enable Wireframes")
                        .changed()
                    {
                        commands.push(UiCommand::SetTopologyWireframes(wireframes_enabled));
                    }
                    ui.separator();
                    let mut show_world_axis_gizmo = editor.show_world_axis_gizmo;
                    if ui
                        .checkbox(&mut show_world_axis_gizmo, "Show World Axis Gizmo")
                        .changed()
                    {
                        commands.push(UiCommand::SetShowWorldAxisGizmo(show_world_axis_gizmo));
                    }
                    let mut show_view_cube = editor.show_view_cube;
                    if ui.checkbox(&mut show_view_cube, "Show View Cube").changed() {
                        commands.push(UiCommand::SetShowViewCube(show_view_cube));
                    }
                });

                ui.menu_button("Triangulation", |ui| {
                    if ui.button("Create Triangulation").clicked() {
                        commands.push(UiCommand::OpenCreateTriangulation);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Cut by Polygon …").clicked() {
                        commands.push(UiCommand::OpenCutTriangulationByPolygon);
                        ui.close();
                    }
                    if ui.button("Cut by Z Range …").clicked() {
                        commands.push(UiCommand::OpenCutTriangulationByZ);
                        ui.close();
                    }
                    if ui.button("Cut by Surface …").clicked() {
                        commands.push(UiCommand::OpenCutTriangulationBySurface);
                        ui.close();
                    }
                    if ui.button("Cut Topology to Pit Shell …").clicked() {
                        commands.push(UiCommand::OpenCutTopologyByPitShell);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Include Pit/Stockpile Solid …").clicked() {
                        commands.push(UiCommand::OpenIncludeSolidInTopology);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Generate Contour Lines …").clicked() {
                        commands.push(UiCommand::OpenContourTriangulation);
                        ui.close();
                    }
                });

                ui.menu_button("Geology", |ui| {
                    if ui.button("Import Block Model ...").clicked() {
                        commands.push(UiCommand::OpenImportBlockModel);
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            !project.block_models.is_empty(),
                            egui::Button::new("Create Ore Triangulation ..."),
                        )
                        .clicked()
                    {
                        commands.push(UiCommand::OpenCreateOreTriangulation);
                        ui.close();
                    }
                });

                // Not implemented
                ui.add_enabled_ui(false, |ui| {
                    ui.menu_button("Roads", |_ui| {});
                });

                // Not implemented
                ui.add_enabled_ui(false, |ui| {
                    ui.menu_button("Survey", |_ui| {});
                });

                // Not implemented
                ui.add_enabled_ui(false, |ui| {
                    ui.menu_button("Geotech", |_ui| {});
                });

                // Not implemented
                ui.add_enabled_ui(false, |ui| {
                    ui.menu_button("Drill and Blast", |_ui| {});
                });

                // Not implemented
                ui.add_enabled_ui(false, |ui| {
                    ui.menu_button("Open Pit", |_ui| {});
                });

                // Not implemented
                ui.add_enabled_ui(false, |ui| {
                    ui.menu_button("Underground", |_ui| {});
                });

                // Not implemented
                ui.add_enabled_ui(false, |ui| {
                    ui.menu_button("Help", |_ui| {});
                });
            });
        })
        .response
        .rect
}

/// Draw the Import sub-menu items.
fn draw_import_menu(ui: &mut egui::Ui, commands: &mut Vec<UiCommand>) {
    ui.menu_button("Import", |ui| {
        for (label, kind) in [
            ("As Layers ...", FileOperationKind::ImportLayers),
            ("As .pidb ...", FileOperationKind::ImportPidb),
            (
                "As Triangulation ...",
                FileOperationKind::ImportTriangulation,
            ),
            ("As Block Model ...", FileOperationKind::ImportBlockModel),
        ] {
            if ui.button(label).clicked() {
                commands.push(UiCommand::OpenFileOperation(kind));
                ui.close();
            }
        }
    });
}

/// Draw the Export sub-menu items.
fn draw_export_menu(ui: &mut egui::Ui, commands: &mut Vec<UiCommand>) {
    ui.menu_button("Export", |ui| {
        for (label, kind) in [
            ("Layer to ...", FileOperationKind::ExportLayer),
            (".pidb to ...", FileOperationKind::ExportPidb),
            (
                "Triangulation to ...",
                FileOperationKind::ExportTriangulation,
            ),
        ] {
            if ui.button(label).clicked() {
                commands.push(UiCommand::OpenFileOperation(kind));
                ui.close();
            }
        }
    });
}
