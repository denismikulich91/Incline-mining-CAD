//! Top application menu bar (File, Design, View, Analyse, Open Pit, Triangulation).

use crate::ui::{EditorState, UiCommand, UiProjectView};

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
                    if ui.button("New PIDB").clicked() {
                        commands.push(UiCommand::NewPidb);
                        ui.close();
                    }
                    if ui.button("Open PIDBs...").clicked() {
                        commands.push(UiCommand::OpenPidb);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Import").clicked() {
                        editor.show_import = true;
                        editor.show_export = false;
                        ui.close();
                    }
                    if ui.button("Export").clicked() {
                        editor.show_import = false;
                        editor.show_export = true;
                        ui.close();
                    }
                    if ui.button("Add Triangulation Folder").clicked() {
                        commands.push(UiCommand::OpenTriangulationFolder);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Preferences...").clicked() {
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

                    let mut show_world_axis_gizmo = editor.show_world_axis_gizmo;
                    if ui
                        .checkbox(&mut show_world_axis_gizmo, "Show World Axis Gizmo")
                        .changed()
                    {
                        commands.push(UiCommand::SetShowWorldAxisGizmo(show_world_axis_gizmo));
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
                    let mut debug_chunk_coloring = editor.debug_chunk_coloring;
                    if ui
                        .checkbox(&mut debug_chunk_coloring, "Colour Triangles by GPU Chunk")
                        .changed()
                    {
                        commands.push(UiCommand::SetDebugChunkColoring(debug_chunk_coloring));
                    }
                });

                // Not implemented
                ui.menu_button("Object", |ui| {
                    if ui.button("Set Selection Z Value...").clicked() {
                        commands.push(UiCommand::OpenSetSelectionZValueDialog);
                        ui.close();
                    }
                });

                ui.menu_button("Roads", |ui| {
                    if ui.button("Convert to Triangulation...").clicked() {
                        commands.push(UiCommand::OpenConvertRoadsToTriangulation);
                        ui.close();
                    }
                    if ui.button("Convert Road to Polylines").clicked() {
                        commands.push(UiCommand::ConvertSelectedRoadsToPolylines);
                        ui.close();
                    }
                });

                ui.menu_button("Triangulation", |ui| {
                    if ui.button("Create Triangulation").clicked() {
                        commands.push(UiCommand::OpenCreateTriangulation);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Clip by Polygon...").clicked() {
                        commands.push(UiCommand::OpenCutTriangulationByPolygon);
                        ui.close();
                    }
                    if ui.button("Slice by Elevation...").clicked() {
                        commands.push(UiCommand::OpenCutTriangulationByZ);
                        ui.close();
                    }
                    if ui.button("Trim by Surface...").clicked() {
                        commands.push(UiCommand::OpenCutTriangulationBySurface);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Cut Topology with Shell").clicked() {
                        commands.push(UiCommand::OpenCutTopologyByPitShell);
                        ui.close();
                    }
                    if ui.button("Merge Shell into Topology").clicked() {
                        commands.push(UiCommand::OpenIncludeSolidInTopology);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Generate Contours...").clicked() {
                        commands.push(UiCommand::OpenContourTriangulation);
                        ui.close();
                    }
                });

                ui.menu_button("Geology", |ui| {
                    if ui
                        .add_enabled(
                            !project.block_models.is_empty(),
                            egui::Button::new("Create Ore Triangulation..."),
                        )
                        .clicked()
                    {
                        commands.push(UiCommand::OpenCreateOreTriangulation);
                        ui.close();
                    }
                });

                ui.menu_button("Survey", |ui| {
                    if ui
                        .add_enabled(
                            project.point_clouds.iter().any(|cloud| cloud.is_loaded),
                            egui::Button::new("Create Terrain TIN..."),
                        )
                        .clicked()
                    {
                        commands.push(UiCommand::OpenPointCloudTin);
                        ui.close();
                    }
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
