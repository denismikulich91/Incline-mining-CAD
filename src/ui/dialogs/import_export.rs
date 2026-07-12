//! Import & Export menus.

use crate::{
    model::{LayerId, formats::MeshFormat, triangulation::TriangulationId},
    ui::{
        state::{DataMenu, EditorState, UiCommand, UiProjectView},
        widgets::menu::{DragableMenu, MenuFieldBool, MenuFieldCombo, MenuFieldFilePicker},
    },
};

const MENU_HEIGHT: f32 = 300.0;
const EXPLORER_WIDTH: f32 = 250.0;
const FIELD_WIDTH: f32 = 280.0;
const DETAILS_WIDTH: f32 = 380.0;
const MENU_WIDTH: f32 = EXPLORER_WIDTH + 4.0 + DETAILS_WIDTH;

fn draw_entry(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    title: &str,
    data_menu: DataMenu,
) -> egui::Response {
    let response = ui.add(
        egui::Button::new(title)
            .min_size(egui::Vec2::new(EXPLORER_WIDTH - 30., 25.0))
            .selected(editor.data_menu == data_menu),
    );
    if response.clicked() {
        editor.data_menu = data_menu;
    }
    response
}

pub(crate) fn draw_import_menu(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    if !editor.show_import {
        return;
    }
    if !is_import_menu(editor.data_menu) {
        editor.data_menu = DataMenu::Dxf;
    }

    let mut show_import = editor.show_import;
    let mut close_after_action = false;
    DragableMenu::new("Import")
        .open(&mut show_import)
        .show(ui.ctx(), |ui| {
            ui.set_height(MENU_HEIGHT);
            ui.set_width(MENU_WIDTH);

            ui.horizontal(|ui| {
                draw_import_explorer(ui, editor);
                ui.add_space(4.0);
                ui.vertical(|ui| {
                    ui.allocate_ui(egui::Vec2::new(DETAILS_WIDTH, MENU_HEIGHT - 25.0), |ui| {
                        ui.set_width(DETAILS_WIDTH);
                        ui.set_height(MENU_HEIGHT - 25.);
                        draw_import_details(ui, editor, project, commands);
                    });
                    ui.allocate_ui_with_layout(
                        egui::Vec2::new(DETAILS_WIDTH, ui.spacing().interact_size.y),
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            let command = import_command(editor);
                            if ui
                                .add_enabled(command.is_some(), egui::Button::new("Import"))
                                .clicked()
                                && let Some(command) = command
                            {
                                commands.push(command);
                                close_after_action = true;
                            }
                            if ui.button("Default").clicked() {
                                reset_import_defaults(editor, project);
                            }
                        },
                    );
                });
            });
        });
    if close_after_action {
        show_import = false;
    }
    editor.show_import = show_import;
}

pub(crate) fn draw_export_menu(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    if !editor.show_export {
        return;
    }
    if !is_export_menu(editor.data_menu) {
        editor.data_menu = DataMenu::Dxf;
    }

    let mut show_export = editor.show_export;
    let mut close_after_action = false;
    DragableMenu::new("Export")
        .open(&mut show_export)
        .show(ui.ctx(), |ui| {
            ui.set_height(MENU_HEIGHT);
            ui.set_width(MENU_WIDTH);

            ui.horizontal(|ui| {
                draw_export_explorer(ui, editor);
                ui.add_space(4.0);
                ui.vertical(|ui| {
                    ui.allocate_ui(egui::Vec2::new(DETAILS_WIDTH, MENU_HEIGHT - 25.0), |ui| {
                        ui.set_width(DETAILS_WIDTH);
                        ui.set_height(MENU_HEIGHT - 25.);
                        draw_export_details(ui, editor, project);
                    });
                    ui.allocate_ui_with_layout(
                        egui::Vec2::new(DETAILS_WIDTH, ui.spacing().interact_size.y),
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            let command = export_command(editor);
                            if ui
                                .add_enabled(command.is_some(), egui::Button::new("Export"))
                                .clicked()
                                && let Some(command) = command
                            {
                                commands.push(command);
                                close_after_action = true;
                            }
                            if ui.button("Default").clicked() {
                                reset_export_defaults(editor, project);
                            }
                        },
                    );
                });
            });
        });
    if close_after_action {
        show_export = false;
    }
    editor.show_export = show_export;
}

fn draw_import_explorer(ui: &mut egui::Ui, editor: &mut EditorState) {
    ui.set_height(MENU_HEIGHT);
    ui.set_width(EXPLORER_WIDTH);
    egui::ScrollArea::new([false, true])
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.vertical(|ui| {
                egui::CollapsingHeader::new("CAD")
                    .default_open(true)
                    .show(ui, |ui| {
                        draw_entry(ui, editor, "Drawing Exchange Format (.dxf)", DataMenu::Dxf);
                        draw_entry(ui, editor, "ProInspector database (.pidb)", DataMenu::Pidb);
                        draw_entry(
                            ui,
                            editor,
                            "Vulcan Design Database (.dgd.isis)",
                            DataMenu::DgdIsis,
                        );
                        draw_entry(ui, editor, "Deswik Unified File (.duf)", DataMenu::Duf);
                    });
                egui::CollapsingHeader::new("Triangulations")
                    .default_open(true)
                    .show(ui, |ui| {
                        draw_entry(ui, editor, "Vulcan Triangulation (.00t)", DataMenu::Tri00t);
                        draw_entry(ui, editor, "Wavefront OBJ (.obj)", DataMenu::Obj);
                        draw_entry(ui, editor, "STL (.stl)", DataMenu::Stl);
                        draw_entry(ui, editor, "PLY (.ply)", DataMenu::Ply);
                    });
                egui::CollapsingHeader::new("Point Clouds")
                    .default_open(true)
                    .show(ui, |ui| {
                        draw_entry(ui, editor, "LAS / LAZ (.las, .laz)", DataMenu::Las);
                        draw_entry(ui, editor, "ASCII Points (.xyz, .pts)", DataMenu::Xyz);
                        draw_entry(ui, editor, "Point Cloud Data (.pcd)", DataMenu::Pcd);
                    });
                egui::CollapsingHeader::new("Block Models")
                    .default_open(true)
                    .show(ui, |ui| {
                        draw_entry(ui, editor, "Vulcan Block Model File (.bmf)", DataMenu::Bmf);
                    });
                egui::CollapsingHeader::new("Textures")
                    .default_open(true)
                    .show(ui, |ui| {
                        draw_entry(ui, editor, "GeoTIFF (.tif, .tiff)", DataMenu::Geotiff);
                    });
            });
        });
}

fn draw_export_explorer(ui: &mut egui::Ui, editor: &mut EditorState) {
    ui.set_height(MENU_HEIGHT);
    ui.set_width(EXPLORER_WIDTH);
    egui::ScrollArea::new([false, true])
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.vertical(|ui| {
                egui::CollapsingHeader::new("CAD")
                    .default_open(true)
                    .show(ui, |ui| {
                        draw_entry(ui, editor, "Drawing Exchange Format (.dxf)", DataMenu::Dxf);
                        draw_entry(ui, editor, "ProInspector database (.pidb)", DataMenu::Pidb);
                    });
                egui::CollapsingHeader::new("Triangulations")
                    .default_open(true)
                    .show(ui, |ui| {
                        draw_entry(ui, editor, "Vulcan Triangulation (.00t)", DataMenu::Tri00t);
                        draw_entry(ui, editor, "Wavefront OBJ (.obj)", DataMenu::Obj);
                        draw_entry(ui, editor, "STL (.stl)", DataMenu::Stl);
                        draw_entry(ui, editor, "PLY (.ply)", DataMenu::Ply);
                    });
            });
        });
}

fn draw_import_details(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    match editor.data_menu {
        DataMenu::Dxf => draw_import_dxf(ui, editor, project, commands),
        DataMenu::Pidb => {
            ui.heading("Open PIDB");
            draw_import_source_picker(ui, editor, commands, "PIDB file", "No .pidb chosen");
        }
        DataMenu::DgdIsis => {
            ui.heading("Import Vulcan Design Database");
            draw_import_source_picker(ui, editor, commands, "Source file", "No .dgd.isis chosen");
        }
        DataMenu::Duf => {
            ui.heading("Import DUF");
            draw_import_source_picker(ui, editor, commands, "Source file", "No .duf chosen");
        }
        DataMenu::Tri00t => draw_import_mesh(ui, editor, commands, "Import Vulcan Triangulation"),
        DataMenu::Obj => draw_import_mesh(ui, editor, commands, "Import Wavefront OBJ"),
        DataMenu::Stl => draw_import_mesh(ui, editor, commands, "Import STL"),
        DataMenu::Ply => draw_import_mesh(ui, editor, commands, "Import PLY"),
        DataMenu::Las => draw_import_mesh(ui, editor, commands, "Import LAS/LAZ Point Cloud"),
        DataMenu::Xyz => draw_import_mesh(ui, editor, commands, "Import ASCII Point Cloud"),
        DataMenu::Pcd => draw_import_mesh(ui, editor, commands, "Import PCD Point Cloud"),
        DataMenu::Bmf => draw_import_bmf(ui, editor, commands),
        DataMenu::Geotiff => draw_import_mesh(ui, editor, commands, "Import GeoTIFF"),
        DataMenu::None => {}
    }
}

fn draw_export_details(ui: &mut egui::Ui, editor: &mut EditorState, project: &UiProjectView) {
    match editor.data_menu {
        DataMenu::Dxf => draw_export_dxf(ui, editor, project),
        DataMenu::Pidb => draw_export_pidb(ui, editor, project),
        DataMenu::Tri00t => draw_export_mesh(ui, editor, project, "Export Vulcan Triangulation"),
        DataMenu::Obj => draw_export_mesh(ui, editor, project, "Export Wavefront OBJ"),
        DataMenu::Stl => draw_export_mesh(ui, editor, project, "Export STL"),
        DataMenu::Ply => draw_export_mesh(ui, editor, project, "Export PLY"),
        _ => {}
    }
}

fn draw_import_dxf(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) {
    ui.heading("Import DXF");
    draw_import_source_picker(ui, editor, commands, "Source file", "No .dxf chosen");
    MenuFieldBool::new("Import as New PIDB", &mut editor.import_dxf_as_pidb).show(ui);
    if !editor.import_dxf_as_pidb {
        active_project_label(ui, "Add to PIDB:", project);
    }
}

fn draw_import_mesh(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    commands: &mut Vec<UiCommand>,
    heading: &str,
) {
    ui.heading(heading);
    draw_import_source_picker(ui, editor, commands, "Source file", "No file chosen");
}

fn draw_import_source_picker(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    commands: &mut Vec<UiCommand>,
    label: &'static str,
    empty_text: &'static str,
) {
    if MenuFieldFilePicker::new(label, selected_import_source_paths(editor))
        .empty_text(empty_text)
        .button_text("Choose...")
        .width(FIELD_WIDTH)
        .show(ui)
        .changed()
    {
        commands.push(UiCommand::ChooseImportSourceFiles(editor.data_menu));
    }
}

fn draw_import_bmf(ui: &mut egui::Ui, editor: &mut EditorState, commands: &mut Vec<UiCommand>) {
    ui.heading("Import Block Model");
    if MenuFieldFilePicker::new("Model file", optional_path_slice(&editor.import_bmf_path))
        .empty_text("No .bmf chosen")
        .button_text("Choose...")
        .width(FIELD_WIDTH)
        .show(ui)
        .changed()
    {
        commands.push(UiCommand::ChooseBlockModelBmf);
    }
    if MenuFieldFilePicker::new(
        "Definition file",
        optional_path_slice(&editor.import_bdf_path),
    )
    .empty_text("No .bdf chosen")
    .button_text("Choose...")
    .width(FIELD_WIDTH)
    .show(ui)
    .changed()
    {
        commands.push(UiCommand::ChooseBlockModelBdf);
    }
}

fn draw_export_dxf(ui: &mut egui::Ui, editor: &mut EditorState, project: &UiProjectView) {
    ui.heading("Export DXF");
    MenuFieldBool::new("Export one layer", &mut editor.export_dxf_layer).show(ui);
    if editor.export_dxf_layer {
        ensure_export_layer(editor, project);
        layer_combo(
            ui,
            "dxf_export_layer",
            "Layer:",
            project,
            &mut editor.export_layer,
        );
    } else {
        active_project_label(ui, "PIDB:", project);
    }
}

fn draw_export_pidb(ui: &mut egui::Ui, _editor: &mut EditorState, project: &UiProjectView) {
    ui.heading("Export PIDB");
    active_project_label(ui, "PIDB:", project);
}

fn draw_export_mesh(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    heading: &str,
) {
    ui.heading(heading);
    ensure_export_triangulation(editor, project);
    triangulation_combo(
        ui,
        "mesh_export_triangulation",
        "Triangulation:",
        project,
        &mut editor.export_triangulation,
    );
}

/// Imports/exports always target the active PIDB; show which one that is.
fn active_project_label(ui: &mut egui::Ui, field_label: &str, project: &UiProjectView) {
    let name = project
        .projects
        .iter()
        .find(|entry| entry.is_active)
        .map(|entry| entry.name.as_str())
        .unwrap_or("No active PIDB");
    ui.horizontal(|ui| {
        ui.label(field_label);
        ui.label(name);
    });
}

fn layer_combo(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    field_label: &str,
    project: &UiProjectView,
    selected: &mut Option<LayerId>,
) {
    let active_entry = project.projects.iter().find(|entry| entry.is_active);
    let selected_label = selected.and_then(|selected| {
        active_entry.and_then(|entry| {
            entry
                .layers
                .iter()
                .find(|layer| layer.is_loaded && layer.id == selected)
                .map(|layer| layer.name.clone())
        })
    });
    let options = active_entry.into_iter().flat_map(|entry| {
        entry
            .layers
            .iter()
            .filter(|layer| layer.is_loaded)
            .map(|layer| (Some(layer.id), layer.name.clone().into()))
    });
    MenuFieldCombo::new(
        id,
        field_label,
        selected,
        selected_label.unwrap_or_else(|| "Choose a loaded layer".to_string()),
        options,
    )
    .width(FIELD_WIDTH)
    .show(ui);
}

fn triangulation_combo(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    field_label: &str,
    project: &UiProjectView,
    selected: &mut Option<TriangulationId>,
) {
    let label = selected
        .and_then(|id| {
            project
                .triangulations
                .iter()
                .find(|entry| entry.id == Some(id) && entry.is_loaded)
        })
        .map(|entry| entry.name.as_str())
        .unwrap_or("Choose a loaded triangulation");
    MenuFieldCombo::new(
        id,
        field_label,
        selected,
        label,
        project
            .triangulations
            .iter()
            .filter(|entry| entry.is_loaded)
            .map(|entry| (entry.id, entry.name.clone().into())),
    )
    .width(FIELD_WIDTH)
    .show(ui);
}

fn selected_import_source_paths(editor: &EditorState) -> &[std::path::PathBuf] {
    if editor.import_source_menu == editor.data_menu {
        &editor.import_source_paths
    } else {
        &[]
    }
}

fn optional_path_slice(path: &Option<std::path::PathBuf>) -> &[std::path::PathBuf] {
    path.as_ref().map(std::slice::from_ref).unwrap_or(&[])
}

fn reset_import_defaults(editor: &mut EditorState, _project: &UiProjectView) {
    if editor.import_source_menu == editor.data_menu {
        editor.import_source_menu = DataMenu::None;
        editor.import_source_paths.clear();
    }
    match editor.data_menu {
        DataMenu::Dxf => {
            editor.import_dxf_as_pidb = true;
        }
        DataMenu::Bmf => {
            editor.import_bmf_path = None;
            editor.import_bdf_path = None;
        }
        _ => {}
    }
}

fn reset_export_defaults(editor: &mut EditorState, project: &UiProjectView) {
    match editor.data_menu {
        DataMenu::Dxf => {
            editor.export_dxf_layer = false;
            editor.export_layer = first_loaded_layer(project);
        }
        DataMenu::Pidb => {}
        DataMenu::Tri00t | DataMenu::Obj | DataMenu::Stl | DataMenu::Ply => {
            editor.export_triangulation = first_loaded_triangulation(project);
        }
        _ => {}
    }
}

fn ensure_export_layer(editor: &mut EditorState, project: &UiProjectView) {
    if !has_loaded_layer(project, editor.export_layer) {
        editor.export_layer = first_loaded_layer(project);
    }
}

fn ensure_export_triangulation(editor: &mut EditorState, project: &UiProjectView) {
    if !has_loaded_triangulation(project, editor.export_triangulation) {
        editor.export_triangulation = first_loaded_triangulation(project);
    }
}

fn import_command(editor: &EditorState) -> Option<UiCommand> {
    let source_paths = selected_import_source_paths(editor);
    match editor.data_menu {
        DataMenu::Dxf if editor.import_dxf_as_pidb => (!source_paths.is_empty())
            .then(|| UiCommand::ImportAsPidbPaths(DataMenu::Dxf, source_paths.to_vec())),
        DataMenu::Dxf => {
            (!source_paths.is_empty()).then(|| UiCommand::ImportDxfPathsInto(source_paths.to_vec()))
        }
        DataMenu::Pidb => {
            (!source_paths.is_empty()).then(|| UiCommand::OpenPidbPaths(source_paths.to_vec()))
        }
        DataMenu::DgdIsis => (!source_paths.is_empty())
            .then(|| UiCommand::ImportAsPidbPaths(DataMenu::DgdIsis, source_paths.to_vec())),
        DataMenu::Duf => (!source_paths.is_empty())
            .then(|| UiCommand::ImportAsPidbPaths(DataMenu::Duf, source_paths.to_vec())),
        DataMenu::Tri00t | DataMenu::Obj | DataMenu::Stl | DataMenu::Ply => (!source_paths
            .is_empty())
        .then(|| UiCommand::ImportTriangulationPaths(source_paths.to_vec())),
        DataMenu::Las | DataMenu::Xyz | DataMenu::Pcd => (!source_paths.is_empty())
            .then(|| UiCommand::ImportPointCloudPaths(source_paths.to_vec())),
        DataMenu::Geotiff => {
            (!source_paths.is_empty()).then(|| UiCommand::ImportRasterPaths(source_paths.to_vec()))
        }
        DataMenu::Bmf => {
            editor
                .import_bmf_path
                .clone()
                .map(|bmf_path| UiCommand::ConfirmImportBlockModel {
                    bmf_path,
                    bdf_path: editor.import_bdf_path.clone(),
                })
        }
        DataMenu::None => None,
    }
}

fn export_command(editor: &EditorState) -> Option<UiCommand> {
    match editor.data_menu {
        DataMenu::Dxf if editor.export_dxf_layer => {
            editor.export_layer.map(UiCommand::ExportLayerDxf)
        }
        DataMenu::Dxf => Some(UiCommand::ExportPidbDxf),
        DataMenu::Pidb => Some(UiCommand::ExportPidbCopy),
        DataMenu::Tri00t | DataMenu::Obj | DataMenu::Stl | DataMenu::Ply => {
            let format = mesh_format(editor.data_menu)?;
            editor
                .export_triangulation
                .map(|id| UiCommand::ExportTriangulationAs(id, format))
        }
        _ => None,
    }
}

fn mesh_format(data_menu: DataMenu) -> Option<MeshFormat> {
    match data_menu {
        DataMenu::Tri00t => Some(MeshFormat::T00),
        DataMenu::Obj => Some(MeshFormat::Obj),
        DataMenu::Stl => Some(MeshFormat::Stl),
        DataMenu::Ply => Some(MeshFormat::Ply),
        _ => None,
    }
}

fn is_import_menu(data_menu: DataMenu) -> bool {
    matches!(
        data_menu,
        DataMenu::Dxf
            | DataMenu::Pidb
            | DataMenu::DgdIsis
            | DataMenu::Duf
            | DataMenu::Tri00t
            | DataMenu::Obj
            | DataMenu::Stl
            | DataMenu::Ply
            | DataMenu::Las
            | DataMenu::Xyz
            | DataMenu::Pcd
            | DataMenu::Bmf
            | DataMenu::Geotiff
    )
}

fn is_export_menu(data_menu: DataMenu) -> bool {
    matches!(
        data_menu,
        DataMenu::Dxf
            | DataMenu::Pidb
            | DataMenu::Tri00t
            | DataMenu::Obj
            | DataMenu::Stl
            | DataMenu::Ply
    )
}

fn first_loaded_layer(project: &UiProjectView) -> Option<LayerId> {
    project
        .projects
        .iter()
        .find(|entry| entry.is_active)
        .and_then(|entry| {
            entry
                .layers
                .iter()
                .find(|layer| layer.is_loaded)
                .map(|layer| layer.id)
        })
}

fn first_loaded_triangulation(project: &UiProjectView) -> Option<TriangulationId> {
    project
        .triangulations
        .iter()
        .find(|entry| entry.is_loaded)
        .and_then(|entry| entry.id)
}

fn has_loaded_layer(project: &UiProjectView, selected: Option<LayerId>) -> bool {
    selected.is_some_and(|selected| {
        project.projects.iter().any(|entry| {
            entry.is_active
                && entry
                    .layers
                    .iter()
                    .any(|layer| layer.is_loaded && layer.id == selected)
        })
    })
}

fn has_loaded_triangulation(project: &UiProjectView, selected: Option<TriangulationId>) -> bool {
    selected.is_some_and(|id| {
        project
            .triangulations
            .iter()
            .any(|entry| entry.is_loaded && entry.id == Some(id))
    })
}
