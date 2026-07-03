//! Left-side explorer panel: design databases (.pidb) and triangulations.

use std::path::PathBuf;

use crate::ui::{
    EditorState, UiCommand, UiProjectView,
    fonts::bold,
    unthemed_icon,
    widgets::explorer::{ExplorerEntry, ExplorerHeader},
};

/// Grey colour used for inactive (not loaded) layers and triangulations.
const INACTIVE_TEXT_COLOR: egui::Color32 = egui::Color32::from_gray(140);

/// Draw the left explorer panel.
///
/// Shows the active PIDB path, a collapsible Design Databases section, and
/// a collapsible Triangulations section (with folder grouping).  Returns the
/// panel's bounding rect.
pub(crate) fn draw_explorer(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    project: &UiProjectView,
    commands: &mut Vec<UiCommand>,
) -> egui::Rect {
    egui::Panel::left("explorer_panel")
        .resizable(true)
        .default_size(250.0)
        .min_size(180.0)
        .show(ui, |ui| {
            // Prevent content from forcing the panel wider than the user has dragged it.
            ui.set_max_width(ui.available_width());
            let path_str = project
                .active_path
                .as_deref()
                .and_then(|p| p.to_str())
                .unwrap_or("No active PIDB");
            ui.add(egui::Label::new(path_str).truncate());
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                ExplorerHeader::new(
                    egui::Id::new("design_db_colapse"),
                    unthemed_icon!("database.svg"),
                    bold("Design Databases"),
                )
                .default_open(true)
                .show(ui, |ui| {
                    if project.projects.is_empty() {
                        ui.label("No open .pidb");
                    }
                    for entry in &project.projects {
                        let title = entry.name.clone();
                        let proj_idx = entry.index;

                        let collapse_id = ui.make_persistent_id(("pidb_project", entry.index));

                        let contains_active_layer =
                            editor.active_layer.is_some_and(|active_layer| {
                                entry.layers.iter().any(|layer| layer.id == active_layer)
                            });
                        let label = if contains_active_layer {
                            bold(&title).color(crate::ui::SELECTION_COLOR)
                        } else {
                            bold(&title)
                        };

                        let is_any_layers_loaded = entry.layers.iter().any(|layer| layer.is_loaded);
                        let path_tooltip = entry
                            .path
                            .as_deref()
                            .and_then(|p| p.to_str())
                            .unwrap_or("Unsaved")
                            .to_owned();
                        let src = if is_any_layers_loaded {
                            unthemed_icon!("unlocked.svg")
                        } else {
                            unthemed_icon!("locked.svg")
                        };
                        let (toggle_response, header_response, _body_response) =
                            ExplorerHeader::new(collapse_id, src, label)
                                .default_open(false)
                                .show(ui, |ui| {
                                    if entry.layers.is_empty() {
                                        ui.label("No layers under this .pidb");
                                    }
                                    for layer in &entry.layers {
                                        let layer_id = layer.id;
                                        let is_active = entry.is_active
                                            && editor.active_layer == Some(layer_id);
                                        let layer_display_name = if layer.dirty {
                                            format!("{} *", layer.name)
                                        } else {
                                            layer.name.clone()
                                        };
                                        let layer_label = if layer.is_loaded {
                                            bold(&layer_display_name)
                                        } else {
                                            egui::RichText::new(&layer_display_name)
                                                .color(INACTIVE_TEXT_COLOR)
                                        };
                                        let layer_resp = ui.add(
                                            ExplorerEntry::new(
                                                unthemed_icon!("mesh.svg"),
                                                layer_label,
                                            )
                                            .selected(is_active),
                                        );
                                        if layer_resp.double_clicked() {
                                            if layer.is_loaded {
                                                commands.push(UiCommand::UnloadLayer(
                                                    proj_idx, layer_id,
                                                ));
                                            } else {
                                                commands
                                                    .push(UiCommand::LoadLayer(proj_idx, layer_id));
                                            }
                                        }
                                        layer_resp.context_menu(|ui| {
                                            if layer.is_loaded {
                                                if ui.button("Unload").clicked() {
                                                    commands.push(UiCommand::UnloadLayer(
                                                        proj_idx, layer_id,
                                                    ));
                                                    ui.close();
                                                }
                                            } else if ui.button("Load").clicked() {
                                                commands
                                                    .push(UiCommand::LoadLayer(proj_idx, layer_id));
                                                ui.close();
                                            }
                                            if layer.is_loaded
                                                && ui.button("Select All Objects").clicked()
                                            {
                                                commands.push(UiCommand::SelectAllObjectsInLayer(
                                                    proj_idx, layer_id,
                                                ));
                                                ui.close();
                                            }
                                            if ui.button("Save").clicked() {
                                                commands.push(UiCommand::SaveProjectForLayer(
                                                    proj_idx, layer_id,
                                                ));
                                                ui.close();
                                            }
                                            if ui.button("Rename").clicked() {
                                                commands.push(UiCommand::BeginRenameLayer(
                                                    proj_idx, layer_id,
                                                ));
                                                ui.close();
                                            }
                                            if ui.button("Duplicate Layer").clicked() {
                                                commands.push(UiCommand::DuplicateLayer(
                                                    proj_idx, layer_id,
                                                ));
                                                ui.close();
                                            }
                                            if ui.button("Move Layer...").clicked() {
                                                commands.push(UiCommand::BeginMoveLayer(
                                                    proj_idx, layer_id,
                                                ));
                                                ui.close();
                                            }
                                            ui.separator();
                                            if ui.button("Delete Layer").clicked() {
                                                commands.push(UiCommand::RequestDeleteLayer(
                                                    proj_idx, layer_id,
                                                ));
                                                ui.close();
                                            }
                                        });
                                    }
                                });

                        let content_response = header_response.inner;
                        let project_header_response = toggle_response
                            .union(header_response.response)
                            .union(content_response.clone())
                            .on_hover_text(&path_tooltip);

                        project_header_response.context_menu(|ui| {
                            if ui.button("Save All Layers").clicked() {
                                commands.push(UiCommand::SaveNamedPidb(proj_idx));
                                ui.close();
                            }
                            if ui.button("Save As…").clicked() {
                                commands.push(UiCommand::SaveNamedPidbAs(proj_idx));
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("Load All Layers").clicked() {
                                commands.push(UiCommand::LoadAllLayersInProject(proj_idx));
                                ui.close();
                            }
                            if ui.button("Unload All Layers").clicked() {
                                commands.push(UiCommand::UnloadAllLayersInProject(proj_idx));
                                ui.close();
                            }
                            if entry.path.is_some() && ui.button("Reveal in Explorer").clicked() {
                                commands.push(UiCommand::RevealPidb(proj_idx));
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("Remove PIDB").clicked() {
                                commands.push(UiCommand::ClosePidb(proj_idx));
                                ui.close();
                            }
                        });
                    }
                });

                ExplorerHeader::new(
                    egui::Id::new("triangulation_colapse"),
                    unthemed_icon!("triangle.svg"),
                    bold("Triangulations"),
                )
                .default_open(!project.triangulations.is_empty())
                .show(ui, |ui| {
                    if project.triangulations.is_empty() {
                        ui.label("No open triangulations");
                    }

                    // Helper closure: render one tri entry row and attach its context menu.
                    let render_tri_entry =
                        |ui: &mut egui::Ui,
                         commands: &mut Vec<UiCommand>,
                         tri: &crate::ui::UiTriangulationEntry,
                         removable: bool| {
                            let tri_path = if tri.is_saved {
                                tri.path.to_str().unwrap_or("").to_owned()
                            } else {
                                "Unsaved triangulation".to_owned()
                            };
                            let tri_path_buf = tri.path.clone();

                            let label = if tri.is_loaded {
                                let dirty_marker = if tri.is_saved { "" } else { " *" };
                                let stats = format!("{}{}", tri.name, dirty_marker);
                                bold(&stats)
                            } else {
                                egui::RichText::new(&tri.name).color(INACTIVE_TEXT_COLOR)
                            };

                            let icon = unthemed_icon!("cube.svg");
                            let response = ui
                                .add(ExplorerEntry::new(icon, label).selected(tri.is_active))
                                .on_hover_text(&tri_path);

                            if response.double_clicked() {
                                if tri.is_loaded {
                                    commands.push(UiCommand::CloseTriangulation(tri.id.unwrap()));
                                } else {
                                    commands
                                        .push(UiCommand::LoadTriangulation(tri_path_buf.clone()));
                                }
                            } else if response.clicked() && tri.is_loaded {
                                commands.push(UiCommand::ActivateTriangulation(tri.id.unwrap()));
                            }

                            if tri.is_loaded {
                                let tri_id = tri.id.unwrap();
                                let tri_visible = tri.visible;
                                let tri_saved = tri.is_saved;
                                response.context_menu(|ui| {
                                    if ui.button("Unload").clicked() {
                                        commands.push(UiCommand::CloseTriangulation(tri_id));
                                        ui.close();
                                    }

                                    if ui
                                        .button(if tri_visible { "Hide" } else { "Show" })
                                        .clicked()
                                    {
                                        commands
                                            .push(UiCommand::ToggleTriangulationVisible(tri_id));
                                        ui.close();
                                    }
                                    ui.separator();
                                    if tri_saved {
                                        if ui.button("Show in Explorer").clicked() {
                                            commands.push(UiCommand::RevealTriangulation(tri_id));
                                            ui.close();
                                        }
                                    } else if ui.button("Save As ...").clicked() {
                                        commands.push(UiCommand::SaveTriangulationAs(tri_id));
                                        ui.close();
                                    }
                                    if tri_saved
                                        && removable
                                        && ui.button("Remove Triangulation").clicked()
                                    {
                                        ui.separator();
                                        commands.push(UiCommand::RemoveTriangulation(
                                            tri_path_buf.clone(),
                                        ));
                                        ui.close();
                                    }
                                });
                            } else {
                                response.context_menu(|ui| {
                                    ui.set_min_width(70.);
                                    if ui.button("Load").clicked() {
                                        commands.push(UiCommand::LoadTriangulation(
                                            tri_path_buf.clone(),
                                        ));
                                        ui.close();
                                    }
                                    if removable {
                                        ui.separator();
                                        if ui.button("Remove Triangulation").clicked() {
                                            commands.push(UiCommand::RemoveTriangulation(
                                                tri_path_buf.clone(),
                                            ));
                                            ui.close();
                                        }
                                    }
                                });
                            }
                        };

                    // Individually-opened files (no group) — shown flat, removable.
                    let individual: Vec<_> = project
                        .triangulations
                        .iter()
                        .filter(|t| t.group.is_none())
                        .collect();
                    for tri in &individual {
                        render_tri_entry(ui, commands, tri, true);
                    }

                    // Folder groups — each dir gets a collapsible sub-header.
                    let mut seen_dirs: Vec<PathBuf> = Vec::new();
                    for tri in &project.triangulations {
                        if let Some(ref dir) = tri.group
                            && !seen_dirs.contains(dir)
                        {
                            seen_dirs.push(dir.clone());
                        }
                    }
                    for dir in seen_dirs {
                        let dir_name = dir
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(dir.to_str().unwrap_or("folder"))
                            .to_owned();
                        let dir_for_menu = dir.clone();
                        let collapse_id = ui.make_persistent_id(("tri_folder", &dir));
                        let folder_tris: Vec<_> = project
                            .triangulations
                            .iter()
                            .filter(|t| t.group.as_deref() == Some(dir.as_path()))
                            .collect();
                        let any_loaded = folder_tris.iter().any(|t| t.is_loaded);
                        let any_unloaded = folder_tris.iter().any(|t| !t.is_loaded);
                        let img = if any_loaded {
                            unthemed_icon!("unlocked.svg")
                        } else {
                            unthemed_icon!("locked.svg")
                        };
                        let (toggle_resp, header_resp, _) =
                            ExplorerHeader::new(collapse_id, img, bold(&dir_name)).show(ui, |ui| {
                                for tri in folder_tris {
                                    render_tri_entry(ui, commands, tri, false);
                                }
                            });
                        let folder_response = toggle_resp
                            .union(header_resp.response)
                            .union(header_resp.inner);
                        folder_response.context_menu(|ui| {
                            if any_unloaded && ui.button("Load All").clicked() {
                                commands.push(UiCommand::ConfirmLoadAllTriangulationsInFolder(
                                    dir_for_menu.clone(),
                                ));
                                ui.close();
                            }
                            if any_loaded && ui.button("Unload All").clicked() {
                                commands.push(UiCommand::CloseAllTriangulationsInFolder(
                                    dir_for_menu.clone(),
                                ));
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("Remove Folder").clicked() {
                                commands.push(UiCommand::RemoveTriangulationFolder(dir_for_menu));
                                ui.close();
                            }
                        });
                    }
                });

                ExplorerHeader::new(
                    egui::Id::new("block_models_collapse"),
                    unthemed_icon!("block_model.svg"),
                    bold("Block Models"),
                )
                .default_open(!project.block_models.is_empty())
                .show(ui, |ui| {
                    if project.block_models.is_empty() {
                        ui.label("No open block models");
                    }
                    for block_model in &project.block_models {
                        let label_text = block_model.name.clone();
                        let label = if block_model.is_loaded {
                            bold(&label_text)
                        } else {
                            egui::RichText::new(&label_text).color(INACTIVE_TEXT_COLOR)
                        };
                        let response = ui
                            .add(
                                ExplorerEntry::new(unthemed_icon!("block_model_entry.svg"), label)
                                    .selected(block_model.is_active),
                            )
                            .on_hover_text(format!(
                                "{}\n{}\n{} numeric variable(s)",
                                block_model.source.bmf_path.display(),
                                block_model
                                    .source
                                    .bdf_path
                                    .as_ref()
                                    .map(|path| path.display().to_string())
                                    .unwrap_or_else(|| "No companion .bdf attached".to_owned()),
                                block_model.numeric_variable_count
                            ));
                        if response.double_clicked() {
                            if let Some(id) = block_model.id {
                                commands.push(UiCommand::CloseBlockModel(id));
                            } else {
                                commands
                                    .push(UiCommand::LoadBlockModel(block_model.source.clone()));
                            }
                        }

                        response.context_menu(|ui| {
                            if let Some(id) = block_model.id {
                                if ui.button("Unload").clicked() {
                                    commands.push(UiCommand::CloseBlockModel(id));
                                    ui.close();
                                }
                                if ui
                                    .button(if block_model.visible { "Hide" } else { "Show" })
                                    .clicked()
                                {
                                    commands.push(UiCommand::ToggleBlockModelVisible(id));
                                    ui.close();
                                }
                                if ui.button("Open Table").clicked() {
                                    commands.push(UiCommand::OpenBlockModelTable(id));
                                    ui.close();
                                }
                                if ui.button("Create Ore Triangulation ...").clicked() {
                                    commands.push(UiCommand::OpenCreateOreTriangulation);
                                    ui.close();
                                }
                                ui.separator();
                                if ui.button("Set Definition File ...").clicked() {
                                    commands.push(UiCommand::SetBlockModelDefinitionFile(id));
                                    ui.close();
                                }
                                if ui.button("Show in Explorer").clicked() {
                                    commands.push(UiCommand::RevealBlockModel(id));
                                    ui.close();
                                }
                                ui.separator();
                            } else {
                                if ui.button("Load").clicked() {
                                    commands.push(UiCommand::LoadBlockModel(
                                        block_model.source.clone(),
                                    ));
                                    ui.close();
                                }
                                if ui.button("Set Definition File ...").clicked() {
                                    commands.push(UiCommand::SetBlockModelSourceDefinitionFile(
                                        block_model.source.clone(),
                                    ));
                                    ui.close();
                                }
                            }
                            if ui.button("Remove").clicked() {
                                commands
                                    .push(UiCommand::RemoveBlockModel(block_model.source.clone()));
                                ui.close();
                            }
                        });
                    }
                })
            });
        })
        .response
        .rect
}
