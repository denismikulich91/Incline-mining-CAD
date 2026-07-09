pub(crate) mod block_model;
pub(crate) mod drawing; // Handles finishing polygons, creating points, etc commands
pub(crate) mod file; // Handles importing, exportings, etc. commands
pub(crate) mod layer; // Handles creating layers, deleting layers, etc. commands
pub(crate) mod point_cloud; // Handles importing/loading point clouds, etc. commands
pub(crate) mod property; // Handles changing colors, fills, etc. commands
pub(crate) mod text; // Handles text editing commands
pub(crate) mod triangulation; // Handles loading meshes, deleting meshes, etc. commands
pub(crate) mod view; /* Handles resetting camera view, , etc. commands */

use anyhow::Result;

use crate::ui::state::{MoveLayerDialog, TriCreatePhase, TriSurfaceType, UiCommand};
use crate::{app::App, userspace_warn};

impl<'a> App<'a> {
    pub(crate) fn apply_history_step(&mut self, undo: bool) {
        let changed = self.workspace.active_project_mut().is_some_and(|project| {
            if undo {
                self.history.undo(&mut project.pidb.document)
            } else {
                self.history.redo(&mut project.pidb.document)
            }
        });
        if !changed {
            return;
        }
        self.refresh_active_project_dirty();
        if self
            .editor
            .active_layer
            .is_some_and(|layer_id| self.active_layer() != Some(layer_id))
        {
            self.editor.active_layer = None;
        }
        self.editor.selected_handles.clear();
        self.editor.canvas_context_menu_open = false;
        self.cancel_fuse();
        self.invalidate_geometry();
    }

    pub(crate) fn handle_ui_commands(&mut self, commands: Vec<UiCommand>) {
        let had_commands = !commands.is_empty();
        for command in commands {
            if let Err(err) = self.handle_ui_command(command) {
                let message = format!("{err:#}");
                userspace_warn!("Command failed: {message}");
            }
        }
        // A "Save and Unload" may have cleaned a project that still has other
        // layers queued for confirmation; unload those without re-prompting.
        self.drain_clean_unload_queue();
        if had_commands {
            self.redraw_requested = true;
        }
    }
    /// Dispatch a UI command to the relevant domain handler (in the sibling
    /// `commands::*` modules). This is a thin router; the work lives in those
    /// per-domain `impl` blocks.
    pub(crate) fn handle_ui_command(&mut self, command: UiCommand) -> Result<()> {
        match command {
            UiCommand::SetActiveTool(tool) => {
                self.set_active_tool_from_toolbar(tool);
                Ok(())
            }
            UiCommand::SetFlyModeEnabled(enabled) => {
                self.set_fly_mode_enabled(enabled);
                Ok(())
            }
            UiCommand::NewPidb => {
                self.choose_new_pidb();
                Ok(())
            }
            UiCommand::OpenPidb => {
                self.choose_open_pidb();
                Ok(())
            }
            UiCommand::OpenPidbPaths(paths) => self.execute_file_dialog_action(
                crate::app::commands::file::FileDialogAction::OpenPidb(paths),
            ),
            UiCommand::CloseStartupDialog => {
                self.startup_dialog_dismissed = true;
                Ok(())
            }
            UiCommand::SaveAllPidbs => self.save_all_dirty_projects().map(|_| ()),
            UiCommand::ImportAsPidbPaths(kind, paths) => {
                self.choose_import_as_pidb_paths(kind, paths);
                Ok(())
            }
            UiCommand::ImportDxfPathsInto(index, paths) => self.import_dxf_paths_into(index, paths),
            UiCommand::ImportTriangulationPaths(paths) => self.execute_file_dialog_action(
                crate::app::commands::file::FileDialogAction::ImportTriangulation(paths),
            ),
            UiCommand::ImportPointCloudPaths(paths) => self.execute_file_dialog_action(
                crate::app::commands::file::FileDialogAction::ImportPointCloud(paths),
            ),
            UiCommand::LoadPointCloud(path) => {
                self.open_point_cloud_path(path);
                Ok(())
            }
            UiCommand::ClosePointCloud(id) => {
                self.close_point_cloud(id);
                Ok(())
            }
            UiCommand::TogglePointCloudVisible(id) => {
                self.toggle_point_cloud_visible(id);
                Ok(())
            }
            UiCommand::RemovePointCloud(path) => {
                self.remove_point_cloud(&path);
                Ok(())
            }
            UiCommand::RevealPointCloud(id) => self.reveal_point_cloud(id),
            UiCommand::ChooseImportSourceFiles(kind) => {
                self.choose_import_source_files(kind);
                Ok(())
            }
            UiCommand::ChooseBlockModelBmf => {
                self.choose_block_model_bmf();
                Ok(())
            }
            UiCommand::ChooseBlockModelBdf => {
                self.choose_block_model_bdf();
                Ok(())
            }
            UiCommand::ConfirmImportBlockModel { bmf_path, bdf_path } => self
                .import_block_model_source(crate::model::block_model::BlockModelSource {
                    bmf_path,
                    bdf_path,
                }),
            UiCommand::OpenTriangulationFolder => {
                self.choose_open_triangulation_folder();
                Ok(())
            }
            UiCommand::ExportPidbDxf(index) => {
                self.choose_export_pidb_dxf(index);
                Ok(())
            }
            UiCommand::ExportPidbCopy(index) => {
                self.choose_export_pidb_copy(index);
                Ok(())
            }
            UiCommand::ExportLayerDxf(index, layer) => {
                self.choose_export_layer_dxf(index, layer);
                Ok(())
            }
            UiCommand::ExportTriangulationAs(id, format) => {
                self.choose_export_triangulation_as(id, format);
                Ok(())
            }
            UiCommand::SaveTriangulationAs(id) => {
                self.spawn_save_triangulation_dialog(id);
                Ok(())
            }
            UiCommand::SaveAndCloseTriangulationAs(id) => {
                self.spawn_save_and_close_triangulation_dialog(id);
                Ok(())
            }
            UiCommand::RevealTriangulation(id) => self.reveal_triangulation(id),
            UiCommand::RevealBlockModel(id) => self.reveal_block_model(id),
            UiCommand::RevealPidb(index) => self.reveal_pidb(index),
            UiCommand::RevealAllTriangulations => {
                for tri in &mut self.triangulations {
                    tri.visible = true;
                }
                self.invalidate_geometry();
                Ok(())
            }
            UiCommand::RequestExit => self.request_exit(),
            UiCommand::SaveAndExit => self.save_and_exit(),
            UiCommand::ExitWithoutSaving => {
                self.exit_without_saving();
                Ok(())
            }
            UiCommand::CancelExit => {
                self.cancel_exit_request();
                Ok(())
            }
            UiCommand::CreateLayer {
                project_index,
                name,
            } => self.create_layer(project_index, name),
            UiCommand::RequestDeleteLayer(project_index, layer_id) => {
                let layer_name = self
                    .workspace
                    .projects
                    .get(project_index)
                    .and_then(|project| project.pidb.document.layer(layer_id))
                    .map(|layer| layer.name.clone());
                self.editor.pending_delete_layer =
                    layer_name.map(|name| (project_index, layer_id, name));
                Ok(())
            }
            UiCommand::DeleteLayer(project_index, layer_id) => {
                self.delete_layer(project_index, layer_id)
            }
            UiCommand::DuplicateLayer(project_index, layer_id) => {
                self.duplicate_layer(project_index, layer_id);
                Ok(())
            }
            UiCommand::BeginMoveLayer(project_index, layer_id) => {
                let target_project_index = self
                    .workspace
                    .projects
                    .iter()
                    .enumerate()
                    .find(|(index, _)| *index != project_index)
                    .map(|(index, _)| index);
                self.editor.move_layer_dialog = Some(MoveLayerDialog {
                    source_project_index: project_index,
                    layer_id,
                    target_project_index,
                });
                Ok(())
            }
            UiCommand::MoveLayerToProject {
                source_project_index,
                layer_id,
                target_project_index,
            } => {
                self.move_layer_to_project(source_project_index, layer_id, target_project_index);
                Ok(())
            }
            UiCommand::BeginRenameLayer(project_index, layer_id) => {
                let current_name = self
                    .workspace
                    .projects
                    .get(project_index)
                    .and_then(|p| p.pidb.document.layer(layer_id))
                    .map(|l| l.name.clone())
                    .unwrap_or_default();
                self.editor.renaming_layer = Some((project_index, layer_id, current_name));
                Ok(())
            }
            UiCommand::RenameLayer {
                project_index,
                layer_id,
                new_name,
            } => {
                let Some((before, is_loaded)) = self
                    .workspace
                    .projects
                    .get(project_index)
                    .and_then(|project| {
                        project.pidb.document.layer(layer_id).map(|layer| {
                            (
                                layer.name.clone(),
                                project.loaded_layers.contains(&layer_id),
                            )
                        })
                    })
                else {
                    self.editor.renaming_layer = None;
                    return Ok(());
                };
                if before != new_name {
                    let mut save_unloaded_rename = false;
                    if self.workspace.active_index != Some(project_index) && is_loaded {
                        self.history.clear();
                        self.workspace.set_active_index(project_index);
                        self.editor.selected_handles.clear();
                    }
                    if is_loaded {
                        self.editor.active_layer = Some(layer_id);
                        if let Some(project) = self.workspace.projects.get_mut(project_index) {
                            self.history.execute(
                                &mut project.pidb.document,
                                crate::model::Command::RenameLayer {
                                    id: layer_id,
                                    before,
                                    after: new_name,
                                },
                            );
                        }
                    } else {
                        self.history.clear();
                        if self.editor.active_layer == Some(layer_id) {
                            self.editor.active_layer = None;
                        }
                        if let Some(project) = self.workspace.projects.get_mut(project_index) {
                            project.pidb.document.rename_layer(layer_id, new_name);
                            project.invalidate_dirty_layers();
                            save_unloaded_rename = project.path.is_some();
                        }
                    }
                    if save_unloaded_rename {
                        self.save_project_layer(project_index, layer_id)?;
                    }
                }
                self.editor.renaming_layer = None;
                Ok(())
            }
            UiCommand::LoadLayer(project, layer) => {
                self.load_layer(project, layer);
                Ok(())
            }
            UiCommand::UnloadLayer(project, layer) => {
                self.try_unload_layer(project, layer);
                Ok(())
            }
            UiCommand::SelectAllObjectsInLayer(project, layer) => {
                self.select_all_objects_in_layer(project, layer);
                Ok(())
            }
            UiCommand::UnloadLayerConfirmed(project, layer) => {
                self.unload_layer_discarding_changes(project, layer);
                Ok(())
            }
            UiCommand::SaveAndUnloadLayer(project, layer) => {
                self.save_and_unload_layer(project, layer)
            }
            UiCommand::LoadAllLayersInProject(index) => {
                self.load_all_layers_in_project(index);
                Ok(())
            }
            UiCommand::UnloadAllLayersInProject(index) => {
                self.unload_all_layers_in_project(index);
                Ok(())
            }
            UiCommand::SaveProjectForLayer(index, layer) => self.save_project_layer(index, layer),
            UiCommand::SaveNamedPidb(index) => self.save_named_project(index),
            UiCommand::SaveNamedPidbAs(index) => {
                self.spawn_save_pidb_as_dialog(index);
                Ok(())
            }
            UiCommand::ClosePidb(index) => {
                self.request_close_project(index);
                Ok(())
            }
            UiCommand::SaveAndClosePidb(index) => self.save_and_close_project(index),
            UiCommand::ClosePidbForce(index) => self.close_project(index),
            UiCommand::LoadTriangulation(path) => self.open_triangulation_path(&path),
            UiCommand::LoadBlockModel(source) => self.open_block_model_source(source),
            UiCommand::CloseBlockModel(id) => {
                self.close_block_model(id);
                Ok(())
            }
            UiCommand::RemoveBlockModel(source) => {
                self.remove_block_model(source);
                Ok(())
            }
            UiCommand::ToggleBlockModelVisible(id) => {
                self.toggle_block_model_visible(id);
                Ok(())
            }
            UiCommand::SetBlockModelColorVariable { id, variable } => {
                self.set_block_model_color_variable(id, variable);
                Ok(())
            }
            UiCommand::SetBlockModelColorStops { id, stops } => {
                self.set_block_model_color_stops(id, stops);
                Ok(())
            }
            UiCommand::SetBlockModelHideEmptyValues { id, hide } => {
                self.set_block_model_hide_empty_values(id, hide);
                Ok(())
            }
            UiCommand::SetBlockModelDefinitionFile(id) => {
                self.choose_set_block_model_bdf(id);
                Ok(())
            }
            UiCommand::SetBlockModelSourceDefinitionFile(source) => {
                self.choose_set_block_model_source_bdf(source);
                Ok(())
            }
            UiCommand::OpenBlockModelTable(id) => {
                self.editor.block_model_table_pages.entry(id).or_insert(0);
                if let Some(model) = self.block_models.iter().find(|model| model.id == id) {
                    crate::ui::elements::tabs::open_block_model_table(id, model.name.clone());
                }
                Ok(())
            }
            UiCommand::OpenCreateOreTriangulation => {
                self.editor.ore_triangulation_open = true;
                self.editor.ore_block_model_id = self
                    .active_block_model
                    .or_else(|| self.block_models.first().map(|model| model.id));
                if let Some(model) = self
                    .editor
                    .ore_block_model_id
                    .and_then(|id| self.block_models.iter().find(|model| model.id == id))
                {
                    self.editor.ore_variable = model
                        .active_numeric_variable
                        .clone()
                        .or_else(|| {
                            model
                                .model
                                .numeric_variables()
                                .into_iter()
                                .find(|var| !var.special)
                                .map(|var| var.name.clone())
                        })
                        .unwrap_or_default();
                    let stem = std::path::Path::new(&model.name)
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or(&model.name);
                    self.editor.ore_name_input = format!("{stem}_ore");
                }
                Ok(())
            }
            UiCommand::ExecuteCreateOreTriangulation {
                block_model_id,
                variable,
                mode,
                min,
                max,
                name,
            } => {
                let result =
                    self.create_ore_triangulation(block_model_id, variable, mode, min, max, name);
                if result.is_ok() {
                    self.editor.ore_triangulation_open = false;
                }
                result
            }
            UiCommand::FinishPolyClose => {
                self.finish_poly_closed();
                Ok(())
            }
            UiCommand::CommitStrokeOpen => {
                self.commit_stroke_open();
                Ok(())
            }
            UiCommand::CancelOffset => {
                self.cancel_offset();
                Ok(())
            }
            UiCommand::CancelRelimit => {
                self.cancel_relimit();
                Ok(())
            }
            UiCommand::ResetView => {
                self.reset_view();
                Ok(())
            }
            UiCommand::SetTopologyWireframes(enabled) => self.set_topology_wireframes(enabled),
            UiCommand::SetDarkMode(enabled) => self.set_dark_mode(enabled),
            UiCommand::SetShowConsole(enabled) => self.set_show_console(enabled),
            UiCommand::SetShowWorldAxisGizmo(enabled) => self.set_show_world_axis_gizmo(enabled),
            UiCommand::SetDebugChunkColoring(enabled) => self.set_debug_chunk_coloring(enabled),
            UiCommand::SetStandardView(view) => {
                if let Some(graphics) = self.graphics.as_mut() {
                    graphics.set_standard_view(view);
                    self.redraw_requested = true;
                }
                Ok(())
            }
            UiCommand::ApplyPreferences(preferences) => self.apply_preferences(preferences),
            UiCommand::OpenPreferences => {
                self.editor.reset_preferences_draft();
                crate::ui::elements::tabs::open_preferences();
                Ok(())
            }
            UiCommand::RemoveTriangulation(path) => {
                self.remove_triangulation(path);
                Ok(())
            }
            UiCommand::RemoveTriangulationFolder(dir) => {
                self.remove_triangulation_folder(dir);
                Ok(())
            }
            UiCommand::ActivateTriangulation(id) => {
                self.activate_triangulation(id);
                Ok(())
            }
            UiCommand::ToggleTriangulationVisible(id) => {
                self.toggle_triangulation_visible(id);
                Ok(())
            }
            UiCommand::CloseTriangulation(id) => {
                if let Some(tri) = self.triangulations.iter().find(|t| t.id == id)
                    && !tri.is_saved
                {
                    self.editor.tri_close_unsaved = Some(id);
                    return Ok(());
                }
                self.close_triangulation(id);
                Ok(())
            }
            UiCommand::CloseTriangulationForce(id) => {
                self.editor.tri_close_unsaved = None;
                self.close_triangulation(id);
                Ok(())
            }
            UiCommand::CommitTextEdit(object_id, content, height, rotation_degrees, color) => {
                self.commit_text_edit(object_id, content, height, rotation_degrees, color);
                Ok(())
            }
            UiCommand::CancelTextEdit => {
                self.cancel_text_edit();
                Ok(())
            }
            UiCommand::BatchSetObjectColor(ids, new_color) => {
                self.batch_set_object_color(ids, new_color);
                Ok(())
            }
            UiCommand::BatchSetPolylineClosed(ids, closed) => {
                self.batch_set_polyline_closed(ids, closed);
                Ok(())
            }
            UiCommand::BatchSetObjectFill(ids, new_fill) => {
                self.batch_set_object_fill(ids, new_fill);
                Ok(())
            }
            UiCommand::BatchSetPolylineLineWeight(ids, weight) => {
                self.batch_set_polyline_line_weight(ids, weight);
                Ok(())
            }
            UiCommand::BatchSetZValue(ids, weight) => {
                self.batch_set_z_value(ids, weight);
                Ok(())
            }
            UiCommand::MoveObjectsToLayer {
                project_index,
                object_ids,
                target_layer,
                copy,
            } => {
                self.move_objects_to_layer(project_index, object_ids, target_layer, copy);
                Ok(())
            }
            UiCommand::SetTriangulationColor(tri_id, new_color) => {
                self.set_triangulation_color(tri_id, new_color);
                Ok(())
            }
            UiCommand::CloseCanvasContextMenu => {
                self.editor.canvas_context_menu_open = false;
                Ok(())
            }
            UiCommand::ZoomToExtents => {
                self.zoom_to_extents();
                Ok(())
            }
            UiCommand::BeginOffsetPick {
                object_ids,
                horiz_dist,
                z_delta,
                project_to_rl,
                collide_with_triangulation,
            } => {
                self.begin_offset_pick(
                    object_ids,
                    horiz_dist,
                    z_delta,
                    project_to_rl,
                    collide_with_triangulation,
                );
                Ok(())
            }
            UiCommand::RelimitLineResize {
                source_id,
                mode,
                value,
            } => {
                self.relimit_resize(source_id, mode, value);
                Ok(())
            }
            UiCommand::OpenOffsetDialog => {
                self.open_offset_dialog();
                Ok(())
            }
            UiCommand::OpenRelimitDialog => {
                self.open_relimit_dialog();
                Ok(())
            }
            UiCommand::OpenBatterBermDialog => {
                self.open_batter_berm_dialog();
                Ok(())
            }
            UiCommand::OpenSetSelectionZValueDialog => {
                let selected_objects: Vec<crate::model::ObjectId> = self
                    .editor
                    .selected_handles
                    .iter()
                    .filter_map(|&handle| match handle {
                        crate::model::SceneEntityId::Object(id) => Some(id),
                        crate::model::SceneEntityId::Triangulation(_)
                        | crate::model::SceneEntityId::BlockModel(_) => None,
                    })
                    .collect();

                if selected_objects.is_empty() {
                    userspace_warn!("Select one or more objects before setting Z");
                    return Ok(());
                }

                self.editor.set_selection_z_dialog =
                    Some(crate::ui::dialogs::SetSelectionZDialog {
                        object_ids: selected_objects,
                        z_input: self.editor.z_input,
                    });
                Ok(())
            }
            UiCommand::CommitBatterBerm => {
                self.commit_batter_berm();
                Ok(())
            }
            UiCommand::CancelBatterBerm => {
                self.cancel_batter_berm();
                Ok(())
            }
            UiCommand::CommitRoad => {
                self.commit_road();
                Ok(())
            }
            UiCommand::CancelRoad => {
                self.cancel_road();
                Ok(())
            }
            UiCommand::ConvertSelectedRoadsToPolylines => {
                self.convert_selected_roads_to_polylines()
            }
            UiCommand::OpenCreateTriangulation => {
                self.editor.tri_create_open = true;
                self.editor.tri_create_phase = TriCreatePhase::MainDialog;
                self.editor.tri_create_source = crate::ui::state::TriCreateSource::Polygons;
                self.editor.selected_handles.clear();
                self.editor.tri_selected_object_ids.clear();
                self.editor.tri_selected_layer_ids.clear();
                self.editor.tri_name_input.clear();
                self.editor.tri_hover_handles.clear();
                Ok(())
            }
            UiCommand::OpenConvertRoadsToTriangulation => {
                self.editor.tri_create_open = true;
                self.editor.tri_create_phase = TriCreatePhase::MainDialog;
                self.editor.tri_create_source = crate::ui::state::TriCreateSource::Roads;
                self.editor.tri_surface_type = TriSurfaceType::Surface;
                self.editor.selected_handles.clear();
                self.editor.tri_selected_object_ids.clear();
                self.editor.tri_selected_layer_ids.clear();
                self.editor.tri_name_input.clear();
                self.editor.tri_hover_handles.clear();
                Ok(())
            }
            UiCommand::ExecuteCreateTriangulation {
                name,
                object_ids,
                surface_type,
            } => self.run_create_triangulation(name, object_ids, surface_type, false),
            UiCommand::ExecuteConvertRoadsToTriangulation { name, object_ids } => {
                self.create_triangulation_from_roads(name, object_ids)
            }
            UiCommand::ExecuteCreateTriangulationWithWeld {
                name,
                object_ids,
                surface_type,
            } => self.run_create_triangulation(name, object_ids, surface_type, true),
            UiCommand::OpenPointCloudTin => {
                self.editor.point_cloud_tin_open = true;
                // Default to the only loaded cloud, or keep a still-valid pick.
                let still_loaded = self
                    .editor
                    .point_cloud_tin_cloud_id
                    .is_some_and(|id| self.point_clouds.iter().any(|cloud| cloud.id == id));
                if !still_loaded {
                    self.editor.point_cloud_tin_cloud_id =
                        self.point_clouds.first().map(|cloud| cloud.id);
                }
                self.editor.point_cloud_tin_name_input.clear();
                Ok(())
            }
            UiCommand::ExecutePointCloudTin {
                cloud_id,
                name,
                max_edge,
                max_points,
            } => self.run_point_cloud_tin(cloud_id, name, max_edge, max_points),
            UiCommand::ConfirmLoadAllTriangulationsInFolder(path) => {
                self.editor.confirm_load_all_folder = Some(path);
                Ok(())
            }
            UiCommand::LoadAllTriangulationsInFolder(path) => {
                self.load_all_triangulations_in_folder(path)
            }
            UiCommand::CloseAllTriangulationsInFolder(path) => {
                self.close_all_triangulations_in_folder(path);
                Ok(())
            }
            UiCommand::OpenCutTriangulationByPolygon => {
                self.editor.tri_cut_poly_open = true;
                self.editor.tri_cut_poly_awaiting_pick = false;
                self.editor.tri_cut_poly_tri_id = self.active_triangulation;
                self.editor.tri_cut_poly_object_id = None;
                self.editor.tri_cut_poly_object_name = String::new();
                self.editor.tri_cut_poly_name_input = self
                    .active_triangulation
                    .and_then(|id| self.triangulations.iter().find(|t| t.id == id))
                    .map(|t| {
                        let stem = std::path::Path::new(&t.name)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or(&t.name);
                        format!("{stem}_cut")
                    })
                    .unwrap_or_default();
                Ok(())
            }
            UiCommand::BeginCutPolyPick => {
                self.editor.tri_cut_poly_awaiting_pick = true;
                Ok(())
            }
            UiCommand::ExecuteCutTriangulationByPolygon {
                tri_id,
                polygon_id,
                name,
            } => {
                let result = self.cut_triangulation_by_polygon(tri_id, polygon_id, name);
                if result.is_ok() {
                    self.editor.tri_cut_poly_open = false;
                    self.editor.tool_highlight_id = None;
                }
                result
            }
            UiCommand::OpenCutTriangulationByZ => {
                self.editor.tri_cut_z_open = true;
                self.editor.tri_cut_z_tri_id = self.active_triangulation;
                let (z_min, z_max) = self
                    .active_triangulation
                    .and_then(|id| self.triangulations.iter().find(|t| t.id == id))
                    .map(|t| {
                        let b = t.mesh.bounds();
                        (b.min.z, b.max.z)
                    })
                    .unwrap_or((0.0, 100.0));
                self.editor.tri_cut_z_min_input = z_min;
                self.editor.tri_cut_z_max_input = z_max;
                self.editor.tri_cut_z_name_input = self
                    .active_triangulation
                    .and_then(|id| self.triangulations.iter().find(|t| t.id == id))
                    .map(|t| {
                        let stem = std::path::Path::new(&t.name)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or(&t.name);
                        format!("{stem}_slice")
                    })
                    .unwrap_or_default();
                Ok(())
            }
            UiCommand::ExecuteCutTriangulationByZ {
                tri_id,
                z_min,
                z_max,
                name,
            } => {
                let result = self.cut_triangulation_by_z(tri_id, z_min, z_max, name);
                if result.is_ok() {
                    self.editor.tri_cut_z_open = false;
                }
                result
            }
            UiCommand::OpenCutTriangulationBySurface => {
                self.editor.tri_cut_surface_open = true;
                self.editor.tri_cut_surface_target_id = self.active_triangulation;
                self.editor.tri_cut_surface_reference_id = None;
                self.editor.tri_cut_surface_side = crate::ui::state::TriSurfaceCutSide::CutTop;
                self.editor.tri_cut_surface_name_input = self
                    .active_triangulation
                    .and_then(|id| {
                        self.triangulations
                            .iter()
                            .find(|triangulation| triangulation.id == id)
                    })
                    .map(|triangulation| {
                        let stem = std::path::Path::new(&triangulation.name)
                            .file_stem()
                            .and_then(|value| value.to_str())
                            .unwrap_or(&triangulation.name);
                        format!("{stem}_surface_cut")
                    })
                    .unwrap_or_default();
                Ok(())
            }
            UiCommand::ExecuteCutTriangulationBySurface {
                target_id,
                reference_id,
                side,
                name,
            } => {
                let result = self.cut_triangulation_by_surface(target_id, reference_id, side, name);
                if result.is_ok() {
                    self.editor.tri_cut_surface_open = false;
                }
                result
            }
            UiCommand::OpenCutTopologyByPitShell => {
                self.editor.tri_cut_pitshell_open = true;
                self.editor.tri_cut_pitshell_topology_id = self.active_triangulation;
                self.editor.tri_cut_pitshell_pitshell_id = None;
                self.editor.tri_cut_pitshell_name_input = self
                    .active_triangulation
                    .and_then(|id| self.triangulations.iter().find(|t| t.id == id))
                    .map(|t| {
                        let stem = std::path::Path::new(&t.name)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or(&t.name);
                        format!("{stem}_cut")
                    })
                    .unwrap_or_default();
                Ok(())
            }
            UiCommand::ExecuteCutTopologyByPitShell {
                topology_id,
                pit_shell_id,
                name,
            } => {
                let result = self.cut_topology_by_pit_shell(topology_id, pit_shell_id, name);
                if result.is_ok() {
                    self.editor.tri_cut_pitshell_open = false;
                }
                result
            }
            UiCommand::OpenIncludeSolidInTopology => {
                self.editor.tri_include_solid_open = true;
                self.editor.tri_include_solid_topology_id = self.active_triangulation;
                self.editor.tri_include_solid_shape_id = None;
                self.editor.tri_include_solid_name_input = self
                    .active_triangulation
                    .and_then(|id| {
                        self.triangulations
                            .iter()
                            .find(|triangulation| triangulation.id == id)
                    })
                    .map(|triangulation| {
                        let stem = std::path::Path::new(&triangulation.name)
                            .file_stem()
                            .and_then(|value| value.to_str())
                            .unwrap_or(&triangulation.name);
                        format!("{stem}_with_shape")
                    })
                    .unwrap_or_default();
                Ok(())
            }
            UiCommand::ExecuteIncludeSolidInTopology {
                topology_id,
                shape_id,
                name,
            } => {
                let result = self.include_solid_in_topology(topology_id, shape_id, name);
                if result.is_ok() {
                    self.editor.tri_include_solid_open = false;
                }
                result
            }
            UiCommand::OpenContourTriangulation => {
                self.editor.tri_contour_open = true;
                self.editor.tri_contour_tri_id = self.active_triangulation;
                self.editor.tri_contour_project_index = self.workspace.active_index.unwrap_or(0);
                Ok(())
            }
            UiCommand::ExecuteContourTriangulation {
                tri_id,
                major_interval,
                minor_interval,
                major_color,
                minor_color,
                project_index,
                z_range,
            } => {
                let result = self.generate_contour_triangulation(
                    tri_id,
                    major_interval,
                    minor_interval,
                    major_color,
                    minor_color,
                    project_index,
                    z_range,
                );
                if result.is_ok() {
                    self.editor.tri_contour_open = false;
                }
                result
            }
            UiCommand::PreviewMoveDelta(delta) => {
                self.ensure_move_session_original();
                self.preview_move_delta(delta);
                self.invalidate_geometry();
                Ok(())
            }
            UiCommand::ApplyChamfer => {
                self.apply_chamfer();
                Ok(())
            }
            UiCommand::CancelChamfer => {
                self.cancel_chamfer();
                Ok(())
            }
            UiCommand::ApplyBezier => {
                self.apply_bezier();
                Ok(())
            }
            UiCommand::CancelBezier => {
                self.cancel_bezier();
                Ok(())
            }
            UiCommand::ApplyMoveDelta(delta) => {
                self.apply_move_delta(delta);
                Ok(())
            }
            UiCommand::CancelMoveDelta => {
                self.cancel_move_delta();
                Ok(())
            }
            UiCommand::ConfirmDeleteSelection => {
                if self.editor.active_tool == crate::ui::state::ActiveTool::Move {
                    self.cancel_move_delta();
                }
                self.delete_selection();
                Ok(())
            }
            UiCommand::Undo => {
                self.apply_history_step(true);
                Ok(())
            }
            UiCommand::Redo => {
                self.apply_history_step(false);
                Ok(())
            }
        }
    }
}
