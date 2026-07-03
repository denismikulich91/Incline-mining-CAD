use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::mpsc,
};

use anyhow::{Context, Result};
use rfd::FileDialog;

use crate::{
    app::{App, file_name},
    model::{
        LayerId,
        block_model::{BlockModelId, BlockModelSource},
        formats::{self, MeshFormat, bmf},
        pidb,
        triangulation::TriangulationId,
    },
    ui::state::{FileOperationDialog, FileOperationKind},
    userspace_log, userspace_warn,
};

/// Result of a background file-dialog. Each variant carries the path(s) chosen
/// along with whatever IDs or indices are needed to act on them.
#[derive(Debug)]
pub(crate) enum FileDialogAction {
    NewPidb(PathBuf),
    OpenPidb(Vec<PathBuf>),
    ImportDxfInto {
        project_runtime_id: u32,
        paths: Vec<PathBuf>,
    },
    /// (source_path, pidb_save_path) — source may be .dxf or .dgd.isis.
    ImportAsPidb(Vec<(PathBuf, PathBuf)>),
    ImportTriangulation(Vec<PathBuf>),
    BlockModelBmf(PathBuf),
    BlockModelBdf(PathBuf),
    SetBlockModelBdf {
        id: BlockModelId,
        path: PathBuf,
    },
    SetBlockModelSourceBdf {
        source: BlockModelSource,
        path: PathBuf,
    },
    OpenTriangulationFolder(PathBuf),
    ExportLayerDxf {
        project_runtime_id: u32,
        layer: LayerId,
        path: PathBuf,
    },
    ExportPidbDxf {
        project_runtime_id: u32,
        path: PathBuf,
    },
    ExportTriangulation {
        id: TriangulationId,
        path: PathBuf,
    },
    SaveTriangulationAs {
        id: TriangulationId,
        path: PathBuf,
    },
    SaveAndCloseTriangulationAs {
        id: TriangulationId,
        path: PathBuf,
    },
    SavePidbAs {
        project_runtime_id: u32,
        path: PathBuf,
    },
    DgdIsisSource(PathBuf),
    ImportDgdIsisAsPidb {
        source: PathBuf,
        dest: PathBuf,
    },
}

impl<'a> App<'a> {
    /// Spawn a background thread that runs a file dialog closure and sends the
    /// result back via a channel. The main thread polls in `poll_file_dialogs`.
    fn spawn_file_dialog<F>(&mut self, f: F)
    where
        F: FnOnce() -> Option<FileDialogAction> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        self.pending_file_dialogs.push(rx);
    }

    /// Drain completed file-dialog receivers and execute each resolved action.
    /// Called every frame from `about_to_wait`.
    pub(crate) fn poll_file_dialogs(&mut self) {
        let mut resolved: Vec<Option<FileDialogAction>> = Vec::new();
        self.pending_file_dialogs.retain(|rx| match rx.try_recv() {
            Ok(action) => {
                resolved.push(action);
                false
            }
            Err(mpsc::TryRecvError::Empty) => true,
            Err(mpsc::TryRecvError::Disconnected) => false,
        });
        for action in resolved.into_iter().flatten() {
            if let Err(err) = self.execute_file_dialog_action(action) {
                let msg = format!("{err:#}");
                userspace_warn!("File dialog action failed: {msg}");
            }
            self.redraw_requested = true;
        }
    }

    fn execute_file_dialog_action(&mut self, action: FileDialogAction) -> Result<()> {
        match action {
            FileDialogAction::NewPidb(path) => {
                let pidb = pidb::new_empty(Some(path.clone()));
                pidb::save(&path, &pidb)?;
                let display = path.display().to_string();
                let project = pidb::open_project(Some(path), pidb, false)?;
                self.set_active_project(project);
                userspace_log!("Created new PIDB: {display}");
                Ok(())
            }
            FileDialogAction::OpenPidb(paths) => {
                let count = paths.len();
                for path in &paths {
                    self.open_pidb_path(path)?;
                }
                userspace_log!("Opened {count} PIDB file(s)");
                Ok(())
            }
            FileDialogAction::ImportDxfInto {
                project_runtime_id,
                paths,
            } => {
                let project_index = self
                    .workspace
                    .project_index_for_runtime_id(project_runtime_id)
                    .context("The selected .pidb is no longer open")?;
                if self.workspace.active_index != Some(project_index) {
                    self.history.clear();
                    self.editor.selected_handles.clear();
                    self.editor.active_layer = None;
                }
                let mut total_added = 0usize;
                let mut first_new_layer: Option<LayerId> = None;
                let mut imported_layers: Vec<LayerId> = Vec::new();
                for path in &paths {
                    let project = self
                        .workspace
                        .projects
                        .get_mut(project_index)
                        .context("The selected .pidb is no longer open")?;
                    let existing: std::collections::HashSet<LayerId> = project
                        .pidb
                        .document
                        .layers()
                        .iter()
                        .map(|l| l.id)
                        .collect();
                    let added = pidb::import_dxf_into(&mut project.pidb, path)?;
                    project.dirty = true;
                    total_added += added;
                    let new_ids: Vec<LayerId> = project
                        .pidb
                        .document
                        .layers()
                        .iter()
                        .filter(|l| !existing.contains(&l.id))
                        .map(|l| l.id)
                        .collect();
                    for &id in &new_ids {
                        project.loaded_layers.insert(id);
                        first_new_layer.get_or_insert(id);
                        imported_layers.push(id);
                    }
                }
                self.workspace.set_active_index(project_index);
                if let Some(id) = first_new_layer {
                    self.editor.active_layer = Some(id);
                }
                for layer_id in imported_layers {
                    self.save_project_layer(project_index, layer_id)?;
                }
                self.invalidate_geometry();
                self.fit_view_to_extents();
                userspace_log!(
                    "Imported {} DXF(s) into project {}: {} object(s)",
                    paths.len(),
                    project_index,
                    total_added
                );
                Ok(())
            }
            FileDialogAction::ImportAsPidb(pairs) => {
                let count = pairs.len();
                for (source_path, pidb_path) in &pairs {
                    let is_isis = source_path
                        .to_string_lossy()
                        .to_ascii_lowercase()
                        .ends_with(".dgd.isis");
                    let mut pidb_data = if is_isis {
                        pidb::pidb_from_dgd_isis(source_path)?
                    } else {
                        pidb::pidb_from_dxf_path(source_path)?
                    };
                    pidb_data.metadata.name = pidb_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("Imported.pidb")
                        .to_string();
                    pidb::save(pidb_path, &pidb_data)?;
                    let project = pidb::open_project(Some(pidb_path.clone()), pidb_data, false)?;
                    self.set_active_project(project);
                    if let Some(idx) = self.workspace.active_index {
                        let layer_ids: Vec<LayerId> = self.workspace.projects[idx]
                            .pidb
                            .document
                            .layers()
                            .iter()
                            .map(|l| l.id)
                            .collect();
                        let project = &mut self.workspace.projects[idx];
                        for &id in &layer_ids {
                            project.loaded_layers.insert(id);
                        }
                        self.editor.active_layer = layer_ids.first().copied();
                    }
                    self.invalidate_geometry();
                    self.fit_view_to_extents();
                }
                userspace_log!("Imported {count} file(s) as .pidb");
                Ok(())
            }
            FileDialogAction::ImportTriangulation(paths) => {
                let count = paths.len();
                for path in &paths {
                    if !self.triangulation_files.contains(path) {
                        self.triangulation_files.push(path.clone());
                    }
                    self.open_triangulation_path(path)?;
                }
                userspace_log!("Imported {count} triangulation file(s) from picker");
                userspace_log!("Imported {count} triangulation file(s)");
                Ok(())
            }
            FileDialogAction::BlockModelBmf(path) => {
                if let Some(dialog) = self.editor.file_operation_dialog.as_mut()
                    && dialog.kind == FileOperationKind::ImportBlockModel
                {
                    dialog.bmf_path = Some(path.clone());
                    if dialog.bdf_path.is_none() {
                        dialog.bdf_path = bmf::same_stem_bdf_path(&path);
                    }
                }
                Ok(())
            }
            FileDialogAction::BlockModelBdf(path) => {
                if let Some(dialog) = self.editor.file_operation_dialog.as_mut()
                    && dialog.kind == FileOperationKind::ImportBlockModel
                {
                    dialog.bdf_path = Some(path.clone());
                    if dialog.bmf_path.is_none() {
                        dialog.bmf_path = bmf::same_stem_bmf_path(&path);
                    }
                }
                Ok(())
            }
            FileDialogAction::SetBlockModelBdf { id, path } => {
                self.set_block_model_definition(id, path)
            }
            FileDialogAction::SetBlockModelSourceBdf { source, path } => {
                self.set_block_model_source_definition(source, path)
            }
            FileDialogAction::OpenTriangulationFolder(dir) => {
                if !dir.is_dir() {
                    return Ok(());
                }
                let entries = Self::scan_triangulation_dir(&dir);
                if entries.is_empty() {
                    return Ok(());
                }
                let total_files = entries.len();
                let mut added_dirs = 0usize;
                if !entries.is_empty() {
                    if !self.triangulation_dirs.contains(&dir) {
                        self.triangulation_dirs.push(dir.clone());
                        added_dirs += 1;
                    }
                    self.triangulation_dir_entries.insert(dir.clone(), entries);
                }
                self.triangulation_dirs.sort();
                self.triangulation_dirs.dedup();
                self.persist_session();
                userspace_log!(
                    "Added triangulation folder {} with {total_files} file(s)",
                    dir.display()
                );
                userspace_log!(
                    "Added triangulation folder: {} ({} files across {added_dirs} folders)",
                    dir.display(),
                    total_files
                );
                Ok(())
            }
            FileDialogAction::ExportLayerDxf {
                project_runtime_id,
                layer,
                path,
            } => {
                let project_index = self
                    .workspace
                    .project_index_for_runtime_id(project_runtime_id)
                    .context("The selected .pidb is no longer open")?;
                let project = self
                    .workspace
                    .projects
                    .get(project_index)
                    .context("The selected .pidb is no longer open")?;
                pidb::export_layer_to_dxf(&project.pidb, layer, &path)?;
                userspace_log!(
                    "Exporting project {project_index} layer {:?} to DXF: {}",
                    layer,
                    path.display()
                );
                userspace_log!(
                    "Exported PIDB index {project_index} layer {:?} to DXF: {}",
                    layer,
                    path.display()
                );
                Ok(())
            }
            FileDialogAction::ExportPidbDxf {
                project_runtime_id,
                path,
            } => {
                let project_index = self
                    .workspace
                    .project_index_for_runtime_id(project_runtime_id)
                    .context("The selected .pidb is no longer open")?;
                let project = self
                    .workspace
                    .projects
                    .get(project_index)
                    .context("The selected .pidb is no longer open")?;
                pidb::export_to_dxf(&project.pidb, &path)?;
                userspace_log!(
                    "Exporting project {project_index} to DXF: {}",
                    path.display()
                );
                userspace_log!(
                    "Exported PIDB index {project_index} to DXF: {}",
                    path.display()
                );
                Ok(())
            }
            FileDialogAction::ExportTriangulation { id, path } => {
                MeshFormat::from_path(&path)
                    .context("Choose a filename ending in .00t, .obj, .stl, or .ply")?;
                if self
                    .triangulations
                    .iter()
                    .any(|other| other.id != id && other.path == path)
                    || self
                        .pending_triangulation_loads
                        .iter()
                        .any(|(p, _)| p == &path)
                {
                    anyhow::bail!(
                        "Another loaded triangulation already uses {}",
                        path.display()
                    );
                }
                let triangulation = self
                    .triangulations
                    .iter()
                    .find(|t| t.id == id)
                    .context("The selected triangulation is no longer loaded")?;
                formats::write_mesh(&triangulation.mesh, &path)?;
                userspace_log!(
                    "Exporting triangulation '{}' to {}",
                    triangulation.name,
                    path.display()
                );
                userspace_log!(
                    "Exported triangulation '{}' to {}",
                    triangulation.name,
                    path.display()
                );
                Ok(())
            }
            FileDialogAction::SaveTriangulationAs { id, path } => {
                self.commit_triangulation_save(id, path)?;
                Ok(())
            }
            FileDialogAction::SaveAndCloseTriangulationAs { id, path } => {
                self.commit_triangulation_save(id, path)?;
                self.editor.tri_close_unsaved = None;
                self.close_triangulation(id);
                Ok(())
            }
            FileDialogAction::SavePidbAs {
                project_runtime_id,
                path,
            } => {
                let project_index = self
                    .workspace
                    .project_index_for_runtime_id(project_runtime_id)
                    .context("The selected .pidb is no longer open")?;
                self.ensure_project_has_no_pending_text_edit(project_index)?;
                self.ensure_pidb_save_path_available(project_index, &path)?;
                let project = self
                    .workspace
                    .projects
                    .get_mut(project_index)
                    .context("No project at that index")?;
                let mut saved_pidb = project.pidb.clone();
                saved_pidb.metadata.name = file_name(&path);
                pidb::save(&path, &saved_pidb)?;
                project.pidb = saved_pidb;
                project.path = Some(path.clone());
                project.dirty = false;
                project.invalidate_dirty_layers();
                userspace_log!("Saved PIDB index {} as {}", project_index, path.display());
                userspace_log!("Saved PIDB index {} as: {}", project_index, path.display());
                self.persist_session();
                Ok(())
            }
            FileDialogAction::DgdIsisSource(path) => {
                if let Some(dialog) = self.editor.file_operation_dialog.as_mut()
                    && dialog.kind == FileOperationKind::ImportPidb
                {
                    dialog.source_path = Some(path);
                }
                Ok(())
            }
            FileDialogAction::ImportDgdIsisAsPidb { source, dest } => {
                let pidb_data = pidb::pidb_from_dgd_isis(&source)?;
                pidb::save(&dest, &pidb_data)?;
                let display = dest.display().to_string();
                let project = pidb::open_project(Some(dest), pidb_data, false)?;
                self.set_active_project(project);
                if let Some(idx) = self.workspace.active_index {
                    let layer_ids: Vec<LayerId> = self.workspace.projects[idx]
                        .pidb
                        .document
                        .layers()
                        .iter()
                        .map(|l| l.id)
                        .collect();
                    let project = &mut self.workspace.projects[idx];
                    for &id in &layer_ids {
                        project.loaded_layers.insert(id);
                    }
                    self.editor.active_layer = layer_ids.first().copied();
                }
                self.invalidate_geometry();
                self.fit_view_to_extents();
                userspace_log!("Imported dgd.isis as: {display}");
                Ok(())
            }
        }
    }

    /// Save the mesh for `id` to `path`, update triangulation metadata, and
    /// register the path in the session. Shared by SaveTriangulationAs and
    /// SaveAndCloseTriangulationAs actions.
    fn commit_triangulation_save(&mut self, id: TriangulationId, path: PathBuf) -> Result<()> {
        if self
            .triangulations
            .iter()
            .any(|other| other.id != id && other.path == path)
            || self
                .pending_triangulation_loads
                .iter()
                .any(|(p, _)| p == &path)
        {
            anyhow::bail!(
                "Another loaded triangulation already uses {}",
                path.display()
            );
        }
        MeshFormat::from_path(&path)
            .context("Choose a filename ending in .00t, .obj, .stl, or .ply")?;
        let triangulation = self
            .triangulations
            .iter()
            .find(|t| t.id == id)
            .context("The triangulation is no longer loaded")?;
        formats::write_mesh(&triangulation.mesh, &path)?;
        let triangulation = self.triangulations.iter_mut().find(|t| t.id == id).unwrap();
        let saved_name = triangulation.name.clone();
        triangulation.path = path.clone();
        triangulation.name = file_name(&path);
        triangulation.is_saved = true;
        if !self.triangulation_files.contains(&path) {
            self.triangulation_files.push(path.clone());
        }
        self.triangulation_excluded_paths.remove(&path);
        userspace_log!("Saved triangulation '{}' to {}", saved_name, path.display());
        self.persist_session();
        userspace_log!("Saved triangulation to {}", path.display());
        Ok(())
    }

    // ── Dialog spawners (non-blocking) ──────────────────────────────────────

    pub(crate) fn open_file_operation_dialog(&mut self, kind: FileOperationKind) {
        let mut dialog = FileOperationDialog::new(kind);
        dialog.project_index = self
            .workspace
            .active_index
            .or_else(|| (!self.workspace.projects.is_empty()).then_some(0));
        dialog.layer = self
            .workspace
            .projects
            .iter()
            .enumerate()
            .flat_map(|(project_index, project)| {
                project
                    .loaded_layers
                    .iter()
                    .copied()
                    .map(move |layer| (project_index, layer))
            })
            .next();
        dialog.triangulation = self.active_triangulation.or_else(|| {
            self.triangulations
                .first()
                .map(|triangulation| triangulation.id)
        });
        self.editor.file_operation_dialog = Some(dialog);
        userspace_log!("Opened file operation dialog: {:?}", kind);
    }

    pub(crate) fn choose_dgd_isis_source(&mut self) {
        self.spawn_file_dialog(|| {
            let path = FileDialog::new()
                .add_filter("Vulcan design database", &["isis"])
                .pick_file()?;
            Some(FileDialogAction::DgdIsisSource(path))
        });
    }

    pub(crate) fn choose_new_pidb(&mut self) {
        self.spawn_file_dialog(|| {
            let path = FileDialog::new()
                .add_filter("ProInspector database", &["pidb"])
                .set_file_name("new_project.pidb")
                .save_file()?;
            Some(FileDialogAction::NewPidb(path))
        });
    }

    pub(crate) fn choose_open_pidb(&mut self) {
        self.spawn_file_dialog(|| {
            let paths = FileDialog::new()
                .add_filter("ProInspector database", &["pidb"])
                .pick_files()?;
            Some(FileDialogAction::OpenPidb(paths))
        });
    }

    pub(crate) fn open_pidb_path(&mut self, path: &Path) -> Result<()> {
        let pidb = pidb::load(path)?;
        let project = pidb::open_project(Some(path.to_path_buf()), pidb, false)?;
        self.set_active_project(project);
        userspace_log!("Opened PIDB: {}", path.display());
        Ok(())
    }

    pub(crate) fn choose_import_dxf_into(&mut self, project_index: usize) {
        let Some(project_runtime_id) = self
            .workspace
            .projects
            .get(project_index)
            .map(|project| project.runtime_id)
        else {
            return;
        };
        self.spawn_file_dialog(move || {
            let paths = FileDialog::new().add_filter("DXF", &["dxf"]).pick_files()?;
            Some(FileDialogAction::ImportDxfInto {
                project_runtime_id,
                paths,
            })
        });
    }

    pub(crate) fn choose_import_as_pidb(&mut self) {
        self.spawn_file_dialog(|| {
            #[cfg(target_os = "macos")]
            let isis_ext: &[&str] = &["isis"];
            #[cfg(not(target_os = "macos"))]
            let isis_ext: &[&str] = &["dgd.isis"];

            let source_paths = FileDialog::new()
                .add_filter(
                    "Supported files",
                    &["dxf"].iter().chain(isis_ext).copied().collect::<Vec<_>>(),
                )
                .add_filter("AutoCAD DXF", &["dxf"])
                .add_filter("Vulcan design database (dgd.isis)", isis_ext)
                .pick_files()?;
            let mut pairs = Vec::new();
            for source_path in source_paths {
                let raw = source_path.to_string_lossy();
                let raw_lower = raw.to_ascii_lowercase();
                let is_dgd_isis = raw_lower.ends_with(".dgd.isis");
                let is_dxf = raw_lower.ends_with(".dxf");
                if !is_dgd_isis && !is_dxf {
                    continue;
                }
                let stem = if is_dgd_isis {
                    let filename = source_path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("imported.dgd.isis");
                    filename[..filename.len().saturating_sub(".dgd.isis".len())].to_owned()
                } else {
                    source_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("imported")
                        .to_owned()
                };
                let default_name = format!("{stem}.pidb");
                let Some(pidb_path) = FileDialog::new()
                    .add_filter("ProInspector database", &["pidb"])
                    .set_file_name(&default_name)
                    .save_file()
                else {
                    continue;
                };
                pairs.push((source_path, pidb_path));
            }
            if pairs.is_empty() {
                None
            } else {
                Some(FileDialogAction::ImportAsPidb(pairs))
            }
        });
    }

    pub(crate) fn confirm_import_dgd_isis(&mut self) {
        let Some(source) = self
            .editor
            .file_operation_dialog
            .as_ref()
            .and_then(|d| d.source_path.clone())
        else {
            return;
        };
        let stem = source
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_suffix(".dgd"))
            .unwrap_or("imported")
            .to_owned();
        let default_name = format!("{stem}.pidb");
        self.spawn_file_dialog(move || {
            let dest = FileDialog::new()
                .add_filter("ProInspector database", &["pidb"])
                .set_file_name(&default_name)
                .save_file()?;
            Some(FileDialogAction::ImportDgdIsisAsPidb { source, dest })
        });
    }

    pub(crate) fn choose_import_triangulation(&mut self) {
        self.spawn_file_dialog(|| {
            let paths = FileDialog::new()
                .add_filter("All supported", &["00t", "obj", "stl", "ply"])
                .add_filter("Vulcan triangulation", &["00t"])
                .add_filter("Wavefront OBJ", &["obj"])
                .add_filter("STL", &["stl"])
                .add_filter("PLY", &["ply"])
                .pick_files()?;
            Some(FileDialogAction::ImportTriangulation(paths))
        });
    }

    pub(crate) fn choose_block_model_bmf(&mut self) {
        self.spawn_file_dialog(|| {
            let path = FileDialog::new()
                .add_filter("Vulcan block model", &["bmf"])
                .pick_file()?;
            Some(FileDialogAction::BlockModelBmf(path))
        });
    }

    pub(crate) fn choose_block_model_bdf(&mut self) {
        self.spawn_file_dialog(|| {
            let path = FileDialog::new()
                .add_filter("Vulcan block definition", &["bdf"])
                .pick_file()?;
            Some(FileDialogAction::BlockModelBdf(path))
        });
    }

    pub(crate) fn choose_set_block_model_bdf(&mut self, id: BlockModelId) {
        self.spawn_file_dialog(move || {
            let path = FileDialog::new()
                .add_filter("Vulcan block definition", &["bdf"])
                .pick_file()?;
            Some(FileDialogAction::SetBlockModelBdf { id, path })
        });
    }

    pub(crate) fn choose_set_block_model_source_bdf(&mut self, source: BlockModelSource) {
        self.spawn_file_dialog(move || {
            let path = FileDialog::new()
                .add_filter("Vulcan block definition", &["bdf"])
                .pick_file()?;
            Some(FileDialogAction::SetBlockModelSourceBdf { source, path })
        });
    }

    pub(crate) fn choose_open_triangulation_folder(&mut self) {
        self.spawn_file_dialog(|| {
            let dir = FileDialog::new().pick_folder()?;
            Some(FileDialogAction::OpenTriangulationFolder(dir))
        });
    }

    pub(crate) fn choose_export_pidb_dxf(&mut self, project_index: usize) {
        let Some(project_runtime_id) = self
            .workspace
            .projects
            .get(project_index)
            .map(|project| project.runtime_id)
        else {
            return;
        };
        self.spawn_file_dialog(move || {
            let path = FileDialog::new()
                .add_filter("DXF", &["dxf"])
                .set_file_name("project.dxf")
                .save_file()?;
            Some(FileDialogAction::ExportPidbDxf {
                project_runtime_id,
                path,
            })
        });
    }

    pub(crate) fn choose_export_layer_dxf(&mut self, project_index: usize, layer: LayerId) {
        let Some(project_runtime_id) = self
            .workspace
            .projects
            .get(project_index)
            .map(|project| project.runtime_id)
        else {
            return;
        };
        self.spawn_file_dialog(move || {
            let path = FileDialog::new()
                .add_filter("DXF", &["dxf"])
                .set_file_name("layer.dxf")
                .save_file()?;
            Some(FileDialogAction::ExportLayerDxf {
                project_runtime_id,
                layer,
                path,
            })
        });
    }

    pub(crate) fn choose_export_triangulation(&mut self, id: TriangulationId) {
        let stem = self
            .triangulations
            .iter()
            .find(|t| t.id == id)
            .and_then(|t| t.path.file_stem().and_then(|s| s.to_str()))
            .unwrap_or("triangulation")
            .to_owned();
        self.spawn_file_dialog(move || {
            let default_name = format!("{stem}.00t");
            let path = FileDialog::new()
                .add_filter("Vulcan triangulation", &["00t"])
                .add_filter("Wavefront OBJ", &["obj"])
                .add_filter("STL", &["stl"])
                .add_filter("PLY", &["ply"])
                .set_file_name(default_name)
                .save_file()?;
            Some(FileDialogAction::ExportTriangulation { id, path })
        });
    }

    pub(crate) fn spawn_save_triangulation_dialog(&mut self, id: TriangulationId) {
        let stem = self
            .triangulations
            .iter()
            .find(|t| t.id == id)
            .and_then(|t| {
                std::path::Path::new(&t.name)
                    .file_stem()
                    .and_then(|s| s.to_str())
            })
            .unwrap_or("triangulation")
            .to_owned();
        self.spawn_file_dialog(move || {
            let default_name = format!("{stem}.00t");
            let path = FileDialog::new()
                .add_filter("Vulcan triangulation", &["00t"])
                .add_filter("Wavefront OBJ", &["obj"])
                .add_filter("STL", &["stl"])
                .add_filter("PLY", &["ply"])
                .set_file_name(&default_name)
                .save_file()?;
            Some(FileDialogAction::SaveTriangulationAs { id, path })
        });
    }

    pub(crate) fn spawn_save_and_close_triangulation_dialog(&mut self, id: TriangulationId) {
        let stem = self
            .triangulations
            .iter()
            .find(|t| t.id == id)
            .and_then(|t| {
                std::path::Path::new(&t.name)
                    .file_stem()
                    .and_then(|s| s.to_str())
            })
            .unwrap_or("triangulation")
            .to_owned();
        self.spawn_file_dialog(move || {
            let default_name = format!("{stem}.00t");
            let path = FileDialog::new()
                .add_filter("Vulcan triangulation", &["00t"])
                .add_filter("Wavefront OBJ", &["obj"])
                .add_filter("STL", &["stl"])
                .add_filter("PLY", &["ply"])
                .set_file_name(&default_name)
                .save_file()?;
            Some(FileDialogAction::SaveAndCloseTriangulationAs { id, path })
        });
    }

    pub(crate) fn spawn_save_pidb_as_dialog(&mut self, project_index: usize) {
        if self.workspace.active_index == Some(project_index) && self.has_pending_move_delta() {
            self.commit_pending_move();
        }
        let Some(project_runtime_id) = self
            .workspace
            .projects
            .get(project_index)
            .map(|project| project.runtime_id)
        else {
            return;
        };
        self.spawn_file_dialog(move || {
            let path = FileDialog::new()
                .add_filter("ProInspector database", &["pidb"])
                .set_file_name("project.pidb")
                .save_file()?;
            Some(FileDialogAction::SavePidbAs {
                project_runtime_id,
                path,
            })
        });
    }

    // ── Synchronous save helpers (used in exit / internal chains) ───────────

    /// Save a triangulation synchronously; returns whether the user confirmed.
    /// Used only by the save-and-exit chain — menu-triggered saves use the
    /// spawned dialog variants above.
    pub(crate) fn save_triangulation_as(&mut self, id: TriangulationId) -> Result<bool> {
        let triangulation = self
            .triangulations
            .iter()
            .find(|t| t.id == id)
            .context("The triangulation is no longer loaded")?;
        let default_name = {
            let stem = std::path::Path::new(&triangulation.name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&triangulation.name);
            format!("{stem}.00t")
        };
        let Some(path) = FileDialog::new()
            .add_filter("Vulcan triangulation", &["00t"])
            .add_filter("Wavefront OBJ", &["obj"])
            .add_filter("STL", &["stl"])
            .add_filter("PLY", &["ply"])
            .set_file_name(&default_name)
            .save_file()
        else {
            return Ok(false);
        };
        self.commit_triangulation_save(id, path)?;
        Ok(true)
    }

    pub(crate) fn reveal_pidb(&mut self, index: usize) -> Result<()> {
        let project = self
            .workspace
            .projects
            .get(index)
            .context("No project at that index")?;
        let path = project
            .path
            .as_deref()
            .context("The .pidb has not been saved yet")?;

        #[cfg(target_os = "windows")]
        let mut command = {
            let mut command = Command::new("explorer");
            command.arg(format!("/select,{}", path.display()));
            command
        };
        #[cfg(target_os = "macos")]
        let mut command = {
            let mut command = Command::new("open");
            command.arg("-R").arg(path);
            command
        };
        #[cfg(all(unix, not(target_os = "macos")))]
        let mut command = {
            let mut command = Command::new("xdg-open");
            command.arg(path.parent().unwrap_or_else(|| Path::new(".")));
            command.env("LANG", "C.UTF-8").env("LC_ALL", "C.UTF-8");
            command
        };

        command
            .spawn()
            .context("Could not open the system file explorer")?;
        userspace_log!("Revealed PIDB path in explorer: {}", path.display());
        userspace_log!("Revealed PIDB in file explorer: {}", path.display());
        Ok(())
    }

    pub(crate) fn reveal_triangulation(&mut self, id: TriangulationId) -> Result<()> {
        let triangulation = self
            .triangulations
            .iter()
            .find(|triangulation| triangulation.id == id && triangulation.is_saved)
            .context("The triangulation has not been saved")?;
        let path = &triangulation.path;

        #[cfg(target_os = "windows")]
        let mut command = {
            let mut command = Command::new("explorer");
            command.arg(format!("/select,{}", path.display()));
            command
        };
        #[cfg(target_os = "macos")]
        let mut command = {
            let mut command = Command::new("open");
            command.arg("-R").arg(path);
            command
        };
        #[cfg(all(unix, not(target_os = "macos")))]
        let mut command = {
            let mut command = Command::new("xdg-open");
            command.arg(path.parent().unwrap_or_else(|| Path::new(".")));
            command.env("LANG", "C.UTF-8").env("LC_ALL", "C.UTF-8");
            command
        };

        command
            .spawn()
            .context("Could not open the system file explorer")?;
        userspace_log!(
            "Revealed triangulation path in explorer: {}",
            path.display()
        );
        userspace_log!(
            "Revealed triangulation '{}' in file explorer",
            triangulation.name
        );
        Ok(())
    }

    pub(crate) fn reveal_block_model(&mut self, id: BlockModelId) -> Result<()> {
        let block_model = self
            .block_models
            .iter()
            .find(|model| model.id == id)
            .context("The block model is no longer loaded")?;
        let path = &block_model.source.bmf_path;

        #[cfg(target_os = "windows")]
        let mut command = {
            let mut command = Command::new("explorer");
            command.arg(format!("/select,{}", path.display()));
            command
        };
        #[cfg(target_os = "macos")]
        let mut command = {
            let mut command = Command::new("open");
            command.arg("-R").arg(path);
            command
        };
        #[cfg(all(unix, not(target_os = "macos")))]
        let mut command = {
            let mut command = Command::new("xdg-open");
            command.arg(path.parent().unwrap_or_else(|| Path::new(".")));
            command.env("LANG", "C.UTF-8").env("LC_ALL", "C.UTF-8");
            command
        };

        command
            .spawn()
            .context("Could not open the system file explorer")?;
        userspace_log!("Revealed block model in file explorer: {}", path.display());
        Ok(())
    }

    pub(crate) fn save_all_dirty_projects(&mut self) -> Result<bool> {
        if self.has_pending_move_delta() {
            self.commit_pending_move();
        }
        let dirty: Vec<usize> = self
            .workspace
            .projects
            .iter()
            .enumerate()
            .filter_map(|(index, project)| {
                (project.dirty || !project.dirty_layers().is_empty()).then_some(index)
            })
            .collect();
        let mut saved = 0;
        for index in dirty {
            self.save_named_project(index)?;
            if self
                .workspace
                .projects
                .get(index)
                .is_some_and(|project| project.dirty)
            {
                return Ok(false);
            }
            saved += 1;
        }
        userspace_log!("Saved {saved} dirty project(s)");
        userspace_log!("Saved {saved} dirty project(s)");
        Ok(true)
    }

    pub(crate) fn request_exit(&mut self) -> Result<()> {
        let has_unsaved_changes = self.workspace.projects.iter().any(|project| project.dirty)
            || self.triangulations.iter().any(|tri| !tri.is_saved);
        if !has_unsaved_changes {
            self.persist_session();
            self.close_requested = true;
            userspace_log!("Exit requested with no unsaved changes");
            userspace_log!("User requested exit (no unsaved changes)");
        } else {
            self.editor.exit_confirm_open = true;
            userspace_log!("User requested exit (unsaved changes present)");
        }
        Ok(())
    }

    pub(crate) fn save_and_exit(&mut self) -> Result<()> {
        if self.save_all_dirty_projects()? && self.save_all_unsaved_triangulations()? {
            self.editor.exit_confirm_open = false;
            self.persist_session();
            self.close_requested = true;
        }
        Ok(())
    }

    fn save_all_unsaved_triangulations(&mut self) -> Result<bool> {
        let unsaved: Vec<TriangulationId> = self
            .triangulations
            .iter()
            .filter(|tri| !tri.is_saved)
            .map(|tri| tri.id)
            .collect();
        for id in unsaved {
            if !self.save_triangulation_as(id)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(crate) fn exit_without_saving(&mut self) {
        self.editor.exit_confirm_open = false;
        self.persist_session();
        self.close_requested = true;
        userspace_log!("User chose to exit without saving");
    }

    pub(crate) fn save_named_project(&mut self, index: usize) -> Result<()> {
        if self.workspace.active_index == Some(index) && self.has_pending_move_delta() {
            self.commit_pending_move();
        }
        self.ensure_project_has_no_pending_text_edit(index)?;
        let has_path = self
            .workspace
            .projects
            .get(index)
            .and_then(|p| p.path.as_ref())
            .is_some();
        if !has_path {
            return self.save_named_project_as(index);
        }
        let project = &mut self.workspace.projects[index];
        let path = project.path.clone().unwrap();
        pidb::save(&path, &project.pidb)?;
        project.dirty = false;
        project.invalidate_dirty_layers();
        userspace_log!("Saved PIDB index {} to {}", index, path.display());
        userspace_log!("Saved PIDB index {}: {}", index, path.display());
        self.persist_session();
        Ok(())
    }

    pub(crate) fn save_named_project_as(&mut self, index: usize) -> Result<()> {
        if self.workspace.active_index == Some(index) && self.has_pending_move_delta() {
            self.commit_pending_move();
        }
        self.ensure_project_has_no_pending_text_edit(index)?;
        let Some(path) = FileDialog::new()
            .add_filter("ProInspector database", &["pidb"])
            .set_file_name("project.pidb")
            .save_file()
        else {
            return Ok(());
        };
        self.ensure_pidb_save_path_available(index, &path)?;
        let project = self
            .workspace
            .projects
            .get_mut(index)
            .context("No project at that index")?;
        let mut saved_pidb = project.pidb.clone();
        saved_pidb.metadata.name = file_name(&path);
        pidb::save(&path, &saved_pidb)?;
        project.pidb = saved_pidb;
        project.path = Some(path.clone());
        project.dirty = false;
        project.invalidate_dirty_layers();
        userspace_log!("Saved PIDB index {} as {}", index, path.display());
        userspace_log!("Saved PIDB index {} as: {}", index, path.display());
        self.persist_session();
        Ok(())
    }

    fn ensure_pidb_save_path_available(&self, project_index: usize, path: &Path) -> Result<()> {
        if self
            .workspace
            .projects
            .iter()
            .enumerate()
            .any(|(index, project)| index != project_index && project.path.as_deref() == Some(path))
        {
            anyhow::bail!("Another open PIDB already uses {}", path.display());
        }
        Ok(())
    }

    pub(crate) fn save_project_layer(&mut self, index: usize, layer_id: LayerId) -> Result<()> {
        if self.workspace.active_index == Some(index) && self.has_pending_move_delta() {
            self.commit_pending_move();
        }
        self.ensure_project_has_no_pending_text_edit(index)?;
        let Some(path) = self
            .workspace
            .projects
            .get(index)
            .and_then(|project| project.path.clone())
        else {
            return self.save_named_project_as(index);
        };
        let project = self
            .workspace
            .projects
            .get_mut(index)
            .context("No project at that index")?;
        pidb::save_layer(&path, &project.pidb, layer_id)?;
        project.dirty = pidb::differs_from_disk(&project.pidb, &path);
        project.invalidate_dirty_layers();
        userspace_log!("Saved layer {:?} in project {}", layer_id, index);
        userspace_log!("Saved layer {:?} in PIDB index {}", layer_id, index);
        self.persist_session();
        Ok(())
    }

    fn ensure_project_has_no_pending_text_edit(&self, project_index: usize) -> Result<()> {
        if !self.editor.text_editing_enabled {
            return Ok(());
        }
        let editing_project = self
            .editor
            .editing_labels_id
            .and_then(|object_id| self.workspace.project_index_for_object(object_id))
            .or(self.workspace.active_index);
        if editing_project == Some(project_index) {
            anyhow::bail!("Apply or discard the current text edit before saving this PIDB");
        }
        Ok(())
    }

    pub(crate) fn refresh_active_project_dirty(&mut self) {
        let Some(project) = self.workspace.active_project_mut() else {
            return;
        };
        let Some(path) = project.path.as_ref() else {
            project.dirty = true;
            project.invalidate_dirty_layers();
            return;
        };
        project.dirty = pidb::differs_from_disk(&project.pidb, path);
        project.invalidate_dirty_layers();
    }

    pub(crate) fn request_close_project(&mut self, index: usize) {
        let Some(project) = self.workspace.projects.get(index) else {
            return;
        };
        if project.dirty {
            self.editor.pending_close_project = Some(index);
        } else if let Err(error) = self.close_project(index) {
            userspace_warn!("Could not close PIDB: {error:#}");
        }
    }

    pub(crate) fn save_and_close_project(&mut self, index: usize) -> Result<()> {
        self.save_named_project(index)?;
        if self
            .workspace
            .projects
            .get(index)
            .is_some_and(|project| project.dirty)
        {
            return Ok(());
        }
        self.close_project(index)
    }

    pub(crate) fn close_project(&mut self, index: usize) -> Result<()> {
        if index >= self.workspace.projects.len() {
            return Ok(());
        }
        self.editor.pending_close_project = None;
        self.workspace.projects.remove(index);
        match self.workspace.active_index {
            Some(i) if i == index => {
                self.workspace.active_index = if !self.workspace.projects.is_empty() {
                    Some(
                        index
                            .saturating_sub(1)
                            .min(self.workspace.projects.len() - 1),
                    )
                } else {
                    None
                };
                self.clear_editor_transient_state();
            }
            Some(i) if i > index => {
                self.workspace.active_index = Some(i - 1);
            }
            _ => {}
        }
        self.persist_session();
        userspace_log!("Closed PIDB index {}", index);
        userspace_log!("Closed PIDB index {}", index);
        self.invalidate_geometry();
        Ok(())
    }
}
