use std::{
    collections::BTreeMap,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    process::Command,
    sync::mpsc,
    task::{Context as TaskContext, Poll, Waker},
};

use anyhow::{Context, Result};
use rfd::AsyncFileDialog;

use crate::{
    app::{App, file_name},
    model::{
        LayerId,
        block_model::{BlockModelId, BlockModelSource},
        formats::{self, MeshFormat, bmf},
        pidb::{self, OpenProject},
        triangulation::TriangulationId,
    },
    ui::state::{DataMenu, TriSurfaceType},
    userspace_log, userspace_warn,
};

fn is_duf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("duf"))
}

fn is_dxf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("dxf"))
}

fn is_dgd_isis_path(path: &Path) -> bool {
    path.to_string_lossy()
        .to_ascii_lowercase()
        .ends_with(".dgd.isis")
}

fn project_has_unsaved_changes(project: &OpenProject) -> bool {
    project.has_unsaved_changes()
}

fn mesh_format_name_and_extension(format: MeshFormat) -> (&'static str, &'static str) {
    match format {
        MeshFormat::Obj => ("Wavefront OBJ", "obj"),
        MeshFormat::Stl => ("STL", "stl"),
        MeshFormat::Ply => ("PLY", "ply"),
        MeshFormat::T00 => ("Vulcan triangulation", "00t"),
    }
}

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
    ImportPointCloud(Vec<PathBuf>),
    SetImportSourcePaths {
        kind: DataMenu,
        paths: Vec<PathBuf>,
    },
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
    ExportPidbCopy {
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
}

pub(crate) type PendingFileDialog = Pin<Box<dyn Future<Output = Option<FileDialogAction>>>>;

/// A triangulation save/export running on a background thread. The worker
/// streams completion fractions over `progress_rx` (shown in the status bar)
/// and delivers the final outcome over `result_rx`; `poll_saves` drains both
/// each frame.
pub(crate) struct PendingSave {
    kind: PendingSaveKind,
    path: PathBuf,
    progress_rx: mpsc::Receiver<f32>,
    result_rx: mpsc::Receiver<Result<()>>,
    latest_progress: f32,
}

enum PendingSaveKind {
    /// Save/Save-As: on success update the triangulation's path/name/is_saved
    /// and register the file in the session; optionally close it afterwards.
    Save {
        id: TriangulationId,
        close_after: bool,
    },
    /// Export a copy: leave the open triangulation's metadata unchanged.
    Export { name: String },
}

impl<'a> App<'a> {
    /// Register an async file dialog that was created on the main thread. The
    /// app polls completion in `poll_file_dialogs`.
    fn spawn_file_dialog<F>(&mut self, future: F)
    where
        F: Future<Output = Option<FileDialogAction>> + 'static,
    {
        self.pending_file_dialogs.push(Box::pin(future));
    }

    /// Drain completed file-dialog futures and execute each resolved action.
    /// Called every frame from `about_to_wait`.
    pub(crate) fn poll_file_dialogs(&mut self) {
        let mut resolved: Vec<Option<FileDialogAction>> = Vec::new();
        let waker = Waker::noop();
        let mut cx = TaskContext::from_waker(waker);
        self.pending_file_dialogs
            .retain_mut(|future| match future.as_mut().poll(&mut cx) {
                Poll::Ready(action) => {
                    resolved.push(action);
                    false
                }
                Poll::Pending => true,
            });
        if self.exit_after_pending_saves && resolved.iter().any(Option::is_none) {
            self.cancel_exit_request();
        }
        for action in resolved.into_iter().flatten() {
            if let Err(err) = self.execute_file_dialog_action(action) {
                let msg = format!("{err:#}");
                userspace_warn!("File dialog action failed: {msg}");
                if self.exit_after_pending_saves {
                    self.cancel_exit_request();
                }
            }
            self.redraw_requested = true;
        }
        self.try_finish_deferred_exit();
    }

    pub(super) fn execute_file_dialog_action(&mut self, action: FileDialogAction) -> Result<()> {
        match action {
            FileDialogAction::NewPidb(path) => {
                let pidb = pidb::new_empty(Some(path.clone()));
                pidb::save(&path, &pidb)?;
                let display = path.display().to_string();
                let project = pidb::open_project(Some(path), pidb)?;
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
                    let is_isis = is_dgd_isis_path(source_path);
                    let is_duf = is_duf_path(source_path);
                    let mut pidb_data = if is_duf {
                        pidb::pidb_from_duf(source_path)?
                    } else if is_isis {
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
                    let project = pidb::open_project(Some(pidb_path.clone()), pidb_data)?;
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
                    if is_duf {
                        self.import_duf_triangulations(source_path)?;
                    }
                }
                userspace_log!("Imported {count} file(s) as .pidb");
                Ok(())
            }
            FileDialogAction::ImportTriangulation(paths) => {
                let count = paths.len();
                for path in &paths {
                    if is_duf_path(path) {
                        self.import_duf_triangulations(path)?;
                    } else if !self.triangulation_files.contains(path) {
                        self.triangulation_files.push(path.clone());
                        self.open_triangulation_path(path)?;
                    } else {
                        self.open_triangulation_path(path)?;
                    }
                }
                userspace_log!("Imported {count} triangulation file(s)");
                Ok(())
            }
            FileDialogAction::ImportPointCloud(paths) => {
                for path in &paths {
                    self.import_point_cloud_path(path)?;
                }
                Ok(())
            }
            FileDialogAction::SetImportSourcePaths { kind, paths } => {
                self.editor.import_source_menu = kind;
                self.editor.import_source_paths = paths;
                Ok(())
            }
            FileDialogAction::BlockModelBmf(path) => {
                self.editor.import_bmf_path = Some(path.clone());
                if self.editor.import_bdf_path.is_none() {
                    self.editor.import_bdf_path = bmf::same_stem_bdf_path(&path);
                }
                Ok(())
            }
            FileDialogAction::BlockModelBdf(path) => {
                self.editor.import_bdf_path = Some(path.clone());
                if self.editor.import_bmf_path.is_none() {
                    self.editor.import_bmf_path = bmf::same_stem_bmf_path(&path);
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
                    "Exported PIDB index {project_index} to DXF: {}",
                    path.display()
                );
                Ok(())
            }
            FileDialogAction::ExportPidbCopy {
                project_runtime_id,
                path,
            } => {
                let project_index = self
                    .workspace
                    .project_index_for_runtime_id(project_runtime_id)
                    .context("The selected .pidb is no longer open")?;
                self.ensure_pidb_save_path_available(project_index, &path)?;
                let project = self
                    .workspace
                    .projects
                    .get(project_index)
                    .context("The selected .pidb is no longer open")?;
                let mut exported = project.pidb.clone();
                exported.metadata.name = file_name(&path);
                pidb::save(&path, &exported)?;
                userspace_log!(
                    "Exported PIDB index {project_index} copy to {}",
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
                if self.pending_saves.iter().any(|save| save.path == path) {
                    anyhow::bail!("A save to {} is already in progress", path.display());
                }
                let triangulation = self
                    .triangulations
                    .iter()
                    .find(|t| t.id == id)
                    .context("The selected triangulation is no longer loaded")?;
                let mesh = std::sync::Arc::clone(&triangulation.mesh);
                let name = triangulation.name.clone();
                userspace_log!("Exporting triangulation '{}' to {}", name, path.display());
                self.spawn_triangulation_write(PendingSaveKind::Export { name }, mesh, path);
                Ok(())
            }
            FileDialogAction::SaveTriangulationAs { id, path } => {
                self.commit_triangulation_save(id, path, false)?;
                Ok(())
            }
            FileDialogAction::SaveAndCloseTriangulationAs { id, path } => {
                // The close happens in poll_saves once the background save
                // succeeds; dismiss the unsaved-close prompt immediately.
                self.commit_triangulation_save(id, path, true)?;
                self.editor.tri_close_unsaved = None;
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
                project.invalidate_disk_snapshot();
                userspace_log!("Saved PIDB index {} as: {}", project_index, path.display());
                self.persist_session();
                Ok(())
            }
        }
    }

    fn import_duf_triangulations(&mut self, path: &Path) -> Result<()> {
        let duf = formats::duf::read_duf(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        if duf.polyfaces.is_empty() {
            userspace_log!("{} contains no supported DUF mesh entities", path.display());
            return Ok(());
        }

        let stem = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("DUF");
        let design_skipped = duf
            .polylines
            .len()
            .saturating_add(duf.points.len())
            .max(duf.skipped.unsupported_design_entities);
        if design_skipped > 0 {
            userspace_log!(
                "Ignored {design_skipped} DUF design entit{} while importing {} as triangulation",
                if design_skipped == 1 { "y" } else { "ies" },
                path.display()
            );
        }

        let mut loaded = 0usize;
        let split_unknown_layer_meshes = duf
            .polyfaces
            .iter()
            .all(|polyface| polyface.layer_name == "DUF Mesh");
        if split_unknown_layer_meshes {
            let width = duf.polyfaces.len().max(1).ilog10() as usize + 1;
            for (index, polyface) in duf.polyfaces.iter().enumerate() {
                if polyface.vertices.is_empty() || polyface.faces.is_empty() {
                    continue;
                }
                let name = format!("{stem} - mesh {:0width$}", index + 1);
                self.finish_generated_triangulation(
                    name,
                    polyface.vertices.clone(),
                    polyface.faces.clone(),
                    TriSurfaceType::Surface,
                )?;
                loaded += 1;
            }
        } else {
            let mut groups: BTreeMap<String, (Vec<formats::tri00t::Vertex>, Vec<[u32; 3]>)> =
                BTreeMap::new();

            for polyface in &duf.polyfaces {
                let (vertices, faces) = groups
                    .entry(polyface.layer_name.clone())
                    .or_insert_with(|| (Vec::new(), Vec::new()));
                let offset = vertices.len() as u32;
                vertices.extend(polyface.vertices.iter().copied());
                faces.extend(
                    polyface
                        .faces
                        .iter()
                        .map(|face| [face[0] + offset, face[1] + offset, face[2] + offset]),
                );
            }

            for (layer_name, (vertices, faces)) in groups {
                if vertices.is_empty() || faces.is_empty() {
                    continue;
                }
                let name = if layer_name.trim().is_empty() {
                    stem.to_owned()
                } else {
                    format!("{stem} - {layer_name}")
                };
                self.finish_generated_triangulation(
                    name,
                    vertices,
                    faces,
                    TriSurfaceType::Surface,
                )?;
                loaded += 1;
            }
        }

        if loaded == 0 {
            userspace_log!("{} contains no supported DUF mesh entities", path.display());
            return Ok(());
        }

        userspace_log!(
            "Imported {loaded} DUF mesh object{} from {}",
            if loaded == 1 { "" } else { "s" },
            path.display()
        );
        Ok(())
    }

    /// Start a background save of the mesh for `id` to `path`. Metadata and
    /// session updates happen in `poll_saves` once the worker finishes. Shared
    /// by SaveTriangulationAs and SaveAndCloseTriangulationAs actions.
    fn commit_triangulation_save(
        &mut self,
        id: TriangulationId,
        path: PathBuf,
        close_after: bool,
    ) -> Result<()> {
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
        if self.pending_saves.iter().any(|save| {
            save.path == path
                || matches!(save.kind, PendingSaveKind::Save { id: pending, .. } if pending == id)
        }) {
            anyhow::bail!("A save involving {} is already in progress", path.display());
        }
        MeshFormat::from_path(&path)
            .context("Choose a filename ending in .00t, .obj, .stl, or .ply")?;
        let triangulation = self
            .triangulations
            .iter()
            .find(|t| t.id == id)
            .context("The triangulation is no longer loaded")?;
        let mesh = std::sync::Arc::clone(&triangulation.mesh);
        userspace_log!(
            "Saving triangulation '{}' to {}",
            triangulation.name,
            path.display()
        );
        self.spawn_triangulation_write(PendingSaveKind::Save { id, close_after }, mesh, path);
        Ok(())
    }

    /// Spawn a worker thread that writes `mesh` to `path`, streaming progress
    /// back for the status bar. Completion is handled in `poll_saves`.
    fn spawn_triangulation_write(
        &mut self,
        kind: PendingSaveKind,
        mesh: std::sync::Arc<formats::tri00t::Triangulation>,
        path: PathBuf,
    ) {
        self.begin_topology_load();
        let (progress_tx, progress_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        self.pending_saves.push(PendingSave {
            kind,
            path: path.clone(),
            progress_rx,
            result_rx,
            latest_progress: 0.0,
        });
        let window = self.window.clone();
        std::thread::spawn(move || {
            let mut last_update: Option<std::time::Instant> = None;
            let result = formats::write_mesh_with_progress(&mesh, &path, &mut |fraction| {
                // Throttle channel sends + redraws so a fast write doesn't
                // flood the event loop.
                let due = last_update
                    .is_none_or(|last| last.elapsed() >= std::time::Duration::from_millis(100));
                if due {
                    last_update = Some(std::time::Instant::now());
                    let _ = progress_tx.send(fraction);
                    if let Some(w) = window.as_ref() {
                        w.request_redraw();
                    }
                }
            })
            .map_err(|err| anyhow::anyhow!("Failed to write {}: {err}", path.display()));
            let _ = result_tx.send(result);
            if let Some(w) = window.as_ref() {
                w.request_redraw();
            }
        });
    }

    /// Drain progress and completion from background saves. Called each frame
    /// alongside the other `poll_*` methods so results land in the same render
    /// they arrive.
    pub(crate) fn poll_saves(&mut self) {
        let pending = std::mem::take(&mut self.pending_saves);
        let mut still_pending = Vec::new();

        for mut save in pending {
            while let Ok(fraction) = save.progress_rx.try_recv() {
                save.latest_progress = fraction;
            }
            match save.result_rx.try_recv() {
                Ok(Ok(())) => {
                    self.finish_background_save();
                    self.redraw_requested = true;
                    match save.kind {
                        PendingSaveKind::Save { id, close_after } => {
                            if let Some(tri) = self.triangulations.iter_mut().find(|t| t.id == id) {
                                let saved_name = tri.name.clone();
                                tri.path = save.path.clone();
                                tri.name = file_name(&save.path);
                                tri.is_saved = true;
                                if !self.triangulation_files.contains(&save.path) {
                                    self.triangulation_files.push(save.path.clone());
                                }
                                self.triangulation_excluded_paths.remove(&save.path);
                                userspace_log!(
                                    "Saved triangulation '{}' to {}",
                                    saved_name,
                                    save.path.display()
                                );
                                self.persist_session();
                                if close_after {
                                    self.close_triangulation(id);
                                }
                            }
                        }
                        PendingSaveKind::Export { name } => {
                            userspace_log!(
                                "Exported triangulation '{}' to {}",
                                name,
                                save.path.display()
                            );
                        }
                    }
                }
                Ok(Err(e)) => {
                    self.finish_background_save();
                    self.redraw_requested = true;
                    let message = format!("{e:#}");
                    userspace_warn!("Save failed: {message}");
                    if self.exit_after_pending_saves {
                        self.cancel_exit_request();
                    }
                }
                Err(mpsc::TryRecvError::Empty) => still_pending.push(save),
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.finish_background_save();
                    self.redraw_requested = true;
                    if self.exit_after_pending_saves {
                        self.cancel_exit_request();
                    }
                }
            }
        }

        self.pending_saves = still_pending;
        // Drive the status bar from the first in-flight save (or clear it).
        self.editor.status_message =
            self.pending_saves
                .first()
                .map(|save| crate::ui::state::StatusBarMessage {
                    text: match &save.kind {
                        PendingSaveKind::Save { .. } => {
                            format!("Saving to {}", save.path.display())
                        }
                        PendingSaveKind::Export { .. } => {
                            format!("Exporting to {}", save.path.display())
                        }
                    },
                    progress: Some(save.latest_progress),
                });
        self.try_finish_deferred_exit();
    }

    // ── Dialog spawners (non-blocking) ──────────────────────────────────────

    pub(crate) fn choose_new_pidb(&mut self) {
        self.spawn_file_dialog(async {
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter("ProInspector database", &["pidb"])
                .set_file_name("new_project.pidb")
                .save_file()
                .await?
                .into();
            Some(FileDialogAction::NewPidb(path))
        });
    }

    pub(crate) fn choose_open_pidb(&mut self) {
        self.spawn_file_dialog(async {
            let paths = AsyncFileDialog::new()
                .add_filter("ProInspector database", &["pidb"])
                .pick_files()
                .await?
                .into_iter()
                .map(Into::into)
                .collect();
            Some(FileDialogAction::OpenPidb(paths))
        });
    }

    pub(crate) fn open_pidb_path(&mut self, path: &Path) -> Result<()> {
        let pidb = pidb::load(path)?;
        let project = pidb::open_project(Some(path.to_path_buf()), pidb)?;
        self.set_active_project(project);
        userspace_log!("Opened PIDB: {}", path.display());
        Ok(())
    }

    pub(crate) fn import_dxf_paths_into(
        &mut self,
        project_index: usize,
        paths: Vec<PathBuf>,
    ) -> Result<()> {
        let project_runtime_id = self
            .workspace
            .projects
            .get(project_index)
            .map(|project| project.runtime_id)
            .context("The selected .pidb is no longer open")?;
        self.execute_file_dialog_action(FileDialogAction::ImportDxfInto {
            project_runtime_id,
            paths,
        })
    }

    pub(crate) fn choose_import_source_files(&mut self, kind: DataMenu) {
        self.spawn_file_dialog(async move {
            #[cfg(target_os = "macos")]
            let isis_ext: &[&str] = &["isis"];
            #[cfg(not(target_os = "macos"))]
            let isis_ext: &[&str] = &["dgd.isis"];

            let dialog = match kind {
                DataMenu::Dxf => AsyncFileDialog::new().add_filter("AutoCAD DXF", &["dxf"]),
                DataMenu::Pidb => {
                    AsyncFileDialog::new().add_filter("ProInspector database", &["pidb"])
                }
                DataMenu::DgdIsis => {
                    AsyncFileDialog::new().add_filter("Vulcan design database (dgd.isis)", isis_ext)
                }
                DataMenu::Duf => AsyncFileDialog::new().add_filter("Deswik DUF", &["duf"]),
                DataMenu::Tri00t => {
                    AsyncFileDialog::new().add_filter("Vulcan triangulation", &["00t"])
                }
                DataMenu::Obj => AsyncFileDialog::new().add_filter("Wavefront OBJ", &["obj"]),
                DataMenu::Stl => AsyncFileDialog::new().add_filter("STL", &["stl"]),
                DataMenu::Ply => AsyncFileDialog::new().add_filter("PLY", &["ply"]),
                DataMenu::Las => {
                    AsyncFileDialog::new().add_filter("LiDAR point cloud", &["las", "laz"])
                }
                DataMenu::Xyz => {
                    AsyncFileDialog::new().add_filter("ASCII point cloud", &["xyz", "pts"])
                }
                DataMenu::Pcd => AsyncFileDialog::new().add_filter("Point Cloud Data", &["pcd"]),
                _ => AsyncFileDialog::new(),
            };
            let paths: Vec<PathBuf> = dialog
                .pick_files()
                .await?
                .into_iter()
                .map(Into::into)
                .collect();
            Some(FileDialogAction::SetImportSourcePaths { kind, paths })
        });
    }

    pub(crate) fn choose_import_as_pidb_paths(
        &mut self,
        kind: DataMenu,
        source_paths: Vec<PathBuf>,
    ) {
        self.spawn_file_dialog(async move {
            let mut pairs = Vec::new();
            for source_path in source_paths {
                let is_dgd_isis = is_dgd_isis_path(&source_path);
                let is_dxf = is_dxf_path(&source_path);
                let is_duf = is_duf_path(&source_path);
                let matches_kind = match kind {
                    DataMenu::Dxf => is_dxf,
                    DataMenu::DgdIsis => is_dgd_isis,
                    DataMenu::Duf => is_duf,
                    _ => is_dgd_isis || is_dxf || is_duf,
                };
                if !matches_kind {
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
                let Some(pidb_path) = AsyncFileDialog::new()
                    .add_filter("ProInspector database", &["pidb"])
                    .set_file_name(&default_name)
                    .save_file()
                    .await
                else {
                    continue;
                };
                pairs.push((source_path, pidb_path.into()));
            }
            if pairs.is_empty() {
                None
            } else {
                Some(FileDialogAction::ImportAsPidb(pairs))
            }
        });
    }

    pub(crate) fn choose_block_model_bmf(&mut self) {
        self.spawn_file_dialog(async {
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter("Vulcan block model", &["bmf"])
                .pick_file()
                .await?
                .into();
            Some(FileDialogAction::BlockModelBmf(path))
        });
    }

    pub(crate) fn choose_block_model_bdf(&mut self) {
        self.spawn_file_dialog(async {
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter("Vulcan block definition", &["bdf"])
                .pick_file()
                .await?
                .into();
            Some(FileDialogAction::BlockModelBdf(path))
        });
    }

    pub(crate) fn choose_set_block_model_bdf(&mut self, id: BlockModelId) {
        self.spawn_file_dialog(async move {
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter("Vulcan block definition", &["bdf"])
                .pick_file()
                .await?
                .into();
            Some(FileDialogAction::SetBlockModelBdf { id, path })
        });
    }

    pub(crate) fn choose_set_block_model_source_bdf(&mut self, source: BlockModelSource) {
        self.spawn_file_dialog(async move {
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter("Vulcan block definition", &["bdf"])
                .pick_file()
                .await?
                .into();
            Some(FileDialogAction::SetBlockModelSourceBdf { source, path })
        });
    }

    pub(crate) fn choose_open_triangulation_folder(&mut self) {
        self.spawn_file_dialog(async {
            let dir: PathBuf = AsyncFileDialog::new().pick_folder().await?.into();
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
        self.spawn_file_dialog(async move {
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter("DXF", &["dxf"])
                .set_file_name("project.dxf")
                .save_file()
                .await?
                .into();
            Some(FileDialogAction::ExportPidbDxf {
                project_runtime_id,
                path,
            })
        });
    }

    pub(crate) fn choose_export_pidb_copy(&mut self, project_index: usize) {
        let Some(project) = self.workspace.projects.get(project_index) else {
            return;
        };
        let project_runtime_id = project.runtime_id;
        let default_name = project
            .path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .map(|name| name.to_owned())
            .unwrap_or_else(|| {
                let stem = Path::new(&project.pidb.metadata.name)
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .unwrap_or("project");
                format!("{}.pidb", sanitize_file_stem(stem))
            });
        self.spawn_file_dialog(async move {
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter("ProInspector database", &["pidb"])
                .set_file_name(&default_name)
                .save_file()
                .await?
                .into();
            Some(FileDialogAction::ExportPidbCopy {
                project_runtime_id,
                path,
            })
        });
    }

    pub(crate) fn choose_export_layer_dxf(&mut self, project_index: usize, layer: LayerId) {
        let Some(project) = self.workspace.projects.get(project_index) else {
            return;
        };
        let project_runtime_id = project.runtime_id;
        let layer_name = project
            .pidb
            .document
            .layer(layer)
            .map(|layer| layer.name.clone())
            .unwrap_or_else(|| "layer".to_string());
        let default_name = format!("{}.dxf", sanitize_file_stem(&layer_name));
        self.spawn_file_dialog(async move {
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter("DXF", &["dxf"])
                .set_file_name(&default_name)
                .save_file()
                .await?
                .into();
            Some(FileDialogAction::ExportLayerDxf {
                project_runtime_id,
                layer,
                path,
            })
        });
    }

    pub(crate) fn choose_export_triangulation_as(
        &mut self,
        id: TriangulationId,
        format: MeshFormat,
    ) {
        let stem = self
            .triangulations
            .iter()
            .find(|t| t.id == id)
            .and_then(|t| t.path.file_stem().and_then(|s| s.to_str()))
            .unwrap_or("triangulation")
            .to_owned();
        self.spawn_file_dialog(async move {
            let (name, extension) = mesh_format_name_and_extension(format);
            let default_name = format!("{stem}.{extension}");
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter(name, &[extension])
                .set_file_name(default_name)
                .save_file()
                .await?
                .into();
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
        self.spawn_file_dialog(async move {
            let default_name = format!("{stem}.00t");
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter("Vulcan triangulation", &["00t"])
                .add_filter("Wavefront OBJ", &["obj"])
                .add_filter("STL", &["stl"])
                .add_filter("PLY", &["ply"])
                .set_file_name(&default_name)
                .save_file()
                .await?
                .into();
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
        self.spawn_file_dialog(async move {
            let default_name = format!("{stem}.00t");
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter("Vulcan triangulation", &["00t"])
                .add_filter("Wavefront OBJ", &["obj"])
                .add_filter("STL", &["stl"])
                .add_filter("PLY", &["ply"])
                .set_file_name(&default_name)
                .save_file()
                .await?
                .into();
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
        self.spawn_file_dialog(async move {
            let path: PathBuf = AsyncFileDialog::new()
                .add_filter("ProInspector database", &["pidb"])
                .set_file_name("project.pidb")
                .save_file()
                .await?
                .into();
            Some(FileDialogAction::SavePidbAs {
                project_runtime_id,
                path,
            })
        });
    }

    // ── Save helpers used in exit / internal chains ────────────────────────

    /// Open Save As for an unsaved triangulation. Returns false because the
    /// dialog completes later through `poll_file_dialogs`.
    pub(crate) fn save_triangulation_as(&mut self, id: TriangulationId) -> Result<bool> {
        self.triangulations
            .iter()
            .find(|t| t.id == id)
            .context("The triangulation is no longer loaded")?;
        self.spawn_save_triangulation_dialog(id);
        Ok(false)
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
            "Revealed triangulation '{}' in file explorer",
            triangulation.name
        );
        Ok(())
    }

    /// Open the platform file manager with `path` selected (or its parent
    /// directory shown, where selection isn't supported).
    pub(crate) fn reveal_in_file_manager(&self, path: &Path) -> Result<()> {
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
        userspace_log!("Revealed '{}' in file explorer", file_name(path));
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
            .filter_map(|(index, project)| project.has_unsaved_changes().then_some(index))
            .collect();
        let mut saved = 0;
        for index in dirty {
            self.save_named_project(index)?;
            if self
                .workspace
                .projects
                .get(index)
                .is_some_and(OpenProject::has_unsaved_changes)
            {
                return Ok(false);
            }
            saved += 1;
        }
        userspace_log!("Saved {saved} dirty project(s)");
        Ok(true)
    }

    pub(crate) fn request_exit(&mut self) -> Result<()> {
        self.exit_after_pending_saves = false;
        if !self.has_unsaved_changes_for_exit() {
            self.persist_session();
            self.close_requested = true;
            userspace_log!("Exit requested with no unsaved changes");
        } else {
            self.editor.exit_confirm_open = true;
            userspace_log!("User requested exit (unsaved changes present)");
        }
        Ok(())
    }

    pub(crate) fn save_and_exit(&mut self) -> Result<()> {
        self.exit_after_pending_saves = true;
        if self.save_all_dirty_projects()? && self.save_all_unsaved_triangulations()? {
            self.finish_deferred_exit();
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
        self.exit_after_pending_saves = false;
        self.editor.exit_confirm_open = false;
        self.persist_session();
        self.close_requested = true;
        userspace_log!("User chose to exit without saving");
    }

    pub(crate) fn cancel_exit_request(&mut self) {
        self.exit_after_pending_saves = false;
        self.editor.exit_confirm_open = false;
    }

    fn has_unsaved_changes_for_exit(&self) -> bool {
        self.workspace
            .projects
            .iter()
            .any(project_has_unsaved_changes)
            || self.triangulations.iter().any(|tri| !tri.is_saved)
    }

    fn try_finish_deferred_exit(&mut self) {
        if self.exit_after_pending_saves
            && self.pending_file_dialogs.is_empty()
            && self.pending_saves.is_empty()
            && !self.has_unsaved_changes_for_exit()
        {
            self.finish_deferred_exit();
        }
    }

    fn finish_deferred_exit(&mut self) {
        self.exit_after_pending_saves = false;
        self.editor.exit_confirm_open = false;
        self.persist_session();
        self.close_requested = true;
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
        project.invalidate_disk_snapshot();
        userspace_log!("Saved PIDB index {}: {}", index, path.display());
        self.persist_session();
        Ok(())
    }

    pub(crate) fn save_named_project_as(&mut self, index: usize) -> Result<()> {
        if self.workspace.active_index == Some(index) && self.has_pending_move_delta() {
            self.commit_pending_move();
        }
        self.ensure_project_has_no_pending_text_edit(index)?;
        self.workspace
            .projects
            .get(index)
            .context("No project at that index")?;
        self.spawn_save_pidb_as_dialog(index);
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
        project.invalidate_disk_snapshot();
        project.invalidate_dirty_layers();
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
        project.invalidate_dirty_layers();
    }

    pub(crate) fn request_close_project(&mut self, index: usize) {
        let Some(project) = self.workspace.projects.get(index) else {
            return;
        };
        if project_has_unsaved_changes(project) {
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
            .is_some_and(OpenProject::has_unsaved_changes)
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
        self.shift_editor_project_indices_after_close(index);
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
        self.invalidate_geometry();
        Ok(())
    }

    fn shift_editor_project_indices_after_close(&mut self, closed_index: usize) {
        shift_project_index(&mut self.editor.pending_close_project, closed_index);
        shift_project_index(&mut self.editor.new_layer_project_index, closed_index);
        shift_project_index(&mut self.editor.import_dxf_project_index, closed_index);
        shift_project_index(&mut self.editor.export_project_index, closed_index);
        shift_plain_project_index(&mut self.editor.tri_contour_project_index, closed_index);

        self.editor.pending_unload_queue = self
            .editor
            .pending_unload_queue
            .drain(..)
            .filter_map(|(mut project_index, layer_id, name)| {
                shift_plain_project_index_checked(&mut project_index, closed_index).then_some((
                    project_index,
                    layer_id,
                    name,
                ))
            })
            .collect();
        if let Some((project_index, _, _)) = &mut self.editor.renaming_layer
            && !shift_plain_project_index_checked(project_index, closed_index)
        {
            self.editor.renaming_layer = None;
        }
        if let Some((project_index, _, _)) = &mut self.editor.pending_delete_layer
            && !shift_plain_project_index_checked(project_index, closed_index)
        {
            self.editor.pending_delete_layer = None;
        }
        if let Some((project_index, _)) = &mut self.editor.export_layer
            && !shift_plain_project_index_checked(project_index, closed_index)
        {
            self.editor.export_layer = None;
        }
        if let Some(dialog) = &mut self.editor.move_layer_dialog {
            let source_ok =
                shift_plain_project_index_checked(&mut dialog.source_project_index, closed_index);
            shift_project_index(&mut dialog.target_project_index, closed_index);
            if !source_ok {
                self.editor.move_layer_dialog = None;
            }
        }
    }
}

fn shift_project_index(index: &mut Option<usize>, closed_index: usize) {
    if let Some(value) = index
        && !shift_plain_project_index_checked(value, closed_index)
    {
        *index = None;
    }
}

fn shift_plain_project_index(index: &mut usize, closed_index: usize) {
    if *index > closed_index {
        *index -= 1;
    }
}

fn shift_plain_project_index_checked(index: &mut usize, closed_index: usize) -> bool {
    if *index == closed_index {
        false
    } else {
        shift_plain_project_index(index, closed_index);
        true
    }
}

/// Replace characters that are invalid in filenames across Windows, macOS, and
/// Linux. Falls back to `"layer"` when the result is empty (e.g. a blank name).
fn sanitize_file_stem(name: &str) -> String {
    const INVALID: &[char] = &['/', '\\', ':', '*', '?', '"', '<', '>', '|'];
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| if INVALID.contains(&c) { '_' } else { c })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "layer".to_string()
    } else {
        trimmed.to_string()
    }
}
