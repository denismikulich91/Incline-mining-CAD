use super::*;

impl<'a> App<'a> {
    fn clear_triangulation_entity_state(&mut self, handle: crate::model::SceneEntityId) {
        self.editor.selected_handles.remove(&handle);
        self.editor.hidden_handles.remove(&handle);
        self.editor.frozen_handles.remove(&handle);
        self.editor.translucent_handles.remove(&handle);
    }

    pub(crate) fn open_triangulation_path(&mut self, path: &std::path::Path) -> Result<()> {
        if self.triangulation_excluded_paths.remove(path) {
            self.refresh_triangulation_dir_entries();
        }
        if self
            .triangulations
            .iter()
            .find(|tri| tri.path.as_path() == path)
            .is_some()
        {
            self.invalidate_geometry();
            return Ok(());
        }
        if self
            .pending_triangulation_loads
            .iter()
            .any(|(pending_path, _)| pending_path == path)
        {
            return Ok(());
        }

        let name = file_name(path);
        let path = path.to_path_buf();
        let scene_was_empty =
            self.triangulations.is_empty() && self.scene_document.objects().is_empty();

        self.begin_topology_load();

        let (tx, rx) = std::sync::mpsc::channel();
        self.pending_triangulation_loads.push((path.clone(), rx));

        let window = self.window.clone();
        std::thread::spawn(move || {
            let result = (|| -> Result<LoadedTriangulation> {
                let mesh = formats::read_mesh(&path)
                    .map_err(|err| anyhow::anyhow!("Failed to read {}: {err}", path.display()))?;
                userspace_log!(
                    "Loaded triangulation '{}' ({}, {} vertices, {} faces)",
                    name,
                    path.display(),
                    mesh.vertex_count(),
                    mesh.face_count()
                );
                let spatial = std::sync::Arc::new(crate::model::spatial::TriangleBvh::build(&mesh));
                let edges = crate::model::triangulation::unique_edges(&mesh);
                let surface_face_order = std::sync::Arc::new(
                    crate::model::triangulation::morton_surface_face_order(&mesh),
                );
                Ok(LoadedTriangulation {
                    name,
                    path,
                    mesh: std::sync::Arc::new(mesh),
                    spatial,
                    edges,
                    surface_face_order,
                    scene_was_empty,
                })
            })();
            let _ = tx.send(result);
            if let Some(w) = window {
                w.request_redraw();
            }
        });

        Ok(())
    }

    /// Drain any completed background triangulation loads and integrate their results.
    /// Called at the start of each frame so results appear in the same render they arrive.
    pub(crate) fn poll_triangulation_loads(&mut self) {
        let receivers = std::mem::take(&mut self.pending_triangulation_loads);
        let mut still_pending = Vec::new();

        for (path, rx) in receivers {
            match rx.try_recv() {
                Ok(Ok(loaded)) => {
                    self.pending_loads = self.pending_loads.saturating_sub(1);
                    let id = TriangulationId(self.next_triangulation_id);
                    self.next_triangulation_id += 1;
                    self.triangulations.push(OpenTriangulation {
                        id,
                        name: loaded.name,
                        path: loaded.path,
                        is_saved: true,
                        mesh: loaded.mesh,
                        spatial: loaded.spatial,
                        edges: loaded.edges,
                        surface_face_order: loaded.surface_face_order,
                        visible: true,
                        color: DEFAULT_TRIANGULATION_COLOR,
                        line_color: [0.05, 0.08, 0.10, 1.0],
                        line_weight: Some(1.0),
                    });
                    if loaded.scene_was_empty {
                        self.fit_view_to_extents();
                    }
                    self.topology_load_pending_gpu = true;
                    self.persist_session();
                    // The mesh renders from triangulation_gpu's per-id cache, not
                    // the document scene. A full invalidate_geometry here would
                    // re-upload every vector object per loaded tri.
                    self.invalidate_topology_bounds_and_redraw();
                }
                Ok(Err(e)) => {
                    self.pending_loads = self.pending_loads.saturating_sub(1);
                    let message = format!("{e:#}");
                    crate::userspace_warn!("Failed to load triangulation: {message}");
                    self.finish_topology_load();
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    still_pending.push((path, rx));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.pending_loads = self.pending_loads.saturating_sub(1);
                    self.finish_topology_load();
                }
            }
        }

        self.pending_triangulation_loads = still_pending;
    }

    fn cancel_pending_triangulation_loads(
        &mut self,
        mut matches: impl FnMut(&std::path::Path) -> bool,
    ) {
        let before = self.pending_triangulation_loads.len();
        self.pending_triangulation_loads
            .retain(|(path, _)| !matches(path));
        let cancelled = before - self.pending_triangulation_loads.len();
        self.pending_loads = self.pending_loads.saturating_sub(cancelled);
        if cancelled > 0 {
            self.finish_topology_load();
        }
    }

    pub(crate) fn activate_triangulation(&mut self, id: TriangulationId) {
        let Some(tri) = self.triangulations.iter().find(|tri| tri.id == id) else {
            return;
        };
        let handle = tri.entity_id();
        if self.active_triangulation == Some(id) && self.editor.selected_handles.contains(&handle) {
            self.active_triangulation = None;
            self.editor.selected_handles.remove(&handle);
            userspace_log!("Deselected triangulation '{}'", tri.name);
            self.request_topology_redraw();
            return;
        }
        let cleared_object_selection = self
            .editor
            .selected_handles
            .iter()
            .any(|handle| matches!(handle, crate::model::SceneEntityId::Object(_)));
        self.active_triangulation = Some(id);
        self.editor.selected_handles.clear();
        self.editor.selected_handles.insert(handle);
        userspace_log!("Activated triangulation '{}'", tri.name);
        if cleared_object_selection {
            self.invalidate_geometry();
        } else {
            self.request_topology_redraw();
        }
    }

    pub(crate) fn toggle_triangulation_visible(&mut self, id: TriangulationId) {
        let Some(tri) = self.triangulations.iter_mut().find(|tri| tri.id == id) else {
            return;
        };
        tri.visible = !tri.visible;
        let action = if tri.visible { "Shown" } else { "Hidden" };
        userspace_log!("{} triangulation '{}'", action, tri.name);
        self.invalidate_topology_bounds_and_redraw();
    }

    pub(crate) fn close_triangulation(&mut self, id: TriangulationId) {
        let Some(index) = self.triangulations.iter().position(|tri| tri.id == id) else {
            return;
        };
        let tri = self.triangulations.remove(index);
        let handle = tri.entity_id();
        self.clear_triangulation_entity_state(handle);
        self.clear_dialog_refs_to_triangulation(id);
        if self.active_triangulation == Some(id) {
            self.active_triangulation = None;
        }
        userspace_log!("Closed triangulation '{}'", tri.name);
        self.invalidate_topology_bounds_and_redraw();
        self.persist_session();
    }

    /// Unload a loaded mesh (if any) and drop the path from the individual-file tracker.
    pub(crate) fn remove_triangulation(&mut self, path: PathBuf) {
        self.cancel_pending_triangulation_loads(|pending_path| pending_path == path);
        if let Some(idx) = self.triangulations.iter().position(|t| t.path == path) {
            let tri = self.triangulations.remove(idx);
            let handle = tri.entity_id();
            self.clear_triangulation_entity_state(handle);
            self.clear_dialog_refs_to_triangulation(tri.id);
            if self.active_triangulation == Some(tri.id) {
                self.active_triangulation = None;
            }
            self.invalidate_topology_bounds_and_redraw();
            userspace_log!("Removed triangulation '{}'", tri.name);
        }
        self.triangulation_files.retain(|f| f != &path);
        if self
            .triangulation_dirs
            .iter()
            .any(|dir| path.parent() == Some(dir.as_path()))
        {
            self.triangulation_excluded_paths.insert(path.clone());
            self.refresh_triangulation_dir_entries();
        }
        userspace_log!(
            "Removed triangulation '{}' from file tracker",
            path.file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default()
        );
        self.persist_session();
    }

    fn clear_dialog_refs_to_triangulation(&mut self, id: TriangulationId) {
        if self.editor.tri_cut_poly_tri_id == Some(id) {
            self.editor.tri_cut_poly_tri_id = None;
            self.editor.tri_cut_poly_open = false;
            self.editor.tri_cut_poly_awaiting_pick = false;
        }
        if self.editor.tri_cut_z_tri_id == Some(id) {
            self.editor.tri_cut_z_tri_id = None;
            self.editor.tri_cut_z_open = false;
        }
        if self.editor.tri_cut_surface_target_id == Some(id)
            || self.editor.tri_cut_surface_reference_id == Some(id)
        {
            self.editor.tri_cut_surface_target_id = None;
            self.editor.tri_cut_surface_reference_id = None;
            self.editor.tri_cut_surface_open = false;
        }
        if self.editor.tri_cut_pitshell_topology_id == Some(id)
            || self.editor.tri_cut_pitshell_pitshell_id == Some(id)
        {
            self.editor.tri_cut_pitshell_topology_id = None;
            self.editor.tri_cut_pitshell_pitshell_id = None;
            self.editor.tri_cut_pitshell_open = false;
        }
        if self.editor.tri_include_solid_topology_id == Some(id)
            || self.editor.tri_include_solid_shape_id == Some(id)
        {
            self.editor.tri_include_solid_topology_id = None;
            self.editor.tri_include_solid_shape_id = None;
            self.editor.tri_include_solid_open = false;
        }
        if self.editor.tri_contour_tri_id == Some(id) {
            self.editor.tri_contour_tri_id = None;
            self.editor.tri_contour_open = false;
        }
    }

    /// Unload every mesh whose path lives in `dir`, and drop the dir from the tracked list.
    pub(crate) fn remove_triangulation_folder(&mut self, dir: PathBuf) {
        self.cancel_pending_triangulation_loads(|pending_path| {
            pending_path.parent() == Some(dir.as_path())
        });
        let to_remove: Vec<_> = self
            .triangulations
            .iter()
            .filter(|t| t.path.parent() == Some(dir.as_path()))
            .map(|t| t.id)
            .collect();
        for id in to_remove {
            if let Some(idx) = self.triangulations.iter().position(|t| t.id == id) {
                let tri = self.triangulations.remove(idx);
                let handle = tri.entity_id();
                self.clear_triangulation_entity_state(handle);
                if self.active_triangulation == Some(tri.id) {
                    self.active_triangulation = None;
                }
            }
        }
        self.triangulation_dirs.retain(|d| d != &dir);
        self.triangulation_excluded_paths
            .retain(|path| path.parent() != Some(dir.as_path()));
        self.triangulation_dir_entries.remove(&dir);
        self.invalidate_topology_bounds_and_redraw();
        userspace_log!("Removed triangulation folder '{}'", dir.display());
        self.persist_session();
    }

    pub(crate) fn load_all_triangulations_in_folder(&mut self, dir: PathBuf) -> anyhow::Result<()> {
        let paths: Vec<PathBuf> = self
            .triangulation_dir_entries
            .get(&dir)
            .cloned()
            .unwrap_or_default();
        let count = paths.len();
        for path in paths {
            self.open_triangulation_path(&path)?;
        }
        userspace_log!(
            "Loaded {} triangulation(s) from folder {}",
            count,
            dir.display()
        );
        Ok(())
    }

    pub(crate) fn close_all_triangulations_in_folder(&mut self, dir: PathBuf) {
        // Cancel any loads that haven't completed yet so they don't reappear after unload.
        self.cancel_pending_triangulation_loads(|pending_path| {
            pending_path.parent() == Some(dir.as_path())
        });
        let ids: Vec<TriangulationId> = self
            .triangulations
            .iter()
            .filter(|t| t.path.parent() == Some(dir.as_path()))
            .map(|t| t.id)
            .collect();
        let count = ids.len();
        for id in ids {
            self.close_triangulation(id);
        }
        userspace_log!(
            "Closed {} triangulation(s) from folder {}",
            count,
            dir.display()
        );
    }

    pub(crate) fn set_triangulation_color(&mut self, tri_id: TriangulationId, new_color: [f32; 4]) {
        if let Some(tri) = self.triangulations.iter_mut().find(|t| t.id == tri_id) {
            tri.color = new_color;
        }
        userspace_log!("Set triangulation {:?} color to {:?}", tri_id, new_color);
        self.request_topology_redraw();
    }
    /// Build the mesh/BVH/edges for a freshly generated triangulation, register
    /// it, and select it. Shared by the CDT and loft generation paths.
    pub(crate) fn finish_generated_triangulation(
        &mut self,
        name: String,
        tri_vertices: Vec<tri00t::Vertex>,
        tri_faces: Vec<[u32; 3]>,
        surface_type: TriSurfaceType,
    ) -> Result<()> {
        self.finish_generated_triangulation_with_edge_builder(
            name,
            tri_vertices,
            tri_faces,
            surface_type,
            crate::model::triangulation::unique_edges,
        )
    }

    fn finish_generated_triangulation_with_edge_builder(
        &mut self,
        name: String,
        tri_vertices: Vec<tri00t::Vertex>,
        tri_faces: Vec<[u32; 3]>,
        surface_type: TriSurfaceType,
        build_edges: impl FnOnce(&tri00t::Triangulation) -> Vec<[u32; 2]>,
    ) -> Result<()> {
        let built = build_generated_triangulation(
            name,
            tri_vertices,
            tri_faces,
            surface_type,
            build_edges,
        )?;
        self.insert_generated_triangulation(built);
        Ok(())
    }

    /// Apply the result of a background job that produces one triangulation
    /// with a single completion log line: log + insert on success, warn on
    /// failure. Shared by the backgrounded cut/create paths.
    pub(crate) fn apply_generated_triangulation_job(
        &mut self,
        result: Result<crate::model::triangulation::GeneratedTriangulationLog>,
    ) {
        match result {
            Ok(log) => {
                userspace_log!("{}", log.message);
                self.insert_generated_triangulation(log.generated);
            }
            Err(err) => {
                let message = format!("{err:#}");
                crate::userspace_warn!("Triangulation operation failed: {message}");
            }
        }
    }

    /// UI-thread half of a generated-triangulation apply: register a
    /// pre-built mesh/BVH/edge bundle as a new `OpenTriangulation` and select
    /// it. The heavy build (mesh assembly + BVH) is done by
    /// `build_generated_triangulation`, which can run on a worker thread.
    pub(crate) fn insert_generated_triangulation(
        &mut self,
        built: crate::model::triangulation::GeneratedTriangulation,
    ) {
        let crate::model::triangulation::GeneratedTriangulation {
            name,
            mesh,
            spatial,
            edges,
            surface_face_order,
            surface_type,
        } = built;
        let vertex_count = mesh.vertex_count();
        let face_count = mesh.face_count();

        let id = TriangulationId(self.next_triangulation_id);
        self.next_triangulation_id += 1;

        let synthetic_path = PathBuf::from(format!("generated::{}::{name}", id.0));

        let cleared_object_selection = self
            .editor
            .selected_handles
            .iter()
            .any(|handle| matches!(handle, crate::model::SceneEntityId::Object(_)));
        self.triangulations.push(OpenTriangulation {
            id,
            name: name.clone(),
            path: synthetic_path,
            is_saved: false,
            mesh,
            spatial,
            edges,
            surface_face_order,
            visible: true,
            color: DEFAULT_TRIANGULATION_COLOR,
            line_color: [0.05, 0.08, 0.10, 1.0],
            line_weight: Some(1.0),
        });
        self.active_triangulation = Some(id);
        self.editor.selected_handles.clear();
        self.editor
            .selected_handles
            .insert(crate::model::SceneEntityId::Triangulation(id));
        userspace_log!(
            "Created triangulation '{name}' ({vertex_count} vertices, {face_count} faces) from surface type {:?}",
            surface_type
        );
        if cleared_object_selection {
            self.invalidate_geometry();
        } else {
            self.invalidate_topology_bounds_and_redraw();
        }
    }
}

/// Worker-thread half of a generated-triangulation apply: assemble the mesh,
/// build its BVH and edge list. Pure (no `App`), so it can run off the UI
/// thread; the result is handed to `insert_generated_triangulation`.
pub(crate) fn build_generated_triangulation(
    name: String,
    tri_vertices: Vec<tri00t::Vertex>,
    tri_faces: Vec<[u32; 3]>,
    surface_type: TriSurfaceType,
    build_edges: impl FnOnce(&tri00t::Triangulation) -> Vec<[u32; 2]>,
) -> Result<crate::model::triangulation::GeneratedTriangulation> {
    if tri_faces.is_empty() {
        anyhow::bail!("Triangulation produced no faces");
    }
    let mesh = tri00t::Triangulation::from_vertices_and_faces(tri_vertices, tri_faces);
    let spatial = std::sync::Arc::new(crate::model::spatial::TriangleBvh::build(&mesh));
    let edges = build_edges(&mesh);
    let surface_face_order = std::sync::Arc::new(
        crate::model::triangulation::morton_surface_face_order(&mesh),
    );
    Ok(crate::model::triangulation::GeneratedTriangulation {
        name,
        mesh: std::sync::Arc::new(mesh),
        spatial,
        edges,
        surface_face_order,
        surface_type,
    })
}
