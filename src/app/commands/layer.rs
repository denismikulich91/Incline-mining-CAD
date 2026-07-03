use anyhow::Result;

use crate::{
    app::App,
    model::{Command, Document, Layer, LayerId, Object, SceneEntityId},
    userspace_log, userspace_warn,
};

fn unique_layer_name(document: &Document, preferred: &str) -> String {
    if document.layer_id_by_name(preferred).is_none() {
        return preferred.to_string();
    }

    for index in 2.. {
        let candidate = format!("{preferred} {index}");
        if document.layer_id_by_name(&candidate).is_none() {
            return candidate;
        }
    }

    unreachable!("unbounded iterator should always find a unique layer name")
}

fn objects_on_layer(document: &Document, layer_id: LayerId) -> Vec<Object> {
    document
        .objects()
        .iter()
        .filter(|object| object.layer() == layer_id)
        .cloned()
        .collect()
}

impl<'a> App<'a> {
    pub(crate) fn create_layer(&mut self, project_index: usize, name: String) -> Result<()> {
        let Some(project) = self.workspace.projects.get_mut(project_index) else {
            return Ok(());
        };
        let layer_id =
            project
                .pidb
                .document
                .add_layer(name.clone(), None, [1.0, 1.0, 1.0, 1.0], true, 0.0);
        project.dirty = true;
        project.loaded_layers.insert(layer_id);
        self.workspace.set_active_index(project_index);
        self.history.clear();
        self.editor.selected_handles.clear();
        self.editor.active_layer = Some(layer_id);
        userspace_log!("Created layer '{name}' in PIDB index {project_index}");
        self.invalidate_geometry();
        Ok(())
    }

    pub(crate) fn delete_layer(&mut self, project_index: usize, layer_id: LayerId) -> Result<()> {
        let Some(project) = self.workspace.projects.get(project_index) else {
            self.editor.pending_delete_layer = None;
            return Ok(());
        };
        let Some(layer) = project.pidb.document.layer(layer_id).cloned() else {
            self.editor.pending_delete_layer = None;
            return Ok(());
        };
        let on_layer: Vec<_> = project
            .pidb
            .document
            .objects()
            .iter()
            .filter(|o| o.layer() == layer_id)
            .cloned()
            .collect();
        self.workspace.set_active_index(project_index);
        let deleted = {
            let Some(project) = self.workspace.projects.get_mut(project_index) else {
                return Ok(());
            };
            self.history.execute(
                &mut project.pidb.document,
                Command::DeleteLayerSnapshot {
                    layer,
                    objects: on_layer,
                },
            );
            project.loaded_layers.remove(&layer_id);
            project.dirty = true;
            true
        };
        if deleted {
            self.editor.pending_delete_layer = None;
            if self.editor.active_layer == Some(layer_id) {
                self.editor.active_layer = None;
            }
            self.editor.selected_handles.clear();
            userspace_log!("Deleted layer {:?} in project {}", layer_id, project_index);
            userspace_log!("Deleted layer {:?} (and all objects on it)", layer_id);
            self.invalidate_geometry();
        }
        Ok(())
    }

    pub(crate) fn duplicate_layer(&mut self, project_index: usize, layer_id: LayerId) {
        self.workspace.set_active_index(project_index);
        let Some(project) = self.workspace.projects.get_mut(project_index) else {
            return;
        };
        let Some(source_layer) = project.pidb.document.layer(layer_id).cloned() else {
            return;
        };
        let source_objects = objects_on_layer(&project.pidb.document, layer_id);
        let duplicate_name = unique_layer_name(
            &project.pidb.document,
            &format!("{} copy", source_layer.name),
        );

        let doc = &mut project.pidb.document;
        let new_layer_id = doc.allocate_layer_id();
        let duplicate_layer = Layer {
            id: new_layer_id,
            name: duplicate_name.clone(),
            color_index: source_layer.color_index,
            color: source_layer.color,
            visible: source_layer.visible,
            elevation: source_layer.elevation,
        };
        let duplicate_objects: Vec<Object> = source_objects
            .into_iter()
            .map(|object| {
                let object_id = doc.allocate_object_id();
                object.with_id_and_layer(object_id, new_layer_id)
            })
            .collect();

        self.history.execute(
            doc,
            Command::AddLayerSnapshot {
                layer: duplicate_layer,
                objects: duplicate_objects,
            },
        );
        project.loaded_layers.insert(new_layer_id);
        project.dirty = true;
        project.invalidate_dirty_layers();
        self.editor.selected_handles.clear();
        userspace_log!("Duplicated layer '{duplicate_name}' in project {project_index}");
        self.invalidate_geometry();
    }

    pub(crate) fn move_layer_to_project(
        &mut self,
        source_project_index: usize,
        layer_id: LayerId,
        target_project_index: usize,
    ) {
        if source_project_index == target_project_index {
            return;
        }

        let Some(source_project) = self.workspace.projects.get(source_project_index) else {
            return;
        };
        let Some(source_layer) = source_project.pidb.document.layer(layer_id).cloned() else {
            return;
        };
        let source_objects = objects_on_layer(&source_project.pidb.document, layer_id);

        let Some(target_project) = self.workspace.projects.get_mut(target_project_index) else {
            return;
        };
        let new_layer_id = target_project.pidb.document.allocate_layer_id();
        let target_name = unique_layer_name(&target_project.pidb.document, &source_layer.name);
        let moved_layer = Layer {
            id: new_layer_id,
            name: target_name.clone(),
            color_index: source_layer.color_index,
            color: source_layer.color,
            visible: source_layer.visible,
            elevation: source_layer.elevation,
        };
        let moved_objects: Vec<Object> = source_objects
            .iter()
            .map(|object| {
                let object_id = target_project.pidb.document.allocate_object_id();
                object.with_id_and_layer(object_id, new_layer_id)
            })
            .collect();
        let moved_handles: Vec<SceneEntityId> = moved_objects
            .iter()
            .map(|object| SceneEntityId::Object(object.id()))
            .collect();

        self.history.clear();
        self.editor.selected_handles.clear();
        self.editor.hidden_handles.clear();
        self.editor.frozen_handles.clear();
        self.editor.translucent_handles.clear();

        if let Some(source_project) = self.workspace.projects.get_mut(source_project_index) {
            for object in &source_objects {
                source_project.pidb.document.remove_object(object.id());
            }
            source_project.pidb.document.delete_layer(layer_id);
            source_project.loaded_layers.remove(&layer_id);
            source_project.dirty = true;
            source_project.invalidate_dirty_layers();
        }

        if let Some(target_project) = self.workspace.projects.get_mut(target_project_index) {
            target_project
                .pidb
                .document
                .append_layer_snapshot(&moved_layer, moved_objects.iter());
            target_project.loaded_layers.insert(new_layer_id);
            target_project.dirty = true;
            target_project.invalidate_dirty_layers();
        }

        if self.editor.active_layer == Some(layer_id) {
            self.editor.active_layer = None;
        }
        self.workspace.set_active_index(target_project_index);
        self.editor.active_layer = Some(new_layer_id);
        self.editor.selected_handles = moved_handles.into_iter().collect();
        userspace_log!(
            "Moved layer {:?} from project {} to project {}",
            layer_id,
            source_project_index,
            target_project_index
        );
        self.invalidate_geometry();
    }

    pub(crate) fn load_layer(&mut self, project_index: usize, layer_id: LayerId) {
        let scene_was_empty =
            self.scene_document.objects().is_empty() && self.triangulations.is_empty();
        let Some(project) = self.workspace.projects.get_mut(project_index) else {
            return;
        };
        let Some(name) = project
            .pidb
            .document
            .layer(layer_id)
            .map(|layer| layer.name.clone())
        else {
            return;
        };
        project.loaded_layers.insert(layer_id);
        // self.workspace.set_active_index(project_index); // we dont want loading a layer to make
        // it active
        self.history.clear();
        self.editor.selected_handles.clear();
        // self.editor.active_layer = Some(layer_id); // we dont want loading a layer to make it
        // active
        userspace_log!("Loaded layer '{name}' in project {project_index}");
        userspace_log!("Loaded layer '{name}' from PIDB index {project_index}");
        self.invalidate_geometry();
        if scene_was_empty {
            self.fit_view_to_extents();
        }
    }

    /// Unload a layer, showing a dirty-check confirmation dialog if the project has
    /// unsaved changes. The dialog's "Save and Unload" / "Unload Without Saving" buttons
    /// dispatch `SaveAndUnloadLayer` / `UnloadLayerConfirmed` which bypass this check.
    pub(crate) fn try_unload_layer(&mut self, project_index: usize, layer_id: LayerId) {
        if self.project_layer_is_dirty(project_index, layer_id) {
            let name = self.layer_display_name(project_index, layer_id);
            self.queue_pending_unload(project_index, layer_id, name);
        } else {
            self.unload_layer(project_index, layer_id);
        }
    }

    fn queue_pending_unload(&mut self, project_index: usize, layer_id: LayerId, name: String) {
        if self
            .editor
            .pending_unload_queue
            .iter()
            .any(|(queued_project, queued_layer, _)| {
                *queued_project == project_index && *queued_layer == layer_id
            })
        {
            return;
        }
        self.editor
            .pending_unload_queue
            .push((project_index, layer_id, name));
    }

    fn project_layer_is_dirty(&self, project_index: usize, layer_id: LayerId) -> bool {
        self.workspace
            .projects
            .get(project_index)
            .filter(|project| project.loaded_layers.contains(&layer_id))
            .is_some_and(|project| project.dirty_layers().contains(&layer_id))
    }

    fn layer_display_name(&self, project_index: usize, layer_id: LayerId) -> String {
        self.workspace
            .projects
            .get(project_index)
            .and_then(|p| p.pidb.document.layer(layer_id))
            .map(|l| l.name.clone())
            .unwrap_or_else(|| "Layer".to_string())
    }

    /// Unload any queued layers whose project is no longer dirty (e.g. after a
    /// "Save and Unload" cleared the whole project), so the per-layer dialog only
    /// ever prompts for layers that still have unsaved changes.
    pub(crate) fn drain_clean_unload_queue(&mut self) {
        let mut i = 0;
        while i < self.editor.pending_unload_queue.len() {
            let (project_index, layer_id, _) = self.editor.pending_unload_queue[i];
            if self.project_layer_is_dirty(project_index, layer_id) {
                i += 1;
            } else {
                self.editor.pending_unload_queue.remove(i);
                self.unload_layer(project_index, layer_id);
            }
        }
    }

    /// Discard a layer's unsaved edits, reverting its objects to the on-disk
    /// version (or removing them entirely if the project was never saved), then
    /// unload it. Backs the "Unload Without Saving" dialog action so reloading
    /// the layer afterwards shows the saved state rather than the discarded edits.
    pub(crate) fn unload_layer_discarding_changes(
        &mut self,
        project_index: usize,
        layer_id: LayerId,
    ) {
        if !self.revert_layer_from_disk(project_index, layer_id) {
            return;
        }
        self.unload_layer(project_index, layer_id);
        if let Some(project) = self.workspace.projects.get_mut(project_index) {
            // Use dirty_layers() rather than a separate differs_from_disk() call: it reads
            // the file once, caches the result, and the next frame's project_view() gets a
            // cache hit instead of a third disk read.
            let still_dirty = project.dirty_layers();
            project.dirty = match &project.path {
                Some(_) => !still_dirty.is_empty(),
                None => {
                    !project.pidb.document.layers().is_empty()
                        || !project.pidb.document.objects().is_empty()
                }
            };
            // dirty_layers() just populated the cache — don't invalidate it here.
        }
        self.editor
            .pending_unload_queue
            .retain(|(queued_project, queued_layer, _)| {
                *queued_project != project_index || *queued_layer != layer_id
            });
        self.invalidate_geometry();
        userspace_log!(
            "Discarded and unloaded layer {:?} in project {}",
            layer_id,
            project_index
        );
        userspace_log!(
            "Discarded changes and unloaded layer {:?} from PIDB index {project_index}",
            layer_id
        );
    }

    fn revert_layer_from_disk(&mut self, project_index: usize, layer_id: LayerId) -> bool {
        let Some(project) = self.workspace.projects.get_mut(project_index) else {
            return false;
        };
        // Recover the runtime namespace that was applied when this project opened
        // so the disk ids line up with the in-memory ones.
        let namespace = (layer_id.0 >> 32) as u32;
        let disk_snapshot = match &project.path {
            Some(path) => match crate::model::pidb::load(path) {
                Ok(mut disk) => {
                    disk.document.apply_runtime_namespace(namespace);
                    disk.document.layer(layer_id).cloned().map(|layer| {
                        let objects: Vec<crate::model::Object> = disk
                            .document
                            .objects()
                            .iter()
                            .filter(|object| object.layer() == layer_id)
                            .cloned()
                            .collect();
                        (layer, objects)
                    })
                }
                // Can't read disk (deleted/locked): keep the layer loaded and intact.
                Err(error) => {
                    userspace_warn!("Could not discard layer changes: {error:#}");
                    return false;
                }
            },
            // Never saved: discarding removes the complete layer.
            None => None,
        };
        let in_memory_ids: Vec<crate::model::ObjectId> = project
            .pidb
            .document
            .objects()
            .iter()
            .filter(|o| o.layer() == layer_id)
            .map(|o| o.id())
            .collect();
        for id in in_memory_ids {
            project.pidb.document.remove_object(id);
        }
        project.pidb.document.delete_layer(layer_id);
        if let Some((layer, objects)) = disk_snapshot {
            project
                .pidb
                .document
                .append_layer_snapshot(&layer, objects.iter());
        }
        project.invalidate_dirty_layers();
        self.history.clear();
        self.editor.selected_handles.clear();
        true
    }

    pub(crate) fn save_and_unload_layer(
        &mut self,
        project_index: usize,
        layer_id: LayerId,
    ) -> Result<()> {
        self.save_project_layer(project_index, layer_id)?;
        if !self.project_layer_is_dirty(project_index, layer_id) {
            self.unload_layer(project_index, layer_id);
            self.editor
                .pending_unload_queue
                .retain(|(queued_project, queued_layer, _)| {
                    *queued_project != project_index || *queued_layer != layer_id
                });
            userspace_log!(
                "Saved and unloaded layer {:?} in project {}",
                layer_id,
                project_index
            );
            userspace_log!(
                "Saved and unloaded layer {:?} from PIDB index {project_index}",
                layer_id
            );
        }
        Ok(())
    }

    pub(crate) fn load_all_layers_in_project(&mut self, project_index: usize) {
        let layer_ids: Vec<LayerId> = self
            .workspace
            .projects
            .get(project_index)
            .map(|p| p.pidb.document.layers().iter().map(|l| l.id).collect())
            .unwrap_or_default();
        let scene_was_empty =
            self.scene_document.objects().is_empty() && self.triangulations.is_empty();
        let Some(project) = self.workspace.projects.get_mut(project_index) else {
            return;
        };
        for id in &layer_ids {
            project.loaded_layers.insert(*id);
        }
        self.workspace.set_active_index(project_index);
        self.history.clear();
        self.editor.selected_handles.clear();
        userspace_log!(
            "Loaded {} layer(s) in project {project_index}",
            layer_ids.len()
        );
        userspace_log!(
            "Loaded {} layer(s) in PIDB index {project_index}",
            layer_ids.len()
        );
        self.invalidate_geometry();
        if scene_was_empty {
            self.fit_view_to_extents();
        }
    }

    pub(crate) fn unload_all_layers_in_project(&mut self, project_index: usize) {
        let layer_ids: Vec<LayerId> = self
            .workspace
            .projects
            .get(project_index)
            .map(|p| p.loaded_layers.iter().copied().collect())
            .unwrap_or_default();
        // Route each through the dirty check so dirty layers prompt the
        // "save first?" dialog (queued) instead of being dropped silently.
        let count = layer_ids.len();
        for id in layer_ids {
            self.try_unload_layer(project_index, id);
        }
        userspace_log!("Requested unload of {count} layer(s) in project {project_index}");
        userspace_warn!("Requested unload of {count} layer(s) from PIDB index {project_index}");
    }

    pub(crate) fn select_all_objects_in_layer(&mut self, project_index: usize, layer_id: LayerId) {
        let Some(project) = self.workspace.projects.get(project_index) else {
            return;
        };
        if !project.loaded_layers.contains(&layer_id) {
            return;
        }
        let handles: Vec<SceneEntityId> = project
            .pidb
            .document
            .objects()
            .iter()
            .filter(|object| object.layer() == layer_id)
            .map(|object| SceneEntityId::Object(object.id()))
            .collect();

        self.workspace.set_active_index(project_index);
        self.history.clear();
        self.editor.active_layer = Some(layer_id);
        self.editor.selected_handles = handles.into_iter().collect();
        self.editor.tri_selected_object_ids.clear();
        self.editor.tri_selected_layer_ids.clear();
        self.editor.canvas_context_menu_open = false;
        let count = self.editor.selected_handles.len();
        userspace_log!("Selected {count} object(s) in layer {:?}", layer_id);
        self.invalidate_geometry();
        self.invalidate_overlay();
    }

    pub(crate) fn unload_layer(&mut self, project_index: usize, layer_id: LayerId) {
        if self.project_layer_is_dirty(project_index, layer_id) {
            let name = self.layer_display_name(project_index, layer_id);
            self.queue_pending_unload(project_index, layer_id, name);
            return;
        }
        let Some(project) = self.workspace.projects.get_mut(project_index) else {
            return;
        };
        let object_handles: Vec<_> = project
            .pidb
            .document
            .objects()
            .iter()
            .filter(|object| object.layer() == layer_id)
            .map(|object| crate::model::SceneEntityId::Object(object.id()))
            .collect();
        if !project.loaded_layers.remove(&layer_id) {
            return;
        }
        self.history.clear();
        for handle in object_handles {
            self.editor.selected_handles.remove(&handle);
            self.editor.hidden_handles.remove(&handle);
            self.editor.frozen_handles.remove(&handle);
            self.editor.translucent_handles.remove(&handle);
        }
        if self.editor.active_layer == Some(layer_id) {
            self.editor.active_layer = None;
        }
        userspace_log!("Unloaded layer {:?} in project {}", layer_id, project_index);
        userspace_log!(
            "Unloaded layer {:?} from PIDB index {project_index}",
            layer_id
        );
        self.invalidate_geometry();
    }
}
