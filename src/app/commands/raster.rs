//! Georeferenced raster import, lifetime and triangulation assignment.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::{
    app::App,
    model::raster::{OpenRasterTexture, RasterTextureId, decode_raster, is_supported_raster_path},
    userspace_log,
};

const MAX_RASTER_PREVIEW_DIMENSION: u32 = 4096;

impl<'a> App<'a> {
    pub(crate) fn import_raster_path(&mut self, path: &Path) -> Result<()> {
        if !is_supported_raster_path(path) {
            anyhow::bail!("Unsupported raster file: {}", path.display());
        }
        if !path.is_file() {
            anyhow::bail!("Raster file does not exist: {}", path.display());
        }
        if !self.raster_files.contains(&path.to_path_buf()) {
            self.raster_files.push(path.to_path_buf());
        }
        self.persist_session();
        self.open_raster_path(path.to_path_buf());
        Ok(())
    }

    pub(crate) fn open_raster_path(&mut self, path: PathBuf) {
        if self
            .raster_textures
            .iter()
            .any(|raster| raster.path == path)
            || self
                .pending_raster_loads
                .iter()
                .any(|(_, pending, _)| *pending == path)
        {
            return;
        }

        let ticket = self.begin_topology_load();
        let (tx, rx) = std::sync::mpsc::channel();
        self.pending_raster_loads.push((ticket, path.clone(), rx));
        let window = self.window.clone();
        crate::app::jobs::spawn_pool_task(move || {
            let result = crate::app::jobs::run_compute_catching_panic(|| {
                decode_raster(&path, MAX_RASTER_PREVIEW_DIMENSION)
            });
            let _ = tx.send(result);
            if let Some(window) = window {
                window.request_redraw();
            }
        });
    }

    pub(crate) fn poll_raster_loads(&mut self) {
        let receivers = std::mem::take(&mut self.pending_raster_loads);
        let mut still_pending = Vec::new();
        for (ticket, path, receiver) in receivers {
            match receiver.try_recv() {
                Ok(Ok(loaded)) => {
                    self.finish_background_task(ticket, false);
                    let id = RasterTextureId(self.next_raster_texture_id);
                    self.next_raster_texture_id += 1;
                    userspace_log!(
                        "Loaded raster {} via {} ({}x{}, preview {}x{})",
                        loaded.name,
                        loaded.driver_name,
                        loaded.source_size[0],
                        loaded.source_size[1],
                        loaded.preview_size[0],
                        loaded.preview_size[1]
                    );
                    self.raster_textures.push(OpenRasterTexture {
                        id,
                        name: loaded.name,
                        path: loaded.path,
                        source_size: loaded.source_size,
                        preview_size: loaded.preview_size,
                        rgba: loaded.rgba,
                        world_to_uv: loaded.world_to_uv,
                        projection: loaded.projection,
                        driver_name: loaded.driver_name,
                    });
                    self.redraw_requested = true;
                }
                Ok(Err(error)) => {
                    crate::userspace_warn!("Failed to load raster {}: {error:#}", path.display());
                    self.finish_background_task(ticket, false);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    still_pending.push((ticket, path, receiver));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    crate::userspace_warn!("Raster loader disconnected for {}", path.display());
                    self.finish_background_task(ticket, false);
                }
            }
        }
        self.pending_raster_loads = still_pending;
    }

    pub(crate) fn clear_active_triangulation_raster(&mut self) -> Result<()> {
        let triangulation_id = self
            .active_triangulation
            .context("Select a triangulation first")?;
        let triangulation = self
            .triangulations
            .iter_mut()
            .find(|triangulation| triangulation.id == triangulation_id)
            .context("The active triangulation is no longer loaded")?;
        triangulation.raster_texture = None;
        self.redraw_requested = true;
        Ok(())
    }

    fn cancel_pending_raster_loads(&mut self, path: &Path) {
        let pending = std::mem::take(&mut self.pending_raster_loads);
        for (ticket, pending_path, receiver) in pending {
            if pending_path == path {
                self.cancel_background_task(ticket);
            } else {
                self.pending_raster_loads
                    .push((ticket, pending_path, receiver));
            }
        }
    }

    /// Drop the raster's pixel data but keep its file entry in the explorer.
    pub(crate) fn unload_raster(&mut self, path: &Path) {
        self.cancel_pending_raster_loads(path);
        let unloaded_ids: std::collections::HashSet<_> = self
            .raster_textures
            .iter()
            .filter(|raster| raster.path == path)
            .map(|raster| raster.id)
            .collect();
        self.raster_textures.retain(|raster| raster.path != path);
        for triangulation in &mut self.triangulations {
            if triangulation
                .raster_texture
                .is_some_and(|id| unloaded_ids.contains(&id))
            {
                triangulation.raster_texture = None;
            }
        }
        self.redraw_requested = true;
    }

    pub(crate) fn remove_raster(&mut self, path: &Path) {
        self.cancel_pending_raster_loads(path);
        let removed_ids: std::collections::HashSet<_> = self
            .raster_textures
            .iter()
            .filter(|raster| raster.path == path)
            .map(|raster| raster.id)
            .collect();
        self.raster_textures.retain(|raster| raster.path != path);
        self.raster_files.retain(|existing| existing != path);
        for triangulation in &mut self.triangulations {
            if triangulation
                .raster_texture
                .is_some_and(|id| removed_ids.contains(&id))
            {
                triangulation.raster_texture = None;
            }
        }
        self.persist_session();
        self.redraw_requested = true;
    }

    /// Drape the raster at `path` over every loaded triangulation whose
    /// world-XY extent overlaps the raster footprint. Until draped, a loaded
    /// raster only shows as a flat plan-view image.
    pub(crate) fn drape_raster_over_surfaces(&mut self, path: &Path) -> Result<()> {
        let raster = self
            .raster_textures
            .iter()
            .find(|raster| raster.path == path)
            .context("Load the texture before draping it")?;
        let raster_id = raster.id;
        let raster_name = raster.name.clone();
        let world_to_uv = raster.world_to_uv;
        let mut draped_any = false;
        for triangulation in &mut self.triangulations {
            let bounds = triangulation.mesh.bounds();
            if !raster_overlaps_extent(
                world_to_uv,
                [bounds.min.x, bounds.min.y],
                [bounds.max.x, bounds.max.y],
            ) {
                continue;
            }
            triangulation.raster_texture = Some(raster_id);
            userspace_log!(
                "Draped raster {} over triangulation {} (overlapping extents)",
                raster_name,
                triangulation.name
            );
            draped_any = true;
        }
        if !draped_any {
            anyhow::bail!("No loaded triangulation overlaps the extents of {raster_name}");
        }
        self.redraw_requested = true;
        Ok(())
    }

    /// Remove the raster at `path` from every triangulation it is draped
    /// over, returning it to the flat plan-view image.
    pub(crate) fn undrape_raster(&mut self, path: &Path) {
        let draped_ids: std::collections::HashSet<_> = self
            .raster_textures
            .iter()
            .filter(|raster| raster.path == path)
            .map(|raster| raster.id)
            .collect();
        for triangulation in &mut self.triangulations {
            if triangulation
                .raster_texture
                .is_some_and(|id| draped_ids.contains(&id))
            {
                triangulation.raster_texture = None;
            }
        }
        self.redraw_requested = true;
    }
}

/// Whether a raster's world footprint overlaps the XY extent `[min, max]`.
/// Maps the extent's corners through the affine world-to-UV transform and
/// intersects their UV bounding box with the raster's [0,1]² UV square —
/// conservative for rotated geotransforms, exact for north-up ones.
fn raster_overlaps_extent(world_to_uv: [f64; 6], min: [f64; 2], max: [f64; 2]) -> bool {
    let [a, b, c, d, e, f] = world_to_uv;
    let corners = [
        [min[0], min[1]],
        [max[0], min[1]],
        [min[0], max[1]],
        [max[0], max[1]],
    ];
    let mut uv_min = [f64::INFINITY; 2];
    let mut uv_max = [f64::NEG_INFINITY; 2];
    for [x, y] in corners {
        let u = a * x + b * y + c;
        let v = d * x + e * y + f;
        uv_min = [uv_min[0].min(u), uv_min[1].min(v)];
        uv_max = [uv_max[0].max(u), uv_max[1].max(v)];
    }
    // NaN bounds (empty mesh) fail these comparisons, so nothing is draped.
    uv_min[0] <= 1.0 && uv_max[0] >= 0.0 && uv_min[1] <= 1.0 && uv_max[1] >= 0.0
}
