//! Point cloud import, load/unload and explorer commands.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use glam::DVec3;

use crate::{
    app::{App, file_name},
    model::{
        formats::point_cloud::{PointCloudFormat, read_point_cloud},
        point_cloud::{LoadedPointCloud, OpenPointCloud, PointCloudId, prepare_for_render},
    },
    userspace_log, userspace_warn,
};

/// Uniform colour for clouds without per-point colours.
const DEFAULT_POINT_CLOUD_COLOR: [f32; 4] = [0.85, 0.87, 0.9, 1.0];
const DEFAULT_POINT_SIZE: f32 = 2.0;

impl<'a> App<'a> {
    /// Entry point for point-cloud files chosen in the Import menu: register
    /// the path in the session (so it appears in the explorer) and load it.
    pub(crate) fn import_point_cloud_path(&mut self, path: &Path) -> Result<()> {
        if PointCloudFormat::from_path(path).is_none() {
            anyhow::bail!("Unsupported point cloud file: {}", path.display());
        }
        if !path.is_file() {
            anyhow::bail!("Point cloud file does not exist: {}", path.display());
        }
        if !self.point_cloud_files.contains(&path.to_path_buf()) {
            self.point_cloud_files.push(path.to_path_buf());
        }
        self.persist_session();
        self.open_point_cloud_path(path.to_path_buf());
        Ok(())
    }

    /// Decode a point cloud on a background thread; completion is drained by
    /// `poll_point_cloud_loads`.
    pub(crate) fn open_point_cloud_path(&mut self, path: PathBuf) {
        if self.point_clouds.iter().any(|cloud| cloud.path == path) {
            return;
        }
        if self
            .pending_point_cloud_loads
            .iter()
            .any(|(_, pending, _)| *pending == path)
        {
            return;
        }

        let name = file_name(&path);
        let ticket = self.begin_topology_load();

        let (tx, rx) = std::sync::mpsc::channel();
        self.pending_point_cloud_loads
            .push((ticket, path.clone(), rx));
        let window = self.window.clone();
        crate::app::jobs::spawn_pool_task(move || {
            let result =
                crate::app::jobs::run_compute_catching_panic(|| -> Result<LoadedPointCloud> {
                    let data = read_point_cloud(&path).with_context(|| {
                        format!("Failed to read point cloud {}", path.display())
                    })?;
                    if data.points.is_empty() {
                        anyhow::bail!("Point cloud {} contains no points", path.display());
                    }
                    let mut min = DVec3::splat(f64::INFINITY);
                    let mut max = DVec3::splat(f64::NEG_INFINITY);
                    for point in data.points.iter().filter(|point| point.is_finite()) {
                        min = min.min(*point);
                        max = max.max(*point);
                    }
                    if !min.is_finite() || !max.is_finite() {
                        anyhow::bail!("Point cloud {} contains no finite points", path.display());
                    }
                    let prepared =
                        prepare_for_render(&data.points, data.colors.as_deref(), (min, max));
                    Ok(LoadedPointCloud {
                        name,
                        path,
                        points: std::sync::Arc::new(data.points),
                        prepared: std::sync::Arc::new(prepared),
                        bounds: (min, max),
                    })
                });
            let _ = tx.send(result);
            if let Some(window) = window {
                window.request_redraw();
            }
        });
    }

    pub(crate) fn poll_point_cloud_loads(&mut self) {
        let receivers = std::mem::take(&mut self.pending_point_cloud_loads);
        let mut still_pending = Vec::new();
        for (ticket, path, rx) in receivers {
            match rx.try_recv() {
                Ok(Ok(loaded)) => {
                    let should_fit = !self.scene_has_renderables();
                    let id = PointCloudId(self.next_point_cloud_id);
                    self.next_point_cloud_id += 1;
                    userspace_log!(
                        "Loaded point cloud {} ({} points)",
                        loaded.name,
                        loaded.points.len()
                    );
                    self.point_clouds.push(OpenPointCloud {
                        id,
                        name: loaded.name,
                        path: loaded.path,
                        points: loaded.points,
                        prepared: loaded.prepared,
                        bounds: loaded.bounds,
                        visible: true,
                        color: DEFAULT_POINT_CLOUD_COLOR,
                        point_size: DEFAULT_POINT_SIZE,
                    });
                    if should_fit {
                        self.fit_view_to_extents();
                    }
                    self.finish_background_task(ticket, true);
                    // Clouds render from point_cloud_gpu's per-id cache, not
                    // the document scene, so only bounds/redraw are stale.
                    self.invalidate_topology_bounds_and_redraw();
                }
                Ok(Err(error)) => {
                    userspace_warn!("Failed to load point cloud: {error:#}");
                    self.finish_background_task(ticket, false);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    still_pending.push((ticket, path, rx));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    userspace_warn!("Point-cloud loader disconnected for {}", path.display());
                    self.finish_background_task(ticket, false);
                }
            }
        }
        self.pending_point_cloud_loads = still_pending;
    }

    pub(crate) fn toggle_point_cloud_visible(&mut self, id: PointCloudId) {
        let Some(cloud) = self.point_clouds.iter_mut().find(|cloud| cloud.id == id) else {
            return;
        };
        cloud.visible = !cloud.visible;
        self.invalidate_topology_bounds_and_redraw();
    }

    pub(crate) fn close_point_cloud(&mut self, id: PointCloudId) {
        self.point_clouds.retain(|cloud| cloud.id != id);
        self.cancel_jobs(|key| *key == crate::app::jobs::JobKey::PointCloud(id));
        self.invalidate_topology_bounds_and_redraw();
    }

    pub(crate) fn remove_point_cloud(&mut self, path: &Path) {
        let pending = std::mem::take(&mut self.pending_point_cloud_loads);
        for (ticket, pending_path, receiver) in pending {
            if pending_path == path {
                self.cancel_background_task(ticket);
            } else {
                self.pending_point_cloud_loads
                    .push((ticket, pending_path, receiver));
            }
        }
        if let Some(removed_id) = self
            .point_clouds
            .iter()
            .find(|cloud| cloud.path == path)
            .map(|cloud| cloud.id)
        {
            self.cancel_jobs(|key| *key == crate::app::jobs::JobKey::PointCloud(removed_id));
        }
        self.point_clouds.retain(|cloud| cloud.path != path);
        self.point_cloud_files.retain(|existing| existing != path);
        self.persist_session();
        self.invalidate_topology_bounds_and_redraw();
    }

    pub(crate) fn reveal_point_cloud(&mut self, id: PointCloudId) -> Result<()> {
        let path = self
            .point_clouds
            .iter()
            .find(|cloud| cloud.id == id)
            .map(|cloud| cloud.path.clone())
            .context("The point cloud is no longer loaded")?;
        self.reveal_in_file_manager(&path)
    }
}
