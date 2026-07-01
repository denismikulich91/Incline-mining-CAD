use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context, Result};
use glam::DVec3;

use crate::{
    app::{App, file_name},
    model::{
        SceneEntityId,
        block_model::{
            BlockModelId, BlockModelSource, ColorStop, ColorTransferFunction, LoadedBlockModel,
            MAX_COLOR_STOPS, MIN_COLOR_STOPS, OpenBlockModel,
        },
        formats::{
            bmf::{self, BmfModel},
            tri00t,
        },
    },
    ui::state::{OreFilterMode, TriSurfaceType},
    userspace_log, userspace_warn,
};

const DEFAULT_BLOCK_MODEL_COLOR: [f32; 4] = [0.2, 0.65, 0.95, 0.42];

impl<'a> App<'a> {
    fn clear_block_model_entity_state(&mut self, handle: SceneEntityId) {
        self.editor.selected_handles.remove(&handle);
        self.editor.hidden_handles.remove(&handle);
        self.editor.frozen_handles.remove(&handle);
        self.editor.translucent_handles.remove(&handle);
    }

    pub(crate) fn import_block_model_source(&mut self, source: BlockModelSource) -> Result<()> {
        if !source.bmf_path.is_file() {
            anyhow::bail!(
                "Block model .bmf does not exist: {}",
                source.bmf_path.display()
            );
        }
        if let Some(path) = &source.bdf_path
            && !path.is_file()
        {
            anyhow::bail!("Block model .bdf does not exist: {}", path.display());
        }
        if let Some(existing) = self
            .block_model_files
            .iter_mut()
            .find(|existing| existing.bmf_path == source.bmf_path)
        {
            *existing = source.clone();
        } else {
            self.block_model_files.push(source.clone());
        }
        userspace_log!("Imported block model source {}", source.bmf_path.display());
        self.persist_session();
        self.open_block_model_source(source)
    }

    pub(crate) fn open_block_model_source(&mut self, source: BlockModelSource) -> Result<()> {
        if self
            .block_models
            .iter()
            .find(|model| model.source.bmf_path == source.bmf_path)
            .is_some()
        {
            self.invalidate_geometry();
            return Ok(());
        }
        if self
            .pending_block_model_loads
            .iter()
            .any(|(pending_source, _)| pending_source.bmf_path == source.bmf_path)
        {
            return Ok(());
        }

        let name = file_name(&source.bmf_path);
        let scene_was_empty = self.triangulations.is_empty()
            && self.block_models.is_empty()
            && self.scene_document.objects().is_empty();
        self.begin_topology_load();

        let (tx, rx) = std::sync::mpsc::channel();
        self.pending_block_model_loads.push((source.clone(), rx));
        let window = self.window.clone();
        std::thread::spawn(move || {
            let result = (|| -> Result<LoadedBlockModel> {
                let model = BmfModel::from_path(&source.bmf_path).with_context(|| {
                    format!("Failed to read block model {}", source.bmf_path.display())
                })?;
                let bdf = source
                    .bdf_path
                    .as_ref()
                    .map(bmf::parse_bdf)
                    .transpose()
                    .with_context(|| "Failed to read companion .bdf")?;
                let renderable_block_indices = model.renderable_block_indices()?;
                let blocks = model.block_bounds()?;
                if !model.has_verified_rotation() {
                    userspace_warn!(
                        "Block model {} has non-zero dip/plunge; only the bearing rotation is \
                         verified against real Vulcan files, so tilted geometry may be placed \
                         incorrectly",
                        source.bmf_path.display()
                    );
                }
                let unsupported = model.unsupported_variables();
                if !unsupported.is_empty() {
                    let names = unsupported
                        .iter()
                        .map(|variable| format!("{} ({})", variable.name, variable.physical_type))
                        .collect::<Vec<_>>()
                        .join(", ");
                    userspace_warn!(
                        "Block model {} has {} variable(s) of an unsupported type that won't be \
                         readable: {names}",
                        source.bmf_path.display(),
                        unsupported.len()
                    );
                }
                Ok(LoadedBlockModel {
                    name,
                    source,
                    model,
                    bdf,
                    blocks,
                    renderable_block_indices,
                    scene_was_empty,
                })
            })();
            let _ = tx.send(result);
            if let Some(window) = window {
                window.request_redraw();
            }
        });
        Ok(())
    }

    pub(crate) fn poll_block_model_loads(&mut self) {
        let receivers = std::mem::take(&mut self.pending_block_model_loads);
        let mut still_pending = Vec::new();
        for (source, rx) in receivers {
            match rx.try_recv() {
                Ok(Ok(loaded)) => {
                    self.pending_loads = self.pending_loads.saturating_sub(1);
                    let id = BlockModelId(self.next_block_model_id);
                    self.next_block_model_id += 1;
                    let first_numeric = preferred_block_model_color_variable(&loaded.model);
                    self.block_models.push(OpenBlockModel {
                        id,
                        name: loaded.name,
                        source: loaded.source.clone(),
                        model: loaded.model,
                        bdf: loaded.bdf,
                        blocks: loaded.blocks,
                        renderable_block_indices: loaded.renderable_block_indices,
                        visible: true,
                        color: DEFAULT_BLOCK_MODEL_COLOR,
                        active_numeric_variable: first_numeric,
                        color_transfer: ColorTransferFunction::default(),
                    });
                    if !self.block_model_files.contains(&loaded.source) {
                        self.block_model_files.push(loaded.source);
                    }
                    if loaded.scene_was_empty {
                        self.fit_view_to_extents();
                    }
                    self.topology_load_pending_gpu = true;
                    self.persist_session();
                    self.invalidate_geometry();
                }
                Ok(Err(error)) => {
                    self.pending_loads = self.pending_loads.saturating_sub(1);
                    userspace_warn!("Failed to load block model: {error:#}");
                    self.finish_topology_load();
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => still_pending.push((source, rx)),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.pending_loads = self.pending_loads.saturating_sub(1);
                    self.finish_topology_load();
                }
            }
        }
        self.pending_block_model_loads = still_pending;
    }

    pub(crate) fn toggle_block_model_visible(&mut self, id: BlockModelId) {
        let Some(model) = self.block_models.iter_mut().find(|model| model.id == id) else {
            return;
        };
        model.visible = !model.visible;
        self.invalidate_geometry();
    }

    pub(crate) fn set_block_model_color_variable(&mut self, id: BlockModelId, variable: String) {
        let Some(model) = self.block_models.iter_mut().find(|model| model.id == id) else {
            return;
        };
        if model.active_numeric_variable.as_deref() != Some(variable.as_str()) {
            model.active_numeric_variable = Some(variable);
            model.color_transfer = ColorTransferFunction::default();
            self.invalidate_geometry();
        }
    }

    pub(crate) fn set_block_model_color_stops(&mut self, id: BlockModelId, stops: Vec<ColorStop>) {
        let Some(model) = self.block_models.iter_mut().find(|model| model.id == id) else {
            return;
        };
        model.color_transfer.stops = normalized_color_stops(stops);
        self.invalidate_geometry();
    }

    pub(crate) fn close_block_model(&mut self, id: BlockModelId) {
        let Some(index) = self.block_models.iter().position(|model| model.id == id) else {
            return;
        };
        let model = self.block_models.remove(index);
        self.clear_block_model_entity_state(model.entity_id());
        self.editor.block_model_table_pages.remove(&id);
        self.editor.block_model_variable_cache.remove(&id);
        if self.active_block_model == Some(id) {
            self.active_block_model = None;
        }
        self.invalidate_geometry();
    }

    pub(crate) fn remove_block_model(&mut self, source: BlockModelSource) {
        let ids: Vec<_> = self
            .block_models
            .iter()
            .filter(|model| model.source.bmf_path == source.bmf_path)
            .map(|model| model.id)
            .collect();
        for id in ids {
            self.close_block_model(id);
        }
        self.block_model_files
            .retain(|existing| existing.bmf_path != source.bmf_path);
        self.persist_session();
        self.invalidate_geometry();
    }

    pub(crate) fn set_block_model_definition(
        &mut self,
        id: BlockModelId,
        bdf_path: PathBuf,
    ) -> Result<()> {
        let Some(model) = self.block_models.iter_mut().find(|model| model.id == id) else {
            anyhow::bail!("The selected block model is no longer loaded");
        };
        let bdf = bmf::parse_bdf(&bdf_path)?;
        model.source.bdf_path = Some(bdf_path.clone());
        model.bdf = Some(bdf);
        if let Some(source) = self
            .block_model_files
            .iter_mut()
            .find(|source| source.bmf_path == model.source.bmf_path)
        {
            source.bdf_path = Some(bdf_path);
        }
        self.persist_session();
        Ok(())
    }

    pub(crate) fn set_block_model_source_definition(
        &mut self,
        mut source: BlockModelSource,
        bdf_path: PathBuf,
    ) -> Result<()> {
        bmf::parse_bdf(&bdf_path)?;
        source.bdf_path = Some(bdf_path.clone());
        if let Some(existing) = self
            .block_model_files
            .iter_mut()
            .find(|existing| existing.bmf_path == source.bmf_path)
        {
            existing.bdf_path = Some(bdf_path);
        } else {
            self.block_model_files.push(source);
        }
        self.persist_session();
        Ok(())
    }

    pub(crate) fn create_ore_triangulation(
        &mut self,
        block_model_id: BlockModelId,
        variable: String,
        mode: OreFilterMode,
        min: f64,
        max: f64,
        name: String,
    ) -> Result<()> {
        let model = self
            .block_models
            .iter()
            .find(|model| model.id == block_model_id)
            .context("The selected block model is no longer loaded")?;
        let values = model.model.numeric_values(&variable)?;
        let selected: Vec<_> = values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                let keep = match mode {
                    OreFilterMode::GreaterOrEqual => *value >= min,
                    OreFilterMode::LessOrEqual => *value <= min,
                    OreFilterMode::Between => *value >= min.min(max) && *value <= min.max(max),
                };
                keep.then_some(index)
            })
            .collect();
        if selected.is_empty() {
            anyhow::bail!("No blocks match the ore filter");
        }
        let (vertices, faces) = boundary_mesh_from_blocks(model, &selected)?;
        self.finish_generated_triangulation(name, vertices, faces, TriSurfaceType::SolidClosed)
    }
}

fn preferred_block_model_color_variable(
    model: &crate::model::formats::bmf::BmfModel,
) -> Option<String> {
    let variables = model.numeric_variables();
    variables
        .iter()
        .find(|variable| !variable.special)
        .or_else(|| variables.first())
        .map(|variable| variable.name.clone())
}

fn normalized_color_stops(mut stops: Vec<ColorStop>) -> Vec<ColorStop> {
    if stops.len() < MIN_COLOR_STOPS {
        stops = ColorTransferFunction::default().stops;
    }
    for stop in &mut stops {
        stop.t = stop.t.clamp(0.0, 1.0);
        for channel in &mut stop.color {
            *channel = channel.clamp(0.0, 1.0);
        }
    }
    stops.sort_by(|a, b| a.t.total_cmp(&b.t));
    stops.dedup_by(|a, b| (a.t - b.t).abs() < 1e-4);
    if stops.len() < MIN_COLOR_STOPS {
        stops = ColorTransferFunction::default().stops;
    }
    stops.truncate(MAX_COLOR_STOPS);
    stops
}

fn boundary_mesh_from_blocks(
    model: &OpenBlockModel,
    selected: &[usize],
) -> Result<(Vec<tri00t::Vertex>, Vec<[u32; 3]>)> {
    let selected_set: HashSet<_> = selected.iter().copied().collect();
    let mut key_to_index = HashMap::new();
    for &index in selected {
        let block = model.blocks[index];
        key_to_index.insert(block_key(block), index);
    }

    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut vertex_map: HashMap<[u64; 3], u32> = HashMap::new();
    for &index in selected {
        let block = model.blocks[index];
        let size = block.upper - block.lower;
        let neighbours = [
            DVec3::new(-size.x, 0.0, 0.0),
            DVec3::new(size.x, 0.0, 0.0),
            DVec3::new(0.0, -size.y, 0.0),
            DVec3::new(0.0, size.y, 0.0),
            DVec3::new(0.0, 0.0, -size.z),
            DVec3::new(0.0, 0.0, size.z),
        ];
        for (face_index, delta) in neighbours.into_iter().enumerate() {
            let neighbour = block_key(crate::model::block_model::BlockBounds {
                lower: block.lower + delta,
                upper: block.upper + delta,
            });
            if key_to_index
                .get(&neighbour)
                .is_some_and(|neighbour_index| selected_set.contains(neighbour_index))
            {
                continue;
            }
            add_block_face(
                model,
                block,
                face_index,
                &mut vertices,
                &mut faces,
                &mut vertex_map,
            )?;
        }
    }
    Ok((vertices, faces))
}

fn add_block_face(
    model: &OpenBlockModel,
    block: crate::model::block_model::BlockBounds,
    face_index: usize,
    vertices: &mut Vec<tri00t::Vertex>,
    faces: &mut Vec<[u32; 3]>,
    vertex_map: &mut HashMap<[u64; 3], u32>,
) -> Result<()> {
    let lo = block.lower;
    let hi = block.upper;
    let corners = [
        DVec3::new(lo.x, lo.y, lo.z),
        DVec3::new(hi.x, lo.y, lo.z),
        DVec3::new(hi.x, hi.y, lo.z),
        DVec3::new(lo.x, hi.y, lo.z),
        DVec3::new(lo.x, lo.y, hi.z),
        DVec3::new(hi.x, lo.y, hi.z),
        DVec3::new(hi.x, hi.y, hi.z),
        DVec3::new(lo.x, hi.y, hi.z),
    ];
    let quads: [[usize; 4]; 6] = [
        [0, 3, 7, 4],
        [1, 5, 6, 2],
        [0, 4, 5, 1],
        [3, 2, 6, 7],
        [0, 1, 2, 3],
        [4, 7, 6, 5],
    ];
    let quad = quads[face_index];
    let mut indices = [0u32; 4];
    for (dst, corner_index) in indices.iter_mut().zip(quad) {
        let world = model.model.local_to_world(corners[corner_index]);
        let key = point_key(world);
        *dst = if let Some(index) = vertex_map.get(&key) {
            *index
        } else {
            let index = u32::try_from(vertices.len())
                .map_err(|_| anyhow::anyhow!("Ore triangulation has too many vertices"))?;
            vertices.push(tri00t::Vertex::new(world.x, world.y, world.z));
            vertex_map.insert(key, index);
            index
        };
    }
    faces.push([indices[0], indices[1], indices[2]]);
    faces.push([indices[0], indices[2], indices[3]]);
    Ok(())
}

fn block_key(block: crate::model::block_model::BlockBounds) -> [u64; 6] {
    [
        quantize(block.lower.x),
        quantize(block.lower.y),
        quantize(block.lower.z),
        quantize(block.upper.x),
        quantize(block.upper.y),
        quantize(block.upper.z),
    ]
}

fn point_key(point: DVec3) -> [u64; 3] {
    [quantize(point.x), quantize(point.y), quantize(point.z)]
}

fn quantize(value: f64) -> u64 {
    (value * 1_000_000.0).round().to_bits()
}
