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
            BlockBounds, BlockModelId, BlockModelSource, ColorStop, ColorTransferFunction,
            LoadedBlockModel, MAX_COLOR_STOPS, MIN_COLOR_STOPS, OpenBlockModel,
            compute_world_bounds, is_no_data_sentinel, numeric_variable_default,
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
            .any(|(_, pending_source, _)| pending_source.bmf_path == source.bmf_path)
        {
            return Ok(());
        }

        let name = file_name(&source.bmf_path);
        let ticket = self.begin_topology_load();

        let (tx, rx) = std::sync::mpsc::channel();
        self.pending_block_model_loads
            .push((ticket, source.clone(), rx));
        let window = self.window.clone();
        crate::app::jobs::spawn_pool_task(move || {
            let result = crate::app::jobs::run_compute_catching_panic(
                || -> Result<LoadedBlockModel> {
                    let model = BmfModel::from_path(&source.bmf_path).with_context(|| {
                        format!("Failed to read block model {}", source.bmf_path.display())
                    })?;
                    let bdf = source
                        .bdf_path
                        .as_ref()
                        .map(bmf::parse_bdf)
                        .transpose()
                        .with_context(|| "Failed to read companion .bdf")?;
                    let renderable_block_indices =
                        std::sync::Arc::new(model.renderable_block_indices()?);
                    let blocks = std::sync::Arc::new(model.block_bounds()?);
                    // Computed once here (off the UI thread) so the per-frame
                    // transparency sort and scene-bounds queries never re-walk
                    // every block's rotated corners.
                    let world_bounds =
                        compute_world_bounds(&model, &blocks, &renderable_block_indices);
                    let active_numeric_variable = preferred_block_model_color_variable(&model);
                    let active_values_cache = OpenBlockModel::prepare_active_values_cache(
                        &model,
                        &renderable_block_indices,
                        active_numeric_variable.as_deref(),
                    );
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
                        // Debug-format the type so padding or unusual characters
                        // in an unrecognized type string show up in the log.
                        let names = unsupported
                            .iter()
                            .map(|variable| {
                                format!("{} ({:?})", variable.name, variable.physical_type)
                            })
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
                        world_bounds,
                        active_numeric_variable,
                        active_values_cache,
                    })
                },
            );
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
        for (ticket, source, rx) in receivers {
            match rx.try_recv() {
                Ok(Ok(loaded)) => {
                    let should_fit = !self.scene_has_renderables();
                    let id = BlockModelId(self.next_block_model_id);
                    self.next_block_model_id += 1;
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
                        active_numeric_variable: loaded.active_numeric_variable,
                        color_transfer: ColorTransferFunction::default(),
                        hide_empty_color_values: true,
                        active_values_cache: loaded.active_values_cache,
                        world_bounds: loaded.world_bounds,
                    });
                    if !self.block_model_files.contains(&loaded.source) {
                        self.block_model_files.push(loaded.source);
                    }
                    if should_fit {
                        self.fit_view_to_extents();
                    }
                    self.finish_background_task(ticket, true);
                    self.persist_session();
                    // The model renders from block_model_gpu's per-id cache, not
                    // the document scene. A full invalidate_geometry here would
                    // re-upload every vector object per loaded model.
                    self.invalidate_topology_bounds_and_redraw();
                }
                Ok(Err(error)) => {
                    userspace_warn!("Failed to load block model: {error:#}");
                    self.finish_background_task(ticket, false);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    still_pending.push((ticket, source, rx));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    userspace_warn!(
                        "Block model loader disconnected for {}",
                        source.bmf_path.display()
                    );
                    self.finish_background_task(ticket, false);
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
        self.invalidate_topology_bounds_and_redraw();
    }

    pub(crate) fn set_block_model_color_variable(&mut self, id: BlockModelId, variable: String) {
        let Some((model_data, renderable_block_indices)) = self
            .block_models
            .iter_mut()
            .find(|model| model.id == id)
            .and_then(|model| {
                (model.active_numeric_variable.as_deref() != Some(variable.as_str())).then(|| {
                    model.active_numeric_variable = Some(variable.clone());
                    model.color_transfer = ColorTransferFunction::default();
                    model.begin_active_values_decode(&variable);
                    (
                        model.model.clone(),
                        std::sync::Arc::clone(&model.renderable_block_indices),
                    )
                })
            })
        else {
            return;
        };
        self.request_topology_redraw();

        let requested_variable = variable.clone();
        let compute = move |cancel: &crate::app::jobs::CancelFlag| {
            if cancel.is_cancelled() {
                anyhow::bail!("Cancelled");
            }
            let prepared = OpenBlockModel::prepare_active_values_cache(
                &model_data,
                &renderable_block_indices,
                Some(&requested_variable),
            );
            Ok((requested_variable, prepared))
        };
        let apply = move |app: &mut App,
                          result: Result<(
            String,
            crate::model::block_model::ActiveValuesCache,
        )>| {
            match result {
                Ok((decoded_variable, prepared)) => {
                    if let Some(model) = app.block_models.iter().find(|model| model.id == id)
                        && model.active_numeric_variable.as_deref()
                            == Some(decoded_variable.as_str())
                    {
                        model.install_active_values_cache(prepared);
                        app.request_topology_redraw();
                    }
                }
                Err(error) if error.to_string() != "Cancelled" => {
                    crate::userspace_warn!(
                        "Could not decode block-model colour variable '{variable}': {error:#}"
                    );
                }
                Err(_) => {}
            }
        };
        self.spawn_job(
            "Decoding block-model colour variable…",
            vec![crate::app::jobs::JobKey::BlockModel(id)],
            compute,
            apply,
        );
    }

    pub(crate) fn set_block_model_color_stops(&mut self, id: BlockModelId, stops: Vec<ColorStop>) {
        let Some(model) = self.block_models.iter_mut().find(|model| model.id == id) else {
            return;
        };
        model.color_transfer.stops = normalized_color_stops(stops);
        self.request_topology_redraw();
    }

    pub(crate) fn set_block_model_hide_empty_values(&mut self, id: BlockModelId, hide: bool) {
        let Some(model) = self.block_models.iter_mut().find(|model| model.id == id) else {
            return;
        };
        if model.hide_empty_color_values != hide {
            model.hide_empty_color_values = hide;
            self.request_topology_redraw();
        }
    }

    pub(crate) fn close_block_model(&mut self, id: BlockModelId) {
        let Some(index) = self.block_models.iter().position(|model| model.id == id) else {
            return;
        };
        let model = self.block_models.remove(index);
        self.cancel_jobs(|key| *key == crate::app::jobs::JobKey::BlockModel(id));
        self.clear_block_model_entity_state(model.entity_id());
        self.editor.block_model_table_pages.remove(&id);
        self.editor.viewport_block_model_id = self
            .editor
            .viewport_block_model_id
            .filter(|active| *active != id);
        self.editor
            .block_model_variable_ranges
            .retain(|(model_id, _), _| *model_id != id);
        if self.active_block_model == Some(id) {
            self.active_block_model = None;
        }
        self.invalidate_topology_bounds_and_redraw();
    }

    pub(crate) fn remove_block_model(&mut self, source: BlockModelSource) {
        let pending = std::mem::take(&mut self.pending_block_model_loads);
        for (ticket, pending_source, receiver) in pending {
            if pending_source.bmf_path == source.bmf_path {
                self.cancel_background_task(ticket);
            } else {
                self.pending_block_model_loads
                    .push((ticket, pending_source, receiver));
            }
        }
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
        self.request_topology_redraw();
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
        // BmfModel is Arc-backed and the two large geometry arrays are Arcs,
        // so this snapshot is constant-time. Variable decoding, filtering,
        // boundary extraction, mesh construction and BVH building all stay
        // off the UI thread.
        let model_data = model.model.clone();
        let blocks = std::sync::Arc::clone(&model.blocks);
        let renderable_block_indices = std::sync::Arc::clone(&model.renderable_block_indices);
        let model_name = model.name.clone();
        let compute = move |cancel: &crate::app::jobs::CancelFlag|
              -> Result<crate::model::triangulation::GeneratedTriangulationLog> {
            if cancel.is_cancelled() {
                anyhow::bail!("Cancelled");
            }
            let values = model_data.numeric_values(&variable)?;
            let default = model_data
                .variable(&variable)
                .and_then(numeric_variable_default);
            let selected = ore_block_indices(
                &values,
                &renderable_block_indices,
                default,
                mode,
                min,
                max,
            );
            if selected.is_empty() {
                anyhow::bail!("No blocks match the ore filter");
            }
            let (vertices, faces) =
                boundary_mesh_from_blocks(&model_data, &blocks, &selected, cancel)?;
            if cancel.is_cancelled() {
                anyhow::bail!("Cancelled");
            }
            let generated = crate::app::commands::triangulation::session::build_generated_triangulation(
                name,
                vertices,
                faces,
                TriSurfaceType::SolidClosed,
                crate::model::triangulation::unique_edges,
            )?;
            Ok(crate::model::triangulation::GeneratedTriangulationLog {
                generated,
                message: format!("Generated ore mesh from block model '{model_name}'"),
            })
        };
        let apply = move |app: &mut App,
                          result: Result<
            crate::model::triangulation::GeneratedTriangulationLog,
        >| {
            app.apply_generated_triangulation_job(result);
        };
        self.spawn_job(
            "Building ore mesh…",
            vec![crate::app::jobs::JobKey::BlockModel(block_model_id)],
            compute,
            apply,
        );
        Ok(())
    }
}

fn ore_block_indices(
    values: &[f64],
    renderable_block_indices: &[usize],
    default: Option<f64>,
    mode: OreFilterMode,
    min: f64,
    max: f64,
) -> Vec<usize> {
    renderable_block_indices
        .iter()
        .copied()
        .filter(|index| {
            let Some(&value) = values.get(*index) else {
                return false;
            };
            if !value.is_finite()
                || default.is_some_and(|default| (value - default).abs() < 1e-8)
                || is_no_data_sentinel(value)
            {
                return false;
            }
            match mode {
                OreFilterMode::GreaterOrEqual => value >= min,
                OreFilterMode::LessOrEqual => value <= min,
                OreFilterMode::Between => value >= min.min(max) && value <= min.max(max),
            }
        })
        .collect()
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
    model: &BmfModel,
    blocks: &[BlockBounds],
    selected: &[usize],
    cancel: &crate::app::jobs::CancelFlag,
) -> Result<(Vec<tri00t::Vertex>, Vec<[u32; 3]>)> {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut vertex_map: HashMap<[u64; 3], u32> = HashMap::new();
    for (index, quad) in exterior_block_face_tiles(blocks, selected)?
        .into_iter()
        .enumerate()
    {
        if index.is_multiple_of(4096) && cancel.is_cancelled() {
            anyhow::bail!("Cancelled");
        }
        add_block_face_tile(model, quad, &mut vertices, &mut faces, &mut vertex_map)?;
    }
    Ok((vertices, faces))
}

#[derive(Clone, Copy, Debug)]
struct FaceRect {
    plane: f64,
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
}

#[derive(Clone, Copy, Debug)]
struct CoveredRect {
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
}

#[derive(Clone, Copy, Debug)]
struct FaceCandidate {
    block_index: usize,
    rect: CoveredRect,
}

#[derive(Debug)]
enum FaceRectBvh {
    Leaf {
        bounds: CoveredRect,
        entries: Vec<FaceCandidate>,
    },
    Branch {
        bounds: CoveredRect,
        left: Box<FaceRectBvh>,
        right: Box<FaceRectBvh>,
    },
}

impl FaceRectBvh {
    fn build(mut entries: Vec<FaceCandidate>) -> Self {
        const LEAF_SIZE: usize = 8;
        let bounds = candidate_bounds(&entries);
        if entries.len() <= LEAF_SIZE {
            return Self::Leaf { bounds, entries };
        }

        let mut center_min = [f64::INFINITY; 2];
        let mut center_max = [f64::NEG_INFINITY; 2];
        for entry in &entries {
            let center = [
                entry.rect.u_min * 0.5 + entry.rect.u_max * 0.5,
                entry.rect.v_min * 0.5 + entry.rect.v_max * 0.5,
            ];
            for axis in 0..2 {
                center_min[axis] = center_min[axis].min(center[axis]);
                center_max[axis] = center_max[axis].max(center[axis]);
            }
        }
        let axis = usize::from(center_max[1] - center_min[1] > center_max[0] - center_min[0]);
        entries.sort_by(|a, b| {
            let a_center = if axis == 0 {
                a.rect.u_min * 0.5 + a.rect.u_max * 0.5
            } else {
                a.rect.v_min * 0.5 + a.rect.v_max * 0.5
            };
            let b_center = if axis == 0 {
                b.rect.u_min * 0.5 + b.rect.u_max * 0.5
            } else {
                b.rect.v_min * 0.5 + b.rect.v_max * 0.5
            };
            a_center.total_cmp(&b_center)
        });
        let right_entries = entries.split_off(entries.len() / 2);
        Self::Branch {
            bounds,
            left: Box::new(Self::build(entries)),
            right: Box::new(Self::build(right_entries)),
        }
    }

    fn query(&self, query: CoveredRect, matches: &mut Vec<FaceCandidate>) {
        match self {
            Self::Leaf { bounds, entries } => {
                if !rects_overlap(*bounds, query) {
                    return;
                }
                matches.extend(
                    entries
                        .iter()
                        .copied()
                        .filter(|entry| rects_overlap(entry.rect, query)),
                );
            }
            Self::Branch {
                bounds,
                left,
                right,
            } => {
                if !rects_overlap(*bounds, query) {
                    return;
                }
                left.query(query, matches);
                right.query(query, matches);
            }
        }
    }
}

fn candidate_bounds(entries: &[FaceCandidate]) -> CoveredRect {
    entries.iter().fold(
        CoveredRect {
            u_min: f64::INFINITY,
            u_max: f64::NEG_INFINITY,
            v_min: f64::INFINITY,
            v_max: f64::NEG_INFINITY,
        },
        |bounds, entry| CoveredRect {
            u_min: bounds.u_min.min(entry.rect.u_min),
            u_max: bounds.u_max.max(entry.rect.u_max),
            v_min: bounds.v_min.min(entry.rect.v_min),
            v_max: bounds.v_max.max(entry.rect.v_max),
        },
    )
}

fn rects_overlap(a: CoveredRect, b: CoveredRect) -> bool {
    a.u_min < b.u_max && b.u_min < a.u_max && a.v_min < b.v_max && b.v_min < a.v_max
}

/// Tiles only the exposed portion of every selected block face. Plane maps
/// find blocks on the opposite side of each face; subdividing at all overlap
/// edges handles one-large-to-many-small sub-block adjacency without leaving
/// coplanar internal partitions in the generated solid.
fn exterior_block_face_tiles(
    blocks: &[BlockBounds],
    selected: &[usize],
) -> Result<Vec<[DVec3; 4]>> {
    let mut seen = HashSet::new();
    let mut selected_indices = Vec::with_capacity(selected.len());
    for &index in selected {
        let block = blocks
            .get(index)
            .with_context(|| format!("Ore block index {index} is out of range"))?;
        if !block.lower.is_finite()
            || !block.upper.is_finite()
            || !(block.lower.cmplt(block.upper).all())
        {
            anyhow::bail!("Ore block {index} has invalid bounds");
        }
        if seen.insert(index) {
            selected_indices.push(index);
        }
    }

    // For each face direction, index the boundary plane of a potential block
    // on the opposite side: upper X blocks cover -X faces, lower X blocks
    // cover +X faces, and likewise for Y/Z.
    let mut plane_entries: [HashMap<u64, Vec<FaceCandidate>>; 6] =
        std::array::from_fn(|_| HashMap::new());
    for &index in &selected_indices {
        let block = blocks[index];
        let planes = [
            block.upper.x,
            block.lower.x,
            block.upper.y,
            block.lower.y,
            block.upper.z,
            block.lower.z,
        ];
        for (face_index, plane) in planes.into_iter().enumerate() {
            let face = block_face_rect(block, face_index);
            plane_entries[face_index]
                .entry(quantize(plane))
                .or_default()
                .push(FaceCandidate {
                    block_index: index,
                    rect: CoveredRect {
                        u_min: face.u_min,
                        u_max: face.u_max,
                        v_min: face.v_min,
                        v_max: face.v_max,
                    },
                });
        }
    }
    let opposite_planes = plane_entries.map(|groups| {
        groups
            .into_iter()
            .map(|(plane, entries)| (plane, FaceRectBvh::build(entries)))
            .collect::<HashMap<_, _>>()
    });

    let mut exposed_planes: [HashMap<u64, (f64, Vec<CoveredRect>)>; 6] =
        std::array::from_fn(|_| HashMap::new());
    for &index in &selected_indices {
        let block = blocks[index];
        for (face_index, plane_index) in opposite_planes.iter().enumerate() {
            let face = block_face_rect(block, face_index);
            let mut covered = Vec::new();
            if let Some(indexed_faces) = plane_index.get(&quantize(face.plane)) {
                let query = CoveredRect {
                    u_min: face.u_min,
                    u_max: face.u_max,
                    v_min: face.v_min,
                    v_max: face.v_max,
                };
                let mut candidates = Vec::new();
                indexed_faces.query(query, &mut candidates);
                for candidate in candidates {
                    if candidate.block_index == index {
                        continue;
                    }
                    let overlap = CoveredRect {
                        u_min: face.u_min.max(candidate.rect.u_min),
                        u_max: face.u_max.min(candidate.rect.u_max),
                        v_min: face.v_min.max(candidate.rect.v_min),
                        v_max: face.v_max.min(candidate.rect.v_max),
                    };
                    if overlap.u_min < overlap.u_max && overlap.v_min < overlap.v_max {
                        covered.push(overlap);
                    }
                }
            }
            exposed_planes[face_index]
                .entry(quantize(face.plane))
                .or_insert_with(|| (face.plane, Vec::new()))
                .1
                .extend(uncovered_face_rects(face, &covered));
        }
    }
    let mut tiles = Vec::new();
    for (face_index, planes) in exposed_planes.into_iter().enumerate() {
        for (_, (plane, exposed)) in planes {
            tiles.extend(conforming_rect_tiles(&exposed).into_iter().map(|rect| {
                face_tile(
                    plane, rect.u_min, rect.u_max, rect.v_min, rect.v_max, face_index,
                )
            }));
        }
    }
    Ok(tiles)
}

fn block_face_rect(block: BlockBounds, face_index: usize) -> FaceRect {
    match face_index {
        0 => FaceRect {
            plane: block.lower.x,
            u_min: block.lower.y,
            u_max: block.upper.y,
            v_min: block.lower.z,
            v_max: block.upper.z,
        },
        1 => FaceRect {
            plane: block.upper.x,
            u_min: block.lower.y,
            u_max: block.upper.y,
            v_min: block.lower.z,
            v_max: block.upper.z,
        },
        2 => FaceRect {
            plane: block.lower.y,
            u_min: block.lower.x,
            u_max: block.upper.x,
            v_min: block.lower.z,
            v_max: block.upper.z,
        },
        3 => FaceRect {
            plane: block.upper.y,
            u_min: block.lower.x,
            u_max: block.upper.x,
            v_min: block.lower.z,
            v_max: block.upper.z,
        },
        4 => FaceRect {
            plane: block.lower.z,
            u_min: block.lower.x,
            u_max: block.upper.x,
            v_min: block.lower.y,
            v_max: block.upper.y,
        },
        _ => FaceRect {
            plane: block.upper.z,
            u_min: block.lower.x,
            u_max: block.upper.x,
            v_min: block.lower.y,
            v_max: block.upper.y,
        },
    }
}

fn uncovered_face_rects(face: FaceRect, covered: &[CoveredRect]) -> Vec<CoveredRect> {
    let mut uncovered = vec![CoveredRect {
        u_min: face.u_min,
        u_max: face.u_max,
        v_min: face.v_min,
        v_max: face.v_max,
    }];
    for cover in covered {
        let mut next = Vec::new();
        for tile in uncovered {
            subtract_covered_rect(tile, *cover, &mut next);
        }
        uncovered = next;
        if uncovered.is_empty() {
            break;
        }
    }
    uncovered
}

fn subtract_covered_rect(tile: CoveredRect, cover: CoveredRect, output: &mut Vec<CoveredRect>) {
    let intersection = CoveredRect {
        u_min: tile.u_min.max(cover.u_min),
        u_max: tile.u_max.min(cover.u_max),
        v_min: tile.v_min.max(cover.v_min),
        v_max: tile.v_max.min(cover.v_max),
    };
    if intersection.u_min >= intersection.u_max || intersection.v_min >= intersection.v_max {
        output.push(tile);
        return;
    }

    if tile.u_min < intersection.u_min {
        output.push(CoveredRect {
            u_min: tile.u_min,
            u_max: intersection.u_min,
            v_min: tile.v_min,
            v_max: tile.v_max,
        });
    }
    if intersection.u_max < tile.u_max {
        output.push(CoveredRect {
            u_min: intersection.u_max,
            u_max: tile.u_max,
            v_min: tile.v_min,
            v_max: tile.v_max,
        });
    }
    if tile.v_min < intersection.v_min {
        output.push(CoveredRect {
            u_min: intersection.u_min,
            u_max: intersection.u_max,
            v_min: tile.v_min,
            v_max: intersection.v_min,
        });
    }
    if intersection.v_max < tile.v_max {
        output.push(CoveredRect {
            u_min: intersection.u_min,
            u_max: intersection.u_max,
            v_min: intersection.v_max,
            v_max: tile.v_max,
        });
    }
}

/// Retile a planar union on its complete endpoint grid. Splitting a large
/// exterior rectangle at neighboring sub-block corners gives the final mesh
/// matching edges instead of topological T-junctions.
fn conforming_rect_tiles(rects: &[CoveredRect]) -> Vec<CoveredRect> {
    let mut u_cuts = Vec::with_capacity(rects.len() * 2);
    let mut v_cuts = Vec::with_capacity(rects.len() * 2);
    for rect in rects {
        u_cuts.extend([rect.u_min, rect.u_max]);
        v_cuts.extend([rect.v_min, rect.v_max]);
    }
    u_cuts.sort_by(f64::total_cmp);
    v_cuts.sort_by(f64::total_cmp);
    u_cuts.dedup_by(|a, b| *a == *b);
    v_cuts.dedup_by(|a, b| *a == *b);

    let mut cells = HashSet::<(usize, usize)>::new();
    for rect in rects {
        let u_start = u_cuts
            .binary_search_by(|value| value.total_cmp(&rect.u_min))
            .expect("rectangle endpoint was inserted");
        let u_end = u_cuts
            .binary_search_by(|value| value.total_cmp(&rect.u_max))
            .expect("rectangle endpoint was inserted");
        let v_start = v_cuts
            .binary_search_by(|value| value.total_cmp(&rect.v_min))
            .expect("rectangle endpoint was inserted");
        let v_end = v_cuts
            .binary_search_by(|value| value.total_cmp(&rect.v_max))
            .expect("rectangle endpoint was inserted");
        for u in u_start..u_end {
            for v in v_start..v_end {
                cells.insert((u, v));
            }
        }
    }
    let mut cells = cells.into_iter().collect::<Vec<_>>();
    cells.sort_unstable();
    cells
        .into_iter()
        .map(|(u, v)| CoveredRect {
            u_min: u_cuts[u],
            u_max: u_cuts[u + 1],
            v_min: v_cuts[v],
            v_max: v_cuts[v + 1],
        })
        .collect()
}

fn face_tile(
    plane: f64,
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
    face_index: usize,
) -> [DVec3; 4] {
    match face_index {
        0 => [
            DVec3::new(plane, u_min, v_min),
            DVec3::new(plane, u_max, v_min),
            DVec3::new(plane, u_max, v_max),
            DVec3::new(plane, u_min, v_max),
        ],
        1 => [
            DVec3::new(plane, u_min, v_min),
            DVec3::new(plane, u_min, v_max),
            DVec3::new(plane, u_max, v_max),
            DVec3::new(plane, u_max, v_min),
        ],
        2 => [
            DVec3::new(u_min, plane, v_min),
            DVec3::new(u_min, plane, v_max),
            DVec3::new(u_max, plane, v_max),
            DVec3::new(u_max, plane, v_min),
        ],
        3 => [
            DVec3::new(u_min, plane, v_min),
            DVec3::new(u_max, plane, v_min),
            DVec3::new(u_max, plane, v_max),
            DVec3::new(u_min, plane, v_max),
        ],
        4 => [
            DVec3::new(u_min, v_min, plane),
            DVec3::new(u_max, v_min, plane),
            DVec3::new(u_max, v_max, plane),
            DVec3::new(u_min, v_max, plane),
        ],
        _ => [
            DVec3::new(u_min, v_min, plane),
            DVec3::new(u_min, v_max, plane),
            DVec3::new(u_max, v_max, plane),
            DVec3::new(u_max, v_min, plane),
        ],
    }
}

fn add_block_face_tile(
    model: &BmfModel,
    quad: [DVec3; 4],
    vertices: &mut Vec<tri00t::Vertex>,
    faces: &mut Vec<[u32; 3]>,
    vertex_map: &mut HashMap<[u64; 3], u32>,
) -> Result<()> {
    let mut indices = [0u32; 4];
    for (dst, corner) in indices.iter_mut().zip(quad) {
        let world = model.local_to_world(corner);
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

fn point_key(point: DVec3) -> [u64; 3] {
    [quantize(point.x), quantize(point.y), quantize(point.z)]
}

fn quantize(value: f64) -> u64 {
    let rounded = (value * 1_000_000.0).round();
    if rounded == 0.0 {
        0.0_f64.to_bits()
    } else {
        rounded.to_bits()
    }
}
