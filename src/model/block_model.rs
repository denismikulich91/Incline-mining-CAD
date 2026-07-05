use std::{cell::RefCell, path::PathBuf, sync::Arc};

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::model::formats::bmf::{BdfDefinition, BmfModel};

pub(crate) const MIN_COLOR_STOPS: usize = 2;
pub(crate) const MAX_COLOR_STOPS: usize = 12;
pub(crate) const FIRST_CUSTOM_COLOR_STOP_ID: u64 = 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BlockModelId(pub(crate) u64);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct BlockModelSource {
    pub(crate) bmf_path: PathBuf,
    pub(crate) bdf_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BlockBounds {
    pub(crate) lower: DVec3,
    pub(crate) upper: DVec3,
}

#[derive(Clone, Debug)]
pub(crate) struct LoadedBlockModel {
    pub(crate) name: String,
    pub(crate) source: BlockModelSource,
    pub(crate) model: BmfModel,
    pub(crate) bdf: Option<BdfDefinition>,
    pub(crate) blocks: Vec<BlockBounds>,
    pub(crate) renderable_block_indices: Vec<usize>,
    pub(crate) world_bounds: Option<(DVec3, DVec3)>,
    pub(crate) scene_was_empty: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ColorStop {
    /// Stable runtime identity used by egui widgets. Rendering only reads
    /// `t` and `color`.
    pub(crate) id: u64,
    /// Position within the active variable's render range, `0..1`.
    pub(crate) t: f32,
    /// Straight (non-premultiplied) RGBA, `0..1` per channel.
    pub(crate) color: [f32; 4],
}

impl PartialEq for ColorStop {
    fn eq(&self, other: &Self) -> bool {
        self.t == other.t && self.color == other.color
    }
}

/// A colour ramp driving grade colouring. Stops are always kept sorted
/// ascending by `t`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ColorTransferFunction {
    pub(crate) stops: Vec<ColorStop>,
}

impl Default for ColorTransferFunction {
    fn default() -> Self {
        Self {
            // Stops use hard cutoffs: each stop's colour is active from its `t`
            // up to the next stop. Placing them at 0, 1/3, 2/3 gives green,
            // yellow and red each an equal third of the range (red owns the
            // last third rather than only the single point at t=1.0).
            stops: vec![
                ColorStop {
                    id: 1,
                    t: 0.0,
                    color: [0.0, 0.86, 0.22, 1.0],
                },
                ColorStop {
                    id: 2,
                    t: 1.0 / 3.0,
                    color: [1.0, 0.85, 0.0, 1.0],
                },
                ColorStop {
                    id: 3,
                    t: 2.0 / 3.0,
                    color: [0.92, 0.0, 0.0, 1.0],
                },
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OpenBlockModel {
    pub(crate) id: BlockModelId,
    pub(crate) name: String,
    pub(crate) source: BlockModelSource,
    pub(crate) model: BmfModel,
    pub(crate) bdf: Option<BdfDefinition>,
    pub(crate) blocks: Vec<BlockBounds>,
    pub(crate) renderable_block_indices: Vec<usize>,
    pub(crate) visible: bool,
    pub(crate) color: [f32; 4],
    pub(crate) active_numeric_variable: Option<String>,
    pub(crate) color_transfer: ColorTransferFunction,
    pub(crate) hide_empty_color_values: bool,
    /// Lazily decoded values for [`Self::active_numeric_variable`], shared
    /// by the renderer's grade colouring and the UI's colour-scale legend so
    /// switching colour state doesn't re-decode the whole variable in each.
    /// A failed decode is cached too so a broken variable isn't re-decoded
    /// every frame.
    pub(crate) active_values_cache: ActiveValuesCache,
    /// World-space AABB of all renderable blocks, computed once at load.
    /// Blocks never move afterwards, so the per-frame transparency sort and
    /// scene-bounds queries read this instead of re-walking every block's
    /// eight rotated corners.
    pub(crate) world_bounds: Option<(DVec3, DVec3)>,
}

pub(crate) type ActiveValuesCache = RefCell<Option<ActiveValuesCacheEntry>>;

/// Cached decode of one numeric variable, keyed on [`ActiveValuesCacheEntry::variable`].
/// Holds the decoded values (a failed decode is cached as `None`) plus the
/// render range derived from them, so both are computed at most once per
/// variable switch rather than re-scanned every frame.
#[derive(Clone, Debug)]
pub(crate) struct ActiveValuesCacheEntry {
    variable: String,
    values: Option<Arc<Vec<f64>>>,
    range: Option<(f64, f64)>,
}

impl OpenBlockModel {
    pub(crate) fn entity_id(&self) -> crate::model::SceneEntityId {
        crate::model::SceneEntityId::BlockModel(self.id)
    }

    /// Populates [`Self::active_values_cache`] for the active variable if it
    /// isn't already current: decodes the values and derives the render range
    /// once, so repeat calls in the same frame (and across frames until the
    /// variable changes) are cache reads.
    fn ensure_active_values_cached(&self, name: &str) {
        let mut cache = self.active_values_cache.borrow_mut();
        if cache.as_ref().is_some_and(|entry| entry.variable == name) {
            return;
        }
        let values = self.model.numeric_values(name).ok().map(Arc::new);
        let range = values.as_ref().and_then(|values| {
            let default = self.model.variable(name).and_then(numeric_variable_default);
            render_value_range(values, &self.renderable_block_indices, default)
        });
        *cache = Some(ActiveValuesCacheEntry {
            variable: name.to_owned(),
            values,
            range,
        });
    }

    /// Decoded values for the active colour variable, decoding at most once
    /// per variable switch. `None` when there is no active variable or it
    /// can't be decoded.
    pub(crate) fn active_numeric_values(&self) -> Option<Arc<Vec<f64>>> {
        let name = self.active_numeric_variable.as_deref()?;
        self.ensure_active_values_cached(name);
        self.active_values_cache.borrow().as_ref()?.values.clone()
    }

    /// Render range `(min, max)` for the active colour variable, computed once
    /// per variable switch and shared by the renderer's grade colouring and
    /// the UI's colour-scale legend. `None` when there is no active variable
    /// or no usable range.
    pub(crate) fn active_value_range(&self) -> Option<(f64, f64)> {
        let name = self.active_numeric_variable.as_deref()?;
        self.ensure_active_values_cached(name);
        self.active_values_cache.borrow().as_ref()?.range
    }

    /// World-space AABB of all renderable blocks, cached at load. Reads the
    /// `world_bounds` field; kept as a method so callers are unchanged.
    pub(crate) fn world_bounds(&self) -> Option<(DVec3, DVec3)> {
        self.world_bounds
    }
}

/// World-space AABB of every renderable block, walking each block's eight
/// rotated corners. O(N); call once at load and cache the result.
pub(crate) fn compute_world_bounds(
    model: &crate::model::formats::bmf::BmfModel,
    blocks: &[BlockBounds],
    renderable_block_indices: &[usize],
) -> Option<(DVec3, DVec3)> {
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    let mut any = false;
    for &index in renderable_block_indices {
        let Some(block) = blocks.get(index) else {
            continue;
        };
        let bounds = block_world_bounds(model, *block);
        min = min.min(bounds.lower);
        max = max.max(bounds.upper);
        any = true;
    }
    any.then_some((min, max))
}

fn block_world_bounds(
    model: &crate::model::formats::bmf::BmfModel,
    block: BlockBounds,
) -> BlockBounds {
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    for corner in block_corners(block) {
        let world = model.local_to_world(corner);
        min = min.min(world);
        max = max.max(world);
    }
    BlockBounds {
        lower: min,
        upper: max,
    }
}

/// A numeric variable's fixed "no value" sentinel, parsed from its BMF
/// `global`/`default` field. Shared by the renderer's grade colouring and
/// the UI's colour-scale legend so both treat the same blocks as unset.
pub(crate) fn numeric_variable_default(
    variable: &crate::model::formats::bmf::BmfVariable,
) -> Option<f64> {
    variable
        .global
        .trim()
        .parse::<f64>()
        .or_else(|_| variable.default.trim().parse::<f64>())
        .ok()
}

/// The (min, max) of `values` at `indices`, skipping non-finite values, the
/// variable's default/"unset" value, and Vulcan's common -99/-999 sentinel
/// "no grade" values, so real ore values don't collapse into one colour.
/// `None` when no value in range is usable. Shared by the renderer's grade
/// colouring and the UI's colour-scale legend so both agree on the range.
pub(crate) fn render_value_range(
    values: &[f64],
    indices: &[usize],
    default: Option<f64>,
) -> Option<(f64, f64)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &index in indices {
        let Some(&value) = values.get(index) else {
            continue;
        };
        if !value.is_finite() || default.is_some_and(|default| (value - default).abs() < 1e-8) {
            continue;
        }
        if value <= -90.0 {
            continue;
        }
        min = min.min(value);
        max = max.max(value);
    }
    (min.is_finite() && max.is_finite() && max > min).then_some((min, max))
}

fn block_corners(block: BlockBounds) -> [DVec3; 8] {
    let lo = block.lower;
    let hi = block.upper;
    [
        DVec3::new(lo.x, lo.y, lo.z),
        DVec3::new(hi.x, lo.y, lo.z),
        DVec3::new(hi.x, hi.y, lo.z),
        DVec3::new(lo.x, hi.y, lo.z),
        DVec3::new(lo.x, lo.y, hi.z),
        DVec3::new(hi.x, lo.y, hi.z),
        DVec3::new(hi.x, hi.y, hi.z),
        DVec3::new(lo.x, hi.y, hi.z),
    ]
}
