use std::{cell::RefCell, path::PathBuf, sync::Arc};

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::model::formats::bmf::{BdfDefinition, BmfModel};

pub(crate) const MIN_COLOR_STOPS: usize = 2;
pub(crate) const MAX_COLOR_STOPS: usize = 12;

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
    pub(crate) scene_was_empty: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ColorStop {
    /// Position within the active variable's render range, `0..1`.
    pub(crate) t: f32,
    /// Straight (non-premultiplied) RGBA, `0..1` per channel.
    pub(crate) color: [f32; 4],
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
            stops: vec![
                ColorStop {
                    t: 0.00,
                    color: [0.0, 0.86, 0.22, 0.0],
                },
                ColorStop {
                    t: 0.25,
                    color: [0.0, 0.86, 0.22, 1.0],
                },
                ColorStop {
                    t: 0.50,
                    color: [1.0, 0.85, 0.0, 1.0],
                },
                ColorStop {
                    t: 0.75,
                    color: [0.92, 0.0, 0.0, 1.0],
                },
                ColorStop {
                    t: 1.00,
                    color: [0.92, 0.0, 0.0, 0.0],
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
    /// Holds `(variable name, decode result)`; a failed decode is cached too
    /// so a broken variable isn't re-decoded every frame.
    pub(crate) active_values_cache: ActiveValuesCache,
}

pub(crate) type ActiveValuesCache = RefCell<Option<(String, Option<Arc<Vec<f64>>>)>>;

impl OpenBlockModel {
    pub(crate) fn entity_id(&self) -> crate::model::SceneEntityId {
        crate::model::SceneEntityId::BlockModel(self.id)
    }

    /// Decoded values for the active colour variable, decoding at most once
    /// per variable switch. `None` when there is no active variable or it
    /// can't be decoded.
    pub(crate) fn active_numeric_values(&self) -> Option<Arc<Vec<f64>>> {
        let name = self.active_numeric_variable.as_deref()?;
        let mut cache = self.active_values_cache.borrow_mut();
        if let Some((cached_name, values)) = cache.as_ref()
            && cached_name == name
        {
            return values.clone();
        }
        let values = self.model.numeric_values(name).ok().map(Arc::new);
        *cache = Some((name.to_owned(), values.clone()));
        values
    }

    pub(crate) fn world_bounds(&self) -> Option<(DVec3, DVec3)> {
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        let mut any = false;
        for &index in &self.renderable_block_indices {
            let Some(block) = self.blocks.get(index) else {
                continue;
            };
            let bounds = block_world_bounds(&self.model, *block);
            min = min.min(bounds.lower);
            max = max.max(bounds.upper);
            any = true;
        }
        any.then_some((min, max))
    }
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
