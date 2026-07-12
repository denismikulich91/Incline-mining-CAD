use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::Result;
use spade::Triangulation as SpadeTriangulation;

use crate::{
    app::{App, file_name},
    model::{
        Layer, Object, ObjectId, formats,
        formats::tri00t,
        triangulation::{LoadedTriangulation, OpenTriangulation, TriangulationId},
    },
    ui::state::{TriSurfaceCutSide, TriSurfaceType},
    userspace_log, userspace_warn,
};

// Linear-space value; displays as sRGB 0.8 (204) grey.
const DEFAULT_TRIANGULATION_COLOR: [f32; 4] = [0.6038, 0.6038, 0.6038, 1.0];

mod contours;
mod creation;
mod cuts;
mod geometry;
mod include;
mod point_cloud_tin;
pub(crate) mod session;

use geometry::*;
