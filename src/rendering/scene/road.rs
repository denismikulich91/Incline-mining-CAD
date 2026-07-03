//! Road scene assembly: thin adapter over the road-network resolver.
//!
//! All road side-line geometry is produced by `model::road_network::resolve`;
//! this module only builds the ghost (in-progress) road from editor state so
//! the preview and the committed result come from the same pipeline.

pub(crate) use crate::model::road_network::{GhostRoad, RoadKey, resolve};

use crate::ui::state::{ActiveTool, EditorState};

/// The pending stroke + cursor as a ghost road, with no rule checks applied.
pub(crate) fn make_ghost_candidate(editor: &EditorState) -> Option<GhostRoad> {
    if editor.active_tool != ActiveTool::MakeRoad {
        return None;
    }
    let mut centerline = editor.pending_stroke.clone();
    if let Some(cursor) = editor.cursor_world {
        centerline.push(cursor);
    }
    if centerline.len() < 2 {
        return None;
    }
    Some(GhostRoad {
        centerline,
        width: editor.road_width,
        camber_degrees: editor.road_camber_degrees,
        shape: editor.road_shape,
    })
}
