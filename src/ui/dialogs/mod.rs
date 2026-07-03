//! Floating dialogs and viewport-docked tool panels.
//!
//! Dialogs are grouped by workflow while their draw functions remain
//! re-exported here to keep existing call sites concise.

use crate::model::ObjectId;

pub(crate) mod confirmations;
pub(crate) mod editing;
pub(crate) mod files;
pub(crate) mod triangulation;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SetSelectionZDialog {
    pub(crate) object_ids: Vec<ObjectId>,
    pub(crate) z_input: String,
}
