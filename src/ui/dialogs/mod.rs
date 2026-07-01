//! Floating dialogs and viewport-docked tool panels.
//!
//! Dialogs are grouped by workflow while their draw functions remain
//! re-exported here to keep existing call sites concise.

pub(crate) mod confirmations;
pub(crate) mod editing;
pub(crate) mod files;
pub(crate) mod triangulation;
