use crate::{app::App, userspace_log};

impl<'a> App<'a> {
    pub(crate) fn save_preferences(&self) -> anyhow::Result<()> {
        crate::app::io::save_config(&crate::app::io::Config {
            topology_wireframes_enabled: self.editor.topology_wireframes_enabled,
            dark_mode: self.editor.dark_mode,
            show_console: self.editor.show_console,
            show_world_axis_gizmo: self.editor.show_world_axis_gizmo,
            renderer_background_color: self.editor.renderer_background_color,
            snap_poll_rate: self.editor.snap_poll_rate,
            frame_rate_cap: self.editor.frame_rate_cap,
            resize_frame_rate_cap: self.editor.resize_frame_rate_cap,
            block_model_interaction_resolution_divisor: self
                .editor
                .block_model_interaction_resolution_divisor,
            frame_counter_enabled: self.editor.frame_counter_enabled,
            debug_chunk_coloring: self.editor.debug_chunk_coloring,
        })?;
        Ok(())
    }

    pub(crate) fn set_show_world_axis_gizmo(&mut self, enabled: bool) -> anyhow::Result<()> {
        self.editor.show_world_axis_gizmo = enabled;
        if let Some(draft) = self.editor.preferences_draft.as_mut() {
            draft.show_world_axis_gizmo = enabled;
        }
        self.save_preferences()?;
        self.redraw_requested = true;
        Ok(())
    }

    pub(crate) fn set_topology_wireframes(&mut self, enabled: bool) -> anyhow::Result<()> {
        self.editor.topology_wireframes_enabled = enabled;
        if let Some(draft) = self.editor.preferences_draft.as_mut() {
            draft.topology_wireframes_enabled = enabled;
        }
        self.save_preferences()?;
        // The topology GPU cache detects the style change during the next
        // render; document geometry does not need rebuilding.
        self.redraw_requested = true;
        userspace_log!("Set topology wireframes = {}", enabled);
        Ok(())
    }

    pub(crate) fn set_dark_mode(&mut self, enabled: bool) -> anyhow::Result<()> {
        self.editor.dark_mode = enabled;
        if let Some(draft) = self.editor.preferences_draft.as_mut() {
            draft.dark_mode = enabled;
        }
        self.save_preferences()?;
        self.redraw_requested = true;
        userspace_log!("Set dark mode = {}", enabled);
        Ok(())
    }

    pub(crate) fn set_show_console(&mut self, enabled: bool) -> anyhow::Result<()> {
        self.editor.show_console = enabled;
        if let Some(draft) = self.editor.preferences_draft.as_mut() {
            draft.show_console = enabled;
        }
        self.save_preferences()?;
        self.redraw_requested = true;
        userspace_log!("Set show console = {}", enabled);
        Ok(())
    }

    pub(crate) fn set_debug_chunk_coloring(&mut self, enabled: bool) -> anyhow::Result<()> {
        self.editor.debug_chunk_coloring = enabled;
        if !enabled {
            self.editor.debug_chunk_stats = None;
        }
        if let Some(draft) = self.editor.preferences_draft.as_mut() {
            draft.debug_chunk_coloring = enabled;
        }
        self.save_preferences()?;
        self.redraw_requested = true;
        userspace_log!("Set debug chunk coloring = {}", enabled);
        Ok(())
    }

    pub(crate) fn apply_preferences(
        &mut self,
        mut preferences: crate::ui::state::PreferencesDraft,
    ) -> anyhow::Result<()> {
        // Clamp once, up front, so the saved config, the applied editor state
        // and the retained draft cannot diverge.
        preferences.snap_poll_rate = preferences.snap_poll_rate.clamp(5, 1000);
        preferences.frame_rate_cap = preferences.frame_rate_cap.clamp(20, 1000);
        preferences.resize_frame_rate_cap = preferences.resize_frame_rate_cap.clamp(20, 1000);
        preferences.block_model_interaction_resolution_divisor = preferences
            .block_model_interaction_resolution_divisor
            .clamp(1, 64);

        crate::app::io::save_config(&crate::app::io::Config {
            topology_wireframes_enabled: preferences.topology_wireframes_enabled,
            dark_mode: preferences.dark_mode,
            show_console: preferences.show_console,
            show_world_axis_gizmo: preferences.show_world_axis_gizmo,
            renderer_background_color: preferences.renderer_background_color,
            snap_poll_rate: preferences.snap_poll_rate,
            frame_rate_cap: preferences.frame_rate_cap,
            resize_frame_rate_cap: preferences.resize_frame_rate_cap,
            block_model_interaction_resolution_divisor: preferences
                .block_model_interaction_resolution_divisor,
            frame_counter_enabled: preferences.frame_counter_enabled,
            debug_chunk_coloring: preferences.debug_chunk_coloring,
        })?;

        self.editor.topology_wireframes_enabled = preferences.topology_wireframes_enabled;
        self.editor.dark_mode = preferences.dark_mode;
        self.editor.show_console = preferences.show_console;
        self.editor.show_world_axis_gizmo = preferences.show_world_axis_gizmo;
        self.editor.renderer_background_color = preferences.renderer_background_color;
        self.editor.snap_poll_rate = preferences.snap_poll_rate;
        self.editor.frame_rate_cap = preferences.frame_rate_cap;
        self.editor.resize_frame_rate_cap = preferences.resize_frame_rate_cap;
        self.editor.block_model_interaction_resolution_divisor =
            preferences.block_model_interaction_resolution_divisor;
        self.editor.frame_counter_enabled = preferences.frame_counter_enabled;
        if !preferences.frame_counter_enabled {
            self.editor.measured_fps = None;
        }
        self.editor.debug_chunk_coloring = preferences.debug_chunk_coloring;
        if !preferences.debug_chunk_coloring {
            self.editor.debug_chunk_stats = None;
        }
        self.editor.preferences_draft = Some(preferences);
        userspace_log!(
            "Applied preferences (wireframes={}, dark_mode={}, snap_rate={}, fps_cap={}, frame_counter={}, debug_chunks={})",
            preferences.topology_wireframes_enabled,
            preferences.dark_mode,
            preferences.snap_poll_rate,
            preferences.frame_rate_cap,
            preferences.frame_counter_enabled,
            preferences.debug_chunk_coloring
        );
        self.redraw_requested = true;
        Ok(())
    }

    /// Reset the camera to a plan view that fits all visible content.
    pub(crate) fn reset_view(&mut self) {
        if let Some(graphics) = self.graphics.as_mut() {
            graphics.fit_to_extents(
                &self.scene_document,
                &self.triangulations,
                &self.block_models,
                &self.point_clouds,
                &self.editor.hidden_handles,
            );
            self.redraw_requested = true;
        }
        userspace_log!("Reset view (fit to extents)");
    }

    /// Fit all visible content while preserving the current orbit angle.
    pub(crate) fn zoom_to_extents(&mut self) {
        if let Some(graphics) = self.graphics.as_mut() {
            graphics.zoom_to_extents(
                &self.scene_document,
                &self.triangulations,
                &self.block_models,
                &self.point_clouds,
                &self.editor.hidden_handles,
            );
            self.redraw_requested = true;
        }
        userspace_log!("Zoom to extents (preserving angle)");
    }
}
