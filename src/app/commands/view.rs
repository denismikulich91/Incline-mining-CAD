use crate::{app::App, userspace_log};

impl<'a> App<'a> {
    pub(crate) fn save_preferences(&self) -> anyhow::Result<()> {
        crate::app::io::save_config(&crate::app::io::Config {
            topology_wireframes_enabled: self.editor.topology_wireframes_enabled,
            dark_mode: self.editor.dark_mode,
            show_console: self.editor.show_console,
            show_world_axis_gizmo: self.editor.show_world_axis_gizmo,
            show_view_cube: self.editor.show_view_cube,
            renderer_background_color: self.editor.renderer_background_color,
            selection_color: self.editor.selection_color,
            snap_poll_rate: self.editor.snap_poll_rate,
            frame_rate_cap: self.editor.frame_rate_cap,
            resize_frame_rate_cap: self.editor.resize_frame_rate_cap,
            frame_counter_enabled: self.editor.frame_counter_enabled,
            topology_folder_search_depth: self.editor.topology_folder_search_depth,
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

    pub(crate) fn set_show_view_cube(&mut self, enabled: bool) -> anyhow::Result<()> {
        self.editor.show_view_cube = enabled;
        if let Some(draft) = self.editor.preferences_draft.as_mut() {
            draft.show_view_cube = enabled;
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

    pub(crate) fn apply_preferences(
        &mut self,
        preferences: crate::ui::state::PreferencesDraft,
    ) -> anyhow::Result<()> {
        crate::app::io::save_config(&crate::app::io::Config {
            topology_wireframes_enabled: preferences.topology_wireframes_enabled,
            dark_mode: preferences.dark_mode,
            show_console: preferences.show_console,
            show_world_axis_gizmo: preferences.show_world_axis_gizmo,
            show_view_cube: preferences.show_view_cube,
            renderer_background_color: preferences.renderer_background_color,
            selection_color: preferences.selection_color,
            snap_poll_rate: preferences.snap_poll_rate.clamp(5, 1000),
            frame_rate_cap: preferences.frame_rate_cap.clamp(20, 1000),
            resize_frame_rate_cap: preferences.resize_frame_rate_cap.clamp(20, 1000),
            frame_counter_enabled: preferences.frame_counter_enabled,
            topology_folder_search_depth: preferences.topology_folder_search_depth.clamp(0, 10),
        })?;

        let selection_changed = self.editor.selection_color != preferences.selection_color;
        self.editor.topology_wireframes_enabled = preferences.topology_wireframes_enabled;
        self.editor.dark_mode = preferences.dark_mode;
        self.editor.show_console = preferences.show_console;
        self.editor.show_world_axis_gizmo = preferences.show_world_axis_gizmo;
        self.editor.show_view_cube = preferences.show_view_cube;
        self.editor.renderer_background_color = preferences.renderer_background_color;
        self.editor.selection_color = preferences.selection_color;
        self.editor.snap_poll_rate = preferences.snap_poll_rate.clamp(5, 1000);
        self.editor.frame_rate_cap = preferences.frame_rate_cap.clamp(20, 1000);
        self.editor.resize_frame_rate_cap = preferences.resize_frame_rate_cap.clamp(20, 1000);
        self.editor.frame_counter_enabled = preferences.frame_counter_enabled;
        self.editor.topology_folder_search_depth =
            preferences.topology_folder_search_depth.clamp(0, 10);
        if !preferences.frame_counter_enabled {
            self.editor.measured_fps = None;
        }
        self.editor.preferences_draft = Some(preferences);
        userspace_log!(
            "Applied preferences (wireframes={}, dark_mode={}, snap_rate={}, fps_cap={}, search_depth={}, frame_counter={})",
            preferences.topology_wireframes_enabled,
            preferences.dark_mode,
            preferences.snap_poll_rate,
            preferences.frame_rate_cap,
            preferences.topology_folder_search_depth,
            preferences.frame_counter_enabled
        );
        if selection_changed {
            self.invalidate_geometry();
        } else {
            self.redraw_requested = true;
        }
        Ok(())
    }

    /// Reset the camera to a plan view that fits all visible content.
    pub(crate) fn reset_view(&mut self) {
        if let Some(graphics) = self.graphics.as_mut() {
            graphics.fit_to_extents(
                &self.scene_document,
                &self.triangulations,
                &self.block_models,
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
                &self.editor.hidden_handles,
            );
            self.redraw_requested = true;
        }
        userspace_log!("Zoom to extents (preserving angle)");
    }
}
