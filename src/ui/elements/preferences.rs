//! Preferences tab: renderer background, selection colour, wireframes, dark mode, and advanced
//! options.

use crate::{
    rendering::color::{
        byte_to_linear_rgba, color32_to_rgba, linear_to_srgb_byte, rgba_to_color32,
    },
    ui::{
        EditorState, UiCommand,
        widgets::menu::{MenuFieldBool, MenuFieldColor32, MenuFieldU32},
    },
};

/// Draw the full preferences page (shown when the Preferences tab is active).
pub(crate) fn draw_preferences(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    commands: &mut Vec<UiCommand>,
) {
    if editor.preferences_draft.is_none() {
        editor.reset_preferences_draft();
    }
    let saved = editor.current_preferences();
    let draft = editor
        .preferences_draft
        .as_mut()
        .expect("preferences draft initialized above");

    egui::Panel::bottom("preferences_actions")
        .resizable(false)
        .exact_size(40.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Restore Defaults").clicked() {
                    *draft = Default::default();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(*draft != saved, egui::Button::new("Save Changes"))
                        .clicked()
                    {
                        commands.push(UiCommand::ApplyPreferences(*draft));
                    }
                    if ui.button("Cancel").clicked() {
                        *draft = saved;
                    }
                });
            });
        });

    egui::CentralPanel::default().show(ui, |ui| {
        ui.heading("Preferences");
        ui.add_space(12.0);

        ui.allocate_ui_with_layout(
            egui::vec2(350.0, ui.available_height()),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                let [r, g, b, _] = draft.renderer_background_color;
                let mut background = egui::Color32::from_rgb(
                    linear_to_srgb_byte(r),
                    linear_to_srgb_byte(g),
                    linear_to_srgb_byte(b),
                );
                if MenuFieldColor32::new("Renderer background", &mut background)
                    .show(ui)
                    .changed()
                {
                    draft.renderer_background_color = [
                        byte_to_linear_rgba(background.r()),
                        byte_to_linear_rgba(background.g()),
                        byte_to_linear_rgba(background.b()),
                        1.0,
                    ];
                }

                let mut selection = rgba_to_color32(draft.selection_color);
                if MenuFieldColor32::new("Selection colour", &mut selection)
                    .show(ui)
                    .changed()
                {
                    draft.selection_color = color32_to_rgba(selection);
                }

                MenuFieldBool::new("Show wireframes", &mut draft.topology_wireframes_enabled)
                    .show(ui);
                MenuFieldBool::new("Enable dark mode", &mut draft.dark_mode).show(ui);
                MenuFieldBool::new("Show world axis gizmo", &mut draft.show_world_axis_gizmo)
                    .show(ui);
                MenuFieldBool::new("Show view cube", &mut draft.show_view_cube).show(ui);

                ui.add_space(8.0);
                ui.separator();
                egui::CollapsingHeader::new("Advanced")
                    .default_open(false)
                    .show(ui, |ui| {
                        MenuFieldU32::new(
                            "Snap to point/line polling rate",
                            &mut draft.snap_poll_rate,
                            5..=1000,
                        )
                        .suffix(" Hz")
                        .show(ui);
                        MenuFieldU32::new("Frame rate cap", &mut draft.frame_rate_cap, 20..=1000)
                            .suffix(" FPS")
                            .show(ui);
                        MenuFieldU32::new(
                            "Frame rate cap while resizing",
                            &mut draft.resize_frame_rate_cap,
                            20..=1000,
                        )
                        .suffix(" FPS")
                        .show(ui);
                        MenuFieldBool::new(
                            "Enable frame counter",
                            &mut draft.frame_counter_enabled,
                        )
                        .show(ui);
                        MenuFieldU32::new(
                            "Topology folder search depth",
                            &mut draft.topology_folder_search_depth,
                            0..=10,
                        )
                        .show(ui);
                    });
            },
        );
    });
}
