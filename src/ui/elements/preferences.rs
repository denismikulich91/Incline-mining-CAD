//! Preferences tab: renderer background, selection colour, wireframes, dark mode, and advanced
//! options.

use crate::{
    rendering::color::{byte_to_linear_rgba, linear_to_srgb_byte},
    ui::{
        EditorState, UiCommand,
        widgets::menu::{MenuFieldBool, MenuFieldColor32, MenuFieldU32, PrefrenceCatagory},
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

    egui::CentralPanel::default().show(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(50.);
            ui.allocate_ui_with_layout(
                egui::vec2(
                    ui.available_width() * 0.4 + 200.,
                    ui.available_height() - 50.,
                ),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    egui::Panel::left("pannel")
                        .exact_size(200.)
                        .show_separator_line(false)
                        .resizable(false)
                        .show(ui, |ui| {
                            egui::ScrollArea::new([false, true]).show(ui, |ui| {
                                if PrefrenceCatagory::new("Interface".to_string())
                                    .active(
                                        editor.active_prefrence_catagory
                                            == crate::ui::state::PrefrenceCatagory::Interface,
                                    )
                                    .show(ui)
                                    .clicked()
                                {
                                    editor.active_prefrence_catagory =
                                        crate::ui::state::PrefrenceCatagory::Interface;
                                }
                                if PrefrenceCatagory::new("Preformance".to_string())
                                    .active(
                                        editor.active_prefrence_catagory
                                            == crate::ui::state::PrefrenceCatagory::Preformance,
                                    )
                                    .show(ui)
                                    .clicked()
                                {
                                    editor.active_prefrence_catagory =
                                        crate::ui::state::PrefrenceCatagory::Preformance;
                                };
                            });
                        });

                    egui::Panel::bottom("preferences_actions")
                        .resizable(false)
                        .exact_size(40.0)
                        .show_separator_line(false)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if ui.button("Restore Defaults").clicked() {
                                    *draft = Default::default();
                                }
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui
                                            .add_enabled(
                                                *draft != saved,
                                                egui::Button::new("Save Changes"),
                                            )
                                            .clicked()
                                        {
                                            commands.push(UiCommand::ApplyPreferences(*draft));
                                        }
                                        if ui.button("Cancel").clicked() {
                                            *draft = saved;
                                        }
                                    },
                                );
                            });
                        });

                    egui::ScrollArea::new([false, true]).show(ui, |ui| {
                        match editor.active_prefrence_catagory {
                            crate::ui::state::PrefrenceCatagory::Interface => {
                                ui.heading("Interface Preferences");
                                ui.add_space(12.0);

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

                                MenuFieldBool::new(
                                    "Show wireframes",
                                    &mut draft.topology_wireframes_enabled,
                                )
                                .show(ui);
                                MenuFieldBool::new("Enable dark mode", &mut draft.dark_mode)
                                    .show(ui);
                                MenuFieldBool::new(
                                    "Show world axis gizmo",
                                    &mut draft.show_world_axis_gizmo,
                                )
                                .show(ui);
                                MenuFieldBool::new(
                                    "Enable frame counter",
                                    &mut draft.frame_counter_enabled,
                                )
                                .show(ui);
                            }
                            crate::ui::state::PrefrenceCatagory::Preformance => {
                                ui.heading("Preformance Preferences");
                                ui.add_space(12.0);
                                MenuFieldU32::new(
                                    "Snap to point/line polling rate",
                                    &mut draft.snap_poll_rate,
                                    5..=1000,
                                )
                                .suffix(" Hz")
                                .show(ui);
                                MenuFieldU32::new(
                                    "Frame rate cap",
                                    &mut draft.frame_rate_cap,
                                    20..=1000,
                                )
                                .suffix(" FPS")
                                .show(ui);
                                MenuFieldU32::new(
                                    "Frame rate cap while resizing",
                                    &mut draft.resize_frame_rate_cap,
                                    20..=1000,
                                )
                                .suffix(" FPS")
                                .show(ui);
                                MenuFieldU32::new(
                                    "Block model raycast downscale while interacting",
                                    &mut draft.block_model_interaction_resolution_divisor,
                                    1..=64,
                                )
                                .suffix("x")
                                .show(ui);
                            }
                        }
                    });
                },
            );
        });
    });
}
