//! Preferences tab: renderer background, selection colour, wireframes, dark mode, and advanced
//! options.

use crate::{
    rendering::color::{byte_to_linear_rgba, linear_to_srgb_byte},
    ui::{
        EditorState, UiCommand,
        widgets::menu::{
            MenuFieldBool, MenuFieldColor32, MenuFieldF64, MenuFieldU32, PreferenceCategory,
        },
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
                                if PreferenceCategory::new("Interface".to_string())
                                    .active(
                                        editor.active_preference_category
                                            == crate::ui::state::PreferenceCategory::Interface,
                                    )
                                    .show(ui)
                                    .clicked()
                                {
                                    editor.active_preference_category =
                                        crate::ui::state::PreferenceCategory::Interface;
                                }
                                if PreferenceCategory::new("Camera".to_string())
                                    .active(
                                        editor.active_preference_category
                                            == crate::ui::state::PreferenceCategory::Camera,
                                    )
                                    .show(ui)
                                    .clicked()
                                {
                                    editor.active_preference_category =
                                        crate::ui::state::PreferenceCategory::Camera;
                                }
                                if PreferenceCategory::new("Performance".to_string())
                                    .active(
                                        editor.active_preference_category
                                            == crate::ui::state::PreferenceCategory::Performance,
                                    )
                                    .show(ui)
                                    .clicked()
                                {
                                    editor.active_preference_category =
                                        crate::ui::state::PreferenceCategory::Performance;
                                };
                                if PreferenceCategory::new("Developer".to_string())
                                    .active(
                                        editor.active_preference_category
                                            == crate::ui::state::PreferenceCategory::Developer,
                                    )
                                    .show(ui)
                                    .clicked()
                                {
                                    editor.active_preference_category =
                                        crate::ui::state::PreferenceCategory::Developer;
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
                        match editor.active_preference_category {
                            crate::ui::state::PreferenceCategory::Interface => {
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
                                MenuFieldBool::new("View points", &mut draft.show_points).show(ui);
                                MenuFieldBool::new("Enable dark mode", &mut draft.dark_mode)
                                    .show(ui);
                                MenuFieldBool::new("Show console", &mut draft.show_console)
                                    .show(ui);
                                MenuFieldBool::new(
                                    "Show world axis gizmo",
                                    &mut draft.show_world_axis_gizmo,
                                )
                                .show(ui);
                            }
                            crate::ui::state::PreferenceCategory::Performance => {
                                ui.heading("Performance Preferences");
                                ui.label("It is recommended you leave these options in their default state");
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
                                    "Block model interaction downscale",
                                    &mut draft.block_model_interaction_resolution_divisor,
                                    1..=64,
                                )
                                .suffix("x")
                                    .show(ui);
                            }
                            crate::ui::state::PreferenceCategory::Camera => {
                                ui.heading("Camera Preferences");
                                ui.add_space(12.0);

                                ui.strong("Plan Mode");
                                ui.add_space(6.0);
                                MenuFieldF64::new(
                                    "Orbit sensitivity",
                                    &mut draft.plan_orbit_sensitivity,
                                    0.0001..=0.02,
                                )
                                .speed(0.0001)
                                .max_decimals(4)
                                .show(ui);
                                MenuFieldF64::new(
                                    "Zoom sensitivity",
                                    &mut draft.plan_zoom_sensitivity,
                                    0.0001..=0.05,
                                )
                                .speed(0.0001)
                                .max_decimals(4)
                                .show(ui);
                                MenuFieldBool::new(
                                    "Invert vertical look",
                                    &mut draft.plan_invert_vertical_look,
                                )
                                .show(ui);
                                MenuFieldBool::new(
                                    "Invert horizontal look",
                                    &mut draft.plan_invert_horizontal_look,
                                )
                                .show(ui);
                                MenuFieldBool::new(
                                    "Zoom towards cursor",
                                    &mut draft.plan_zoom_towards_cursor,
                                )
                                .show(ui);

                                ui.add_space(18.0);
                                ui.strong("Fly Mode");
                                ui.add_space(6.0);
                                MenuFieldF64::new(
                                    "Field of view",
                                    &mut draft.fly_field_of_view_degrees,
                                    20.0..=120.0,
                                )
                                .suffix("°")
                                .show(ui);
                                MenuFieldF64::new(
                                    "Mouse look sensitivity",
                                    &mut draft.fly_mouse_look_sensitivity,
                                    0.0001..=0.02,
                                )
                                .speed(0.0001)
                                .max_decimals(4)
                                .show(ui);
                                MenuFieldBool::new(
                                    "Invert vertical look",
                                    &mut draft.fly_invert_vertical_look,
                                )
                                .show(ui);
                                MenuFieldBool::new(
                                    "Invert horizontal look",
                                    &mut draft.fly_invert_horizontal_look,
                                )
                                .show(ui);
                                MenuFieldF64::new(
                                    "Near clip limit",
                                    &mut draft.fly_near_clip_limit,
                                    0.01..=100.0,
                                )
                                .speed(0.01)
                                .suffix("m")
                                .show(ui);
                                MenuFieldF64::new(
                                    "Maximum clip span",
                                    &mut draft.fly_max_clip_span,
                                    100.0..=1_000_000.0,
                                )
                                .speed(100.0)
                                .suffix("m")
                                .show(ui);
                            }
                            crate::ui::state::PreferenceCategory::Developer => {
                                ui.heading("Developer");
                                ui.add_space(12.0);
                                MenuFieldBool::new(
                                    "Enable frame counter",
                                    &mut draft.frame_counter_enabled,
                                )
                                .show(ui);
                                ui.add_space(12.0);
                                MenuFieldBool::new(
                                    "Colour triangles by GPU chunk",
                                    &mut draft.debug_chunk_coloring,
                                )
                                .show(ui);
                                ui.add_space(4.0);
                                ui.label(
                                    "Visualises the Morton spatial chunking used for frustum \
                                     culling.",
                                );
                                ui.add_space(12.0);
                                MenuFieldBool::new(
                                    "Show camera clipping planes",
                                    &mut draft.debug_clip_planes,
                                )
                                .show(ui);
                                ui.add_space(4.0);
                                ui.label(
                                    "Shows the live near and far projection distances in the \
                                     status bar.",
                                );
                            }
                        }
                    });
                },
            );
        });
    });
}
