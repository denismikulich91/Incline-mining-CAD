//! Debug console: a scrollable text buffer visible when the Console tab is active.
//!
//! Messages come from the bounded runtime log shared with exit/crash reports.

/// Draw the console content, filling the remaining UI area.
///
/// Returns the panel's bounding rect.
pub(crate) fn draw_console(ui: &mut egui::Ui, max_height: f32) -> egui::Rect {
    let lines = crate::logging::console_lines();

    let surface_fill = ui.visuals().extreme_bg_color;
    egui::Panel::bottom("Console")
        .resizable(true)
        .max_size(max_height)
        .frame(egui::Frame::new().fill(surface_fill))
        .show(ui, |ui| {
            let font_id = egui::TextStyle::Monospace.resolve(ui.style());
            let row_height = (font_id.size + 8.0).max(20.0);
            if lines.is_empty() {
                ui.allocate_ui_with_layout(
                    egui::vec2(ui.available_width(), row_height),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        ui.set_min_height(row_height);
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new("No console messages yet")
                                    .monospace()
                                    .weak(),
                            )
                            .truncate(),
                        );
                    },
                );
            } else {
                // Only create widgets and text layouts for rows inside the
                // viewport. The previous multiline TextEdit cloned and laid
                // out the complete 2 MiB log every frame.
                egui::ScrollArea::both()
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show_rows(ui, row_height, lines.len(), |ui, range| {
                        for line in &lines[range] {
                            let color = match line.level {
                                crate::logging::Level::Info => {
                                    egui::Color32::from_rgb(110, 180, 255)
                                }
                                crate::logging::Level::Warn => {
                                    egui::Color32::from_rgb(245, 190, 75)
                                }
                                crate::logging::Level::Error => {
                                    egui::Color32::from_rgb(245, 95, 105)
                                }
                            };
                            ui.allocate_ui_with_layout(
                                egui::vec2(ui.available_width(), row_height),
                                egui::Layout::top_down(egui::Align::LEFT),
                                |ui| {
                                    ui.set_min_height(row_height);
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(&line.text)
                                                .font(font_id.clone())
                                                .color(color),
                                        )
                                        .truncate(),
                                    );
                                },
                            );
                        }
                    });
            }
        })
        .response
        .rect
}
