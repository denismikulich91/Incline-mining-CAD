//! Debug console: a scrollable text buffer visible when the Console tab is active.
//!
//! Messages come from the bounded runtime log shared with exit/crash reports.

/// Draw the console content, filling the remaining UI area.
pub(crate) fn draw_console(ui: &mut egui::Ui) {
    let lines = crate::logging::console_lines();
    let console_text = lines
        .iter()
        .map(|line| line.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let surface_fill = ui.visuals().extreme_bg_color;
    egui::Panel::bottom("Console")
        .resizable(true)
        .frame(egui::Frame::new().fill(surface_fill))
        .show(ui, |ui| {
            ui.set_min_size(ui.available_size());
            egui::ScrollArea::both()
                .auto_shrink([false; 2])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    let font_id = egui::TextStyle::Monospace.resolve(ui.style());
                    let row_height = (font_id.size + 8.0).max(20.0);
                    if console_text.is_empty() {
                        ui.add_sized(
                            [ui.available_width(), row_height],
                            egui::Label::new(
                                egui::RichText::new("No console messages yet")
                                    .monospace()
                                    .weak(),
                            ),
                        );
                    } else {
                        let mut console_view = console_text.as_str();
                        let mut layouter =
                            |ui: &egui::Ui, _buf: &dyn egui::TextBuffer, wrap_width: f32| {
                                let mut job =
                                    console_text_layout(&lines, font_id.clone(), row_height);
                                job.wrap.max_width = wrap_width;
                                ui.fonts_mut(|fonts| fonts.layout_job(job))
                            };
                        egui::TextEdit::multiline(&mut console_view)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(lines.len().max(1))
                            .frame(egui::Frame::NONE)
                            .margin(egui::Margin::same(0))
                            .layouter(&mut layouter)
                            .show(ui);
                    }
                });
        });
}

fn console_text_layout(
    lines: &[crate::logging::ConsoleLine],
    font_id: egui::FontId,
    row_height: f32,
) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = f32::INFINITY;

    for (index, line) in lines.iter().enumerate() {
        let color = match line.level {
            crate::logging::Level::Info => egui::Color32::from_rgb(110, 180, 255),
            crate::logging::Level::Warn => egui::Color32::from_rgb(245, 190, 75),
            crate::logging::Level::Error => egui::Color32::from_rgb(245, 95, 105),
        };
        let text = if index + 1 == lines.len() {
            line.text.clone()
        } else {
            format!("{}\n", line.text)
        };
        job.append(
            &text,
            0.0,
            egui::TextFormat {
                font_id: font_id.clone(),
                color,
                line_height: Some(row_height),
                ..Default::default()
            },
        );
    }

    job
}
