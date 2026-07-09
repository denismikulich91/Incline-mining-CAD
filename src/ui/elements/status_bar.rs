//! Bottom status bar: app identity, selected count, cursor coords, FPS counter.

use crate::ui::EditorState;

/// Draw the bottom status bar panel.
pub(crate) fn draw_status_bar(ui: &mut egui::Ui, editor: &EditorState) -> egui::Rect {
    egui::Panel::bottom("status_bar")
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{} {}", crate::APP_NAME, crate::APP_RELEASE));
                ui.separator();
                ui.label(format!("Selected: {}", editor.selected_handles.len()));
                ui.separator();
                ui.add_enabled(!editor.fly_mode_enabled, egui::Label::new("Orthographic"));
                ui.add_enabled(editor.fly_mode_enabled, egui::Label::new("Perspective"));
                ui.separator();
                match editor.cursor_world {
                    Some(p) => ui.label(format!(" {:.2}, {:.2}, {:.2} ", p.x, p.y, p.z)),
                    None => ui.label(" --, --, -- "),
                };
                if editor.frame_counter_enabled {
                    ui.separator();
                    match editor.measured_fps {
                        Some(fps) => ui.label(format!("FPS: {fps:.1}")),
                        None => ui.label("FPS: --"),
                    };
                }
                if editor.debug_chunk_coloring {
                    ui.separator();
                    match editor.debug_chunk_stats {
                        Some((rendered, total)) => ui.label(format!(
                            "Chunks: {rendered}/{total} ({} culled)",
                            total.saturating_sub(rendered)
                        )),
                        None => ui.label("Chunks: --"),
                    };
                }
                if let Some(msg) = &editor.status_message {
                    ui.separator();
                    match msg.progress {
                        Some(p) => ui.label(format!(
                            "{} ({:.0}%)",
                            msg.text,
                            (p * 100.0).clamp(0.0, 100.0)
                        )),
                        None => ui.label(&msg.text),
                    };
                }
            });
        })
        .response
        .rect
}
