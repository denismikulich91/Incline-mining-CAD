use std::{fmt::Debug, hash::Hash};

use crate::{
    model::block_model::{ColorStop, MAX_COLOR_STOPS, MIN_COLOR_STOPS},
    ui::{fonts::bold, state::UiCommand},
};

/// A compact tool panel pinned inside the 3D viewport.
///
/// Apears in the bottom left. Used for tool configuration.
pub(crate) struct ViewportDockPanel {
    id: egui::Id,
    title: egui::WidgetText,
    viewport_rect: egui::Rect,
    min_width: f32,
    margin: egui::Vec2,
}

impl ViewportDockPanel {
    pub(crate) fn new(
        id_source: impl Hash + Debug,
        title: impl Into<egui::WidgetText>,
        viewport_rect: egui::Rect,
    ) -> Self {
        Self {
            id: egui::Id::new(id_source),
            title: title.into().fallback_text_style(egui::TextStyle::Button),
            viewport_rect,
            min_width: 0.0,
            margin: egui::vec2(10.0, 10.0),
        }
    }

    pub(crate) fn min_width(mut self, min_width: f32) -> Self {
        self.min_width = min_width;
        self
    }

    pub(crate) fn show<R>(
        self,
        ctx: &egui::Context,
        add_contents: impl FnOnce(&mut egui::Ui) -> R,
    ) {
        let pos = egui::pos2(
            self.viewport_rect.left() + self.margin.x,
            self.viewport_rect.bottom() - self.margin.y,
        );
        egui::Area::new(self.id)
            .order(egui::Order::Foreground)
            .pivot(egui::Align2::LEFT_BOTTOM)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                egui::Frame::window(&ctx.global_style())
                    .inner_margin(egui::Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        if self.min_width > 0.0 {
                            ui.set_min_width(self.min_width);
                        }
                        ui.set_max_width(320.0);
                        ui.label(self.title);
                        ui.add_space(4.0);
                        add_contents(ui)
                    })
                    .inner
            });
    }
}

/// Minimum gap (in normalized `t`) enforced between adjacent colour-transfer
/// stops when dragging, so segments never collapse to zero width.
const STOP_EPSILON: f32 = 0.01;
const COLOR_STOP_HANDLE_WIDTH: f32 = 18.0;
const COLOR_PICKER_BUTTON_WIDTH: f32 = 40.0;
const COLOR_PICKER_BUTTON_HEIGHT: f32 = 18.0;
const COLOR_PICKER_EDGE_BUFFER: f32 = 8.0;
const LEGEND_SIDE_GUTTER: f32 = COLOR_PICKER_BUTTON_WIDTH * 0.5 + COLOR_PICKER_EDGE_BUFFER;
const VISIBLE_ALPHA_EPSILON: f32 = 0.004;

/// Interpolates a colour-transfer function's rgba at `t`, mirroring the
/// shader's `ramp_color` (`src/rendering/shaders/block_model.wgsl`) so the
/// legend matches what's actually drawn. Outside the first/last stop's
/// position, the colour clamps to that endpoint.
fn interpolate_stops(stops: &[ColorStop], t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    if stops.is_empty() {
        return egui::Color32::TRANSPARENT;
    }
    let first = stops[0];
    let last = stops[stops.len() - 1];
    if t <= first.t {
        return color32_from_straight(first.color);
    }
    if t >= last.t {
        return color32_from_straight(last.color);
    }
    for window in stops.windows(2) {
        let (a, b) = (window[0], window[1]);
        if t >= a.t && t <= b.t {
            let mut mixed = color_at_visible_stops(stops, t);
            mixed[3] = a.color[3].max(b.color[3]);
            return color32_from_straight(mixed);
        }
    }
    color32_from_straight(last.color)
}

fn color_at_visible_stops(stops: &[ColorStop], t: f32) -> [f32; 4] {
    let before = stops
        .iter()
        .rev()
        .find(|stop| stop.t <= t && stop.color[3] >= VISIBLE_ALPHA_EPSILON);
    let after = stops
        .iter()
        .find(|stop| stop.t >= t && stop.color[3] >= VISIBLE_ALPHA_EPSILON);

    match (before, after) {
        (Some(a), Some(b)) if (b.t - a.t).abs() > 1e-6 => {
            let k = ((t - a.t) / (b.t - a.t)).clamp(0.0, 1.0);
            let mut mixed = [0.0f32; 4];
            for (mixed, (a, b)) in mixed[..3]
                .iter_mut()
                .zip(a.color[..3].iter().zip(b.color[..3].iter()))
            {
                *mixed = a + (b - a) * k;
            }
            mixed[3] = a.color[3].max(b.color[3]);
            mixed
        }
        (Some(stop), _) | (_, Some(stop)) => stop.color,
        (None, None) => [0.0, 0.0, 0.0, 0.0],
    }
}

fn color32_from_straight(c: [f32; 4]) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        (c[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (c[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (c[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (c[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

fn color32_to_straight(c: egui::Color32) -> [f32; 4] {
    unmultiplied_srgba_to_straight(c.to_srgba_unmultiplied())
}

fn straight_to_unmultiplied_srgba(c: [f32; 4]) -> [u8; 4] {
    c.map(|component| (component.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn unmultiplied_srgba_to_straight(c: [u8; 4]) -> [f32; 4] {
    c.map(|component| component as f32 / 255.0)
}

/// Formats a grade value with a decimal precision that scales down as the
/// magnitude grows, so labels stay short without losing meaningful digits.
fn format_grade(value: f64) -> String {
    if value.abs() >= 1000.0 {
        format!("{value:.0}")
    } else if value.abs() >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.3}")
    }
}

/// An interactive colour-scale legend/editor pinned to the bottom-center of
/// the 3D viewport: a dropdown to pick which numeric variable colours the
/// block model, and a multi-stop gradient bar whose handles can be dragged
/// (to move a stop's position) and clicked (to open a colour+alpha picker).
/// The number of tick labels along the bar scales with the bar's width.
pub(crate) struct ColorScaleLegend<'a> {
    id: egui::Id,
    model: &'a crate::model::block_model::OpenBlockModel,
    /// `None` when the active variable has no usable render range (e.g.
    /// every rendered block is the sentinel/default value) — the dropdown
    /// still needs to be shown so the user isn't stuck on that variable.
    range: Option<(f64, f64)>,
    viewport_rect: egui::Rect,
}

impl<'a> ColorScaleLegend<'a> {
    pub(crate) fn new(
        id_source: impl Hash + Debug,
        model: &'a crate::model::block_model::OpenBlockModel,
        range: Option<(f64, f64)>,
        viewport_rect: egui::Rect,
    ) -> Self {
        Self {
            id: egui::Id::new(id_source),
            model,
            range,
            viewport_rect,
        }
    }

    /// The gradient bar's (and therefore the whole legend's) width, which
    /// scales with the viewport's own width up to a size that stays
    /// readable. Applied to the enclosing frame too, so the window around
    /// the bar always matches it instead of sizing itself independently.
    fn bar_width(&self) -> f32 {
        (self.viewport_rect.width() * 0.35).clamp(200.0, 520.0)
    }

    pub(crate) fn show(self, ctx: &egui::Context, commands: &mut Vec<UiCommand>) {
        let bar_width = self.bar_width();
        let content_width = bar_width + LEGEND_SIDE_GUTTER * 2.0;
        let pos = egui::pos2(
            self.viewport_rect.center().x,
            self.viewport_rect.bottom() - 14.0,
        );
        egui::Area::new(self.id)
            .order(egui::Order::Foreground)
            .pivot(egui::Align2::CENTER_BOTTOM)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                egui::Frame::window(&ctx.global_style())
                    .inner_margin(egui::Margin::symmetric(10, 8))
                    .show(ui, |ui| {
                        ui.set_min_width(content_width);
                        ui.set_max_width(content_width);
                        ui.vertical_centered(|ui| {
                            self.draw_variable_dropdown(ui, content_width, commands);
                            ui.add_space(4.0);
                            if let Some((min, max)) = self.range {
                                self.draw_bar(ui, content_width, bar_width, min, max, commands);
                            } else {
                                self.draw_no_data(ui, content_width);
                            }
                        });
                    });
            });
    }

    fn draw_variable_dropdown(
        &self,
        ui: &mut egui::Ui,
        bar_width: f32,
        commands: &mut Vec<UiCommand>,
    ) {
        let variables: Vec<&str> = self
            .model
            .model
            .numeric_variables()
            .into_iter()
            .filter(|variable| !variable.special)
            .map(|variable| variable.name.as_str())
            .collect();
        let current = self.model.active_numeric_variable.as_deref().unwrap_or("");
        let selected_text = if current.is_empty() {
            "Choose a variable"
        } else {
            current
        };
        egui::ComboBox::from_id_salt((self.id, "variable"))
            .selected_text(egui::RichText::new(selected_text).strong())
            .width(bar_width)
            .show_ui(ui, |ui| {
                for name in variables {
                    if ui.selectable_label(name == current, name).clicked() {
                        commands.push(UiCommand::SetBlockModelColorVariable {
                            id: self.model.id,
                            variable: name.to_owned(),
                        });
                    }
                }
            });
    }

    /// Reserves the same footprint as `draw_bar` so the legend doesn't
    /// resize when a variable has no usable range, and explains why there's
    /// no gradient instead of silently falling back to a plain colour.
    fn draw_no_data(&self, ui: &mut egui::Ui, content_width: f32) {
        let handle_height = 18.0;
        let bar_height = 16.0;
        let editor_height = COLOR_PICKER_BUTTON_HEIGHT + 4.0;
        let label_height = 16.0;
        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(
                content_width,
                handle_height + bar_height + editor_height + label_height + 5.0,
            ),
            egui::Sense::hover(),
        );
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No data for this variable",
            egui::FontId::proportional(11.0),
            ui.visuals().weak_text_color(),
        );
    }

    fn draw_bar(
        &self,
        ui: &mut egui::Ui,
        content_width: f32,
        bar_width: f32,
        min: f64,
        max: f64,
        commands: &mut Vec<UiCommand>,
    ) {
        let handle_height = 18.0;
        let bar_height = 16.0;
        let editor_height = COLOR_PICKER_BUTTON_HEIGHT + 4.0;
        let label_height = 16.0;
        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(
                content_width,
                handle_height + bar_height + editor_height + label_height + 5.0,
            ),
            egui::Sense::hover(),
        );
        let handle_row_rect =
            egui::Rect::from_min_size(rect.min, egui::vec2(content_width, handle_height));
        let bar_rect = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + LEGEND_SIDE_GUTTER, handle_row_rect.bottom()),
            egui::vec2(bar_width, bar_height),
        );
        let editor_row_rect = egui::Rect::from_min_size(
            egui::pos2(rect.min.x, bar_rect.bottom() + 2.0),
            egui::vec2(content_width, editor_height),
        );

        let mut stops = self.model.color_transfer.stops.clone();
        let mut changed = false;
        let mut remove_index = None;
        let selected_id = self.id.with("selected_color_stop");
        let mut selected_stop = ui
            .data_mut(|data| data.get_persisted::<usize>(selected_id))
            .unwrap_or(0)
            .min(stops.len().saturating_sub(1));

        let text_color = ui.visuals().text_color();
        {
            let painter = ui.painter();
            const STRIPS: usize = 96;
            let strip_width = bar_rect.width() / STRIPS as f32;
            for i in 0..STRIPS {
                let t = i as f32 / (STRIPS - 1) as f32;
                let strip_rect = egui::Rect::from_min_size(
                    egui::pos2(bar_rect.left() + i as f32 * strip_width, bar_rect.top()),
                    egui::vec2(strip_width + 0.5, bar_rect.height()),
                );
                painter.rect_filled(strip_rect, 0.0, interpolate_stops(&stops, t));
            }
            painter.rect_stroke(
                bar_rect,
                0.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(40)),
                egui::StrokeKind::Outside,
            );
        }

        let bar_response = ui.interact(
            bar_rect,
            self.id.with("color_stop_bar"),
            egui::Sense::click(),
        );
        if bar_response.double_clicked()
            && stops.len() < MAX_COLOR_STOPS
            && let Some(pos) = bar_response.interact_pointer_pos()
        {
            let t = ((pos.x - bar_rect.left()) / bar_rect.width().max(1.0)).clamp(0.0, 1.0);
            stops.push(ColorStop {
                t,
                color: color32_to_straight(interpolate_stops(&stops, t)),
            });
            selected_stop = stops.len() - 1;
            changed = true;
        }

        // Draggable position handles, drawn just above the bar. Position comes
        // from the pointer rather than accumulated drag deltas, so handles stay
        // attached to the cursor even on long drags.
        for i in 0..stops.len() {
            let x = bar_rect.left() + bar_rect.width() * stops[i].t;
            let handle_rect = egui::Rect::from_center_size(
                egui::pos2(x, handle_row_rect.center().y),
                egui::vec2(COLOR_STOP_HANDLE_WIDTH, handle_height),
            );
            let handle_id = self.id.with(("color_stop_handle", i));
            let response = ui.interact(handle_rect, handle_id, egui::Sense::click_and_drag());
            if response.secondary_clicked() && stops.len() > MIN_COLOR_STOPS {
                remove_index = Some(i);
            }
            if response.clicked() {
                selected_stop = i;
            }
            if response.dragged() {
                let lower = if i == 0 {
                    0.0
                } else {
                    stops[i - 1].t + STOP_EPSILON
                };
                let upper = if i + 1 == stops.len() {
                    1.0
                } else {
                    stops[i + 1].t - STOP_EPSILON
                };
                let (lo, hi) = (lower.min(upper), lower.max(upper));
                if let Some(pos) = response.interact_pointer_pos() {
                    let t = ((pos.x - bar_rect.left()) / bar_rect.width().max(1.0)).clamp(lo, hi);
                    if (stops[i].t - t).abs() > f32::EPSILON {
                        stops[i].t = t;
                        changed = true;
                    }
                }
                selected_stop = i;
            }
            let painter = ui.painter();
            let marker_color = interpolate_stops(&stops, stops[i].t);
            let center = egui::pos2(x, handle_row_rect.center().y);
            let active = i == selected_stop || response.dragged() || response.hovered();
            painter.line_segment(
                [
                    egui::pos2(x, handle_row_rect.bottom() - 1.0),
                    egui::pos2(x, bar_rect.top()),
                ],
                egui::Stroke::new(
                    if active { 1.5 } else { 1.0 },
                    ui.visuals().widgets.noninteractive.bg_stroke.color,
                ),
            );
            painter.circle_filled(center, if active { 5.5 } else { 4.5 }, marker_color);
            painter.circle_stroke(
                center,
                if active { 5.5 } else { 4.5 },
                egui::Stroke::new(
                    if active { 1.5 } else { 1.0 },
                    if active {
                        egui::Color32::BLACK
                    } else {
                        egui::Color32::from_gray(40)
                    },
                ),
            );
        }

        if let Some(index) = remove_index {
            stops.remove(index);
            selected_stop = selected_stop.min(stops.len().saturating_sub(1));
            changed = true;
        }

        // Keep the always-visible legend clean: only the selected stop exposes
        // the full colour+alpha picker, instead of placing a button under every
        // marker where they collide as soon as stops move close together.
        if !stops.is_empty() {
            let selected_x = bar_rect.left() + bar_rect.width() * stops[selected_stop].t;
            let swatch_x = selected_x.clamp(
                rect.left() + COLOR_PICKER_BUTTON_WIDTH * 0.5,
                rect.right() - COLOR_PICKER_BUTTON_WIDTH * 0.5,
            );
            let swatch_rect = egui::Rect::from_center_size(
                egui::pos2(swatch_x, editor_row_rect.center().y),
                egui::vec2(COLOR_PICKER_BUTTON_WIDTH, COLOR_PICKER_BUTTON_HEIGHT),
            );
            let mut srgba = straight_to_unmultiplied_srgba(stops[selected_stop].color);
            let response = ui
                .scope_builder(egui::UiBuilder::new().max_rect(swatch_rect), |ui| {
                    ui.spacing_mut().interact_size =
                        egui::vec2(COLOR_PICKER_BUTTON_WIDTH, COLOR_PICKER_BUTTON_HEIGHT);
                    ui.color_edit_button_srgba_unmultiplied(&mut srgba)
                })
                .inner
                .on_hover_text("Click to edit color; right-click to remove");
            let picker_remove_clicked = (response.secondary_clicked()
                || (ui.rect_contains_pointer(swatch_rect)
                    && ui.input(|input| input.pointer.secondary_clicked())))
                && stops.len() > MIN_COLOR_STOPS;
            if picker_remove_clicked {
                stops.remove(selected_stop);
                selected_stop = selected_stop.min(stops.len().saturating_sub(1));
                changed = true;
            }
            if response.changed() {
                stops[selected_stop].color = unmultiplied_srgba_to_straight(srgba);
                changed = true;
            }
        }

        if changed {
            let selected_t = stops.get(selected_stop).map(|stop| stop.t);
            // Keep the array sorted by position; drags are already clamped
            // between neighbours, but colour-only edits leave order intact,
            // so a defensive sort here is cheap insurance against drift.
            stops.sort_by(|a, b| a.t.total_cmp(&b.t));
            stops.dedup_by(|a, b| (a.t - b.t).abs() < 1e-4);
            selected_stop = selected_t
                .and_then(|t| {
                    stops
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| (a.t - t).abs().total_cmp(&(b.t - t).abs()))
                        .map(|(index, _)| index)
                })
                .unwrap_or(0);
            commands.push(UiCommand::SetBlockModelColorStops {
                id: self.model.id,
                stops,
            });
        }
        ui.data_mut(|data| data.insert_persisted(selected_id, selected_stop));

        let fractions: &[f32] = if bar_width < 260.0 {
            &[0.0, 1.0]
        } else if bar_width < 420.0 {
            &[0.0, 0.5, 1.0]
        } else {
            &[0.0, 0.25, 0.5, 0.75, 1.0]
        };
        let painter = ui.painter();
        for &t in fractions {
            let x = bar_rect.left() + bar_rect.width() * t;
            let value = min + (max - min) * t as f64;
            let anchor = if t <= 0.01 {
                egui::Align2::LEFT_TOP
            } else if t >= 0.99 {
                egui::Align2::RIGHT_TOP
            } else {
                egui::Align2::CENTER_TOP
            };
            let label_top = editor_row_rect.bottom() + 1.0;
            painter.text(
                egui::pos2(x, label_top),
                anchor,
                format_grade(value),
                egui::FontId::proportional(11.0),
                text_color,
            );
        }
    }
}

/// A compact notification pinned to the top of the 3D viewport.
pub(crate) struct ViewportLabel {
    id: egui::Id,
    text: String,
    viewport_rect: egui::Rect,
    margin: f32,
    style: ViewportLabelStyle,
}

#[derive(Clone, Copy)]
pub(crate) enum ViewportLabelStyle {
    Neutral,
    Important,
}

impl ViewportLabel {
    pub(crate) fn new(
        id_source: impl Hash + Debug,
        text: impl Into<String>,
        viewport_rect: egui::Rect,
    ) -> Self {
        Self {
            id: egui::Id::new(id_source),
            text: text.into(),
            viewport_rect,
            margin: 12.0,
            style: ViewportLabelStyle::Neutral,
        }
    }

    pub(crate) fn style(mut self, style: ViewportLabelStyle) -> Self {
        self.style = style;
        self
    }

    pub(crate) fn show(self, ctx: &egui::Context) {
        let pos = egui::pos2(
            self.viewport_rect.center().x,
            self.viewport_rect.top() + self.margin,
        );
        egui::Area::new(self.id)
            .order(egui::Order::Foreground)
            .pivot(egui::Align2::CENTER_TOP)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                let (fill, stroke) = match self.style {
                    ViewportLabelStyle::Neutral => (
                        egui::Color32::from_rgb(255, 246, 218),
                        egui::Color32::from_rgb(226, 210, 164),
                    ),
                    ViewportLabelStyle::Important => (
                        egui::Color32::from_rgb(255, 224, 232),
                        egui::Color32::from_rgb(232, 164, 184),
                    ),
                };
                egui::Frame::new()
                    .fill(fill)
                    .stroke(egui::Stroke::new(1.0, stroke))
                    .corner_radius(egui::CornerRadius::same(4))
                    .inner_margin(egui::Margin::symmetric(10, 5))
                    .show(ui, |ui| {
                        let text = match self.style {
                            ViewportLabelStyle::Neutral => egui::RichText::new(self.text)
                                .color(egui::Color32::from_rgb(52, 43, 25)),
                            ViewportLabelStyle::Important => {
                                bold(&self.text).color(egui::Color32::BLACK)
                            }
                        };
                        ui.add(egui::Label::new(text).wrap_mode(egui::TextWrapMode::Extend));
                    });
            });
    }
}
