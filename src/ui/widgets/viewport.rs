use std::{fmt::Debug, hash::Hash};

use crate::{
    model::block_model::{
        ColorStop, MAX_COLOR_STOPS, MIN_COLOR_STOPS, OpenBlockModel, numeric_variable_default,
        render_value_range,
    },
    ui::{
        fonts::bold,
        state::{EditorState, UiCommand},
    },
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

/// Clamps `raw_t` to `0..1` and, if that lands within `STOP_EPSILON` of an
/// existing stop, nudges it just outside that stop's epsilon band.
///
/// Without this, a double-click meant to insert a new stop near (or, after
/// edge-clamping, exactly on top of) an existing one produces a stop whose
/// `t` is within the later dedup pass's `1e-4` tolerance of the existing
/// stop — so the new stop is silently collapsed back out and insertion
/// appears to do nothing. This is most visible at the bar's edges: clamping
/// an overshot click to exactly `0.0`/`1.0` collides with a stop already
/// sitting at that exact edge (the common case for default colour ramps).
fn nudge_away_from_existing(stops: &[ColorStop], raw_t: f32) -> f32 {
    let mut t = raw_t.clamp(0.0, 1.0);
    for _ in 0..=stops.len() {
        let Some(collision) = stops.iter().find(|stop| (stop.t - t).abs() < STOP_EPSILON) else {
            return t;
        };
        t = if collision.t >= t {
            (collision.t - STOP_EPSILON).max(0.0)
        } else {
            (collision.t + STOP_EPSILON).min(1.0)
        };
    }
    t
}

/// Inserts a new stop at `t` into an already-sorted `stops` vec, keeping it
/// sorted, and returns the index it landed at.
///
/// Inserting in sorted order (rather than appending and re-sorting later) is
/// what keeps positional-neighbour logic correct on the very frame of
/// insertion: the value popup and drag clamps derive a stop's allowed range
/// from `stops[i-1]`/`stops[i+1]`, so a freshly-appended stop parked at the
/// end of the vec would be treated as the right-most one and clamped to the
/// old last stop's position.
fn insert_stop_sorted(stops: &mut Vec<ColorStop>, id: u64, t: f32) -> usize {
    let index = stops.partition_point(|stop| stop.t < t);
    let color = color32_to_straight(interpolate_stops(stops, t));
    stops.insert(index, ColorStop { id, t, color });
    index
}

/// Samples a colour-transfer function's rgba at `t`, mirroring the shader's
/// hard cutoffs. Values below the first stop are transparent; each stop then
/// remains active until the next.
fn interpolate_stops(stops: &[ColorStop], t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    if stops.is_empty() {
        return egui::Color32::TRANSPARENT;
    }
    let first = stops[0];
    let last = stops[stops.len() - 1];
    if t < first.t {
        return egui::Color32::TRANSPARENT;
    }
    if t >= last.t {
        return color32_from_straight(last.color);
    }
    for i in 1..stops.len() {
        if t < stops[i].t {
            return color32_from_straight(stops[i - 1].color);
        }
    }
    color32_from_straight(last.color)
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

fn trim_decimal_zeros(mut value: String) -> String {
    if let Some(dot) = value.find('.') {
        while value.ends_with('0') {
            value.pop();
        }
        if value.len() == dot + 1 {
            value.pop();
        }
    }
    if value == "-0" { "0".to_owned() } else { value }
}

/// Formats a grade value with a decimal precision that scales down as the
/// magnitude grows, so labels stay short without losing meaningful digits.
fn format_grade(value: f64) -> String {
    let decimals = if value.abs() >= 1000.0 {
        0
    } else if value.abs() >= 10.0 {
        1
    } else {
        3
    };
    trim_decimal_zeros(format!("{value:.decimals$}"))
}

fn inferred_decimal_places(value: f64) -> usize {
    for decimals in 0..=4 {
        let scale = 10_f64.powi(decimals as i32);
        let rounded = (value * scale).round() / scale;
        let tolerance = 1e-8 * value.abs().max(1.0);
        if (rounded - value).abs() <= tolerance {
            return decimals;
        }
    }
    4
}

fn format_grade_range(min: f64, max: f64) -> String {
    let decimals = inferred_decimal_places(min).max(inferred_decimal_places(max));
    format!("({min:.decimals$} - {max:.decimals$})")
}

/// An interactive colour-scale legend/editor pinned to the bottom-center of
/// the 3D viewport.
pub(crate) struct ColorScaleLegend<'a> {
    id: egui::Id,
    models: &'a [OpenBlockModel],
    viewport_rect: egui::Rect,
}

impl<'a> ColorScaleLegend<'a> {
    pub(crate) fn new(
        id_source: impl Hash + Debug,
        models: &'a [OpenBlockModel],
        viewport_rect: egui::Rect,
    ) -> Self {
        Self {
            id: egui::Id::new(id_source),
            models,
            viewport_rect,
        }
    }

    fn bar_width(&self) -> f32 {
        (self.viewport_rect.width() * 0.35).clamp(220.0, 560.0)
    }

    pub(crate) fn show(
        self,
        ctx: &egui::Context,
        editor: &mut EditorState,
        commands: &mut Vec<UiCommand>,
    ) {
        // Any visible block model that *can* be colour-mapped belongs in the
        // legend's model selector, including one that has numeric variables but
        // none currently chosen — it still shows the "Choose a variable" picker
        // and a no-data bar, so requiring an active variable here would
        // needlessly drop it from the selector.
        let visible_models = self
            .models
            .iter()
            .filter(|model| model.visible && model_has_selectable_variable(model))
            .collect::<Vec<_>>();
        if visible_models.is_empty() {
            return;
        }

        editor.viewport_block_model_id = editor
            .viewport_block_model_id
            .filter(|id| visible_models.iter().any(|model| model.id == *id))
            .or_else(|| visible_models.first().map(|model| model.id));
        let Some(model) = editor
            .viewport_block_model_id
            .and_then(|id| visible_models.iter().copied().find(|model| model.id == id))
        else {
            return;
        };

        let bar_width = self.bar_width();
        let content_width = bar_width + LEGEND_SIDE_GUTTER * 2.0;
        let range = active_variable_range(editor, model);
        let pos = egui::pos2(
            self.viewport_rect.center().x,
            self.viewport_rect.bottom() - 14.0,
        );
        let area_response = egui::Area::new(self.id)
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
                            self.draw_model_dropdown(ui, content_width, &visible_models, editor);
                            if visible_models.len() > 1 {
                                ui.add_space(4.0);
                            }
                            self.draw_variable_dropdown(ui, content_width, model, editor, commands);
                            ui.add_space(4.0);
                            self.draw_empty_values_toggle(ui, content_width, model, commands);
                            ui.add_space(4.0);
                            if let Some((min, max)) = range {
                                self.draw_bar(
                                    ui,
                                    content_width,
                                    bar_width,
                                    min,
                                    max,
                                    model,
                                    editor,
                                    commands,
                                );
                            } else {
                                self.draw_no_data(ui, content_width);
                            }
                        });
                    });
            });
        // This HUD-style legend is meant to always float above other viewport
        // panels. egui only brings an Area to front when a click hits it as
        // the topmost layer, so once some other floating panel is brought
        // forward and happens to overlap this one, clicks here would never
        // land (the other panel would keep winning hit-tests) without this.
        //
        // Every popup the legend spawns (model/variable dropdowns, stop value
        // input, colour picker) must be registered as a *sublayer* of this
        // area at its call site. `move_to_top` only flags layers into a set
        // that `Areas::end_pass` uses as a stable-sort key, so a popup that
        // once sank below the legend (any frame it was closed while the
        // legend was promoted) can never climb back past it by flagging —
        // clicking the legend re-flags it in the same frame and the tie
        // preserves the old order. Sublayers are re-inserted directly above
        // their parent after the sort, which is the only deterministic way
        // to keep the popups on top of this always-promoted panel.
        ctx.move_to_top(area_response.response.layer_id);
    }

    fn draw_model_dropdown(
        &self,
        ui: &mut egui::Ui,
        content_width: f32,
        visible_models: &[&OpenBlockModel],
        editor: &mut EditorState,
    ) {
        if visible_models.len() <= 1 {
            return;
        }
        let current_id = editor.viewport_block_model_id;
        let selected_text = current_id
            .and_then(|id| visible_models.iter().find(|model| model.id == id))
            .map(|model| model.name.as_str())
            .unwrap_or("Choose a block model");

        // Built on the same Button + `Popup::menu` pattern as the variable
        // dropdown rather than `egui::ComboBox`: the combo's own popup did not
        // stay above / clickable through this always-promoted legend, whereas
        // capturing the real popup response and pinning it as a sublayer does.
        let popup_id = self.id.with("block_model_popup");
        let open = egui::Popup::is_id_open(ui.ctx(), popup_id);
        let button_response = ui
            .add_sized(
                egui::vec2(content_width, 22.0),
                egui::Button::selectable(open, egui::RichText::new(selected_text).strong()),
            )
            .on_hover_text("Choose the block model to colour");

        let popup_response = egui::Popup::menu(&button_response)
            .id(popup_id)
            .width(content_width)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                for model in visible_models {
                    let selected = Some(model.id) == current_id;
                    if ui
                        .selectable_label(selected, egui::RichText::new(&model.name).strong())
                        .clicked()
                    {
                        editor.viewport_block_model_id = Some(model.id);
                        egui::Popup::close_id(ui.ctx(), popup_id);
                    }
                }
            });
        // Pin the model-list popup above the self-promoting legend (see the
        // sublayer comment in `show`), same as the variable dropdown.
        if let Some(popup_response) = popup_response {
            ui.ctx()
                .set_sublayer(ui.layer_id(), popup_response.response.layer_id);
        }
    }

    fn draw_empty_values_toggle(
        &self,
        ui: &mut egui::Ui,
        content_width: f32,
        model: &OpenBlockModel,
        commands: &mut Vec<UiCommand>,
    ) {
        let mut hide = model.hide_empty_color_values;
        ui.allocate_ui_with_layout(
            egui::vec2(content_width, 20.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                if ui.checkbox(&mut hide, "Hide empty").changed() {
                    commands.push(UiCommand::SetBlockModelHideEmptyValues { id: model.id, hide });
                }
            },
        );
    }

    fn draw_variable_dropdown(
        &self,
        ui: &mut egui::Ui,
        content_width: f32,
        model: &OpenBlockModel,
        editor: &mut EditorState,
        commands: &mut Vec<UiCommand>,
    ) {
        let current = model.active_numeric_variable.as_deref().unwrap_or("");
        let selected_text = model
            .active_numeric_variable
            .as_deref()
            .map(|name| {
                if let Some((min, max)) = cached_variable_range(editor, model, name) {
                    format!("{name} {}", format_grade_range(min, max))
                } else {
                    format!("{name} (no range)")
                }
            })
            .unwrap_or_else(|| "Choose a variable".to_owned());
        let filter_id = self.id.with(("variable_filter", model.id));
        let mut filter = ui
            .data_mut(|data| data.get_persisted::<String>(filter_id))
            .unwrap_or_default();

        let popup_id = self.id.with(("variable_popup", model.id));
        let open = egui::Popup::is_id_open(ui.ctx(), popup_id);
        let button_response = ui
            .add_sized(
                egui::vec2(content_width, 22.0),
                egui::Button::selectable(open, egui::RichText::new(selected_text).strong()),
            )
            .on_hover_text("Choose the active block model variable");

        let popup_response = egui::Popup::menu(&button_response)
            .id(popup_id)
            .width(content_width)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                let response = ui.add(
                    egui::TextEdit::singleline(&mut filter)
                        .hint_text("Filter variables")
                        .desired_width(content_width - 12.0),
                );
                if response.changed() {
                    ui.data_mut(|data| data.insert_persisted(filter_id, filter.clone()));
                }
                ui.add_space(4.0);

                let needle = filter.trim().to_ascii_lowercase();
                let mut any = false;
                egui::ScrollArea::vertical()
                    .max_height(220.0)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        for variable in model
                            .model
                            .numeric_variables()
                            .into_iter()
                            .filter(|variable| !variable.special)
                        {
                            let name = variable.name.as_str();
                            if !needle.is_empty() && !name.to_ascii_lowercase().contains(&needle) {
                                continue;
                            }
                            any = true;
                            let range_text = cached_variable_range(editor, model, name)
                                .map(|(min, max)| format_grade_range(min, max))
                                .unwrap_or_else(|| "(no usable range)".to_owned());
                            let selected = name == current;
                            let row = ui
                                .horizontal(|ui| {
                                    let response = ui.selectable_label(
                                        selected,
                                        egui::RichText::new(name).strong(),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                egui::RichText::new(range_text)
                                                    .color(ui.visuals().weak_text_color()),
                                            );
                                        },
                                    );
                                    response
                                })
                                .inner;
                            if row.clicked() {
                                commands.push(UiCommand::SetBlockModelColorVariable {
                                    id: model.id,
                                    variable: name.to_owned(),
                                });
                                egui::Popup::close_id(ui.ctx(), popup_id);
                            }
                        }
                    });
                if !any {
                    ui.label(
                        egui::RichText::new("No matches").color(ui.visuals().weak_text_color()),
                    );
                }
            });
        // Pin the variable-list popup above the self-promoting legend (see
        // the sublayer comment in `show`). Without this, a popup small
        // enough to sit *below* the anchor button (e.g. when a filter such
        // as "dog" matches nothing) lands entirely over the legend's body
        // and is hidden and un-clickable behind it — indistinguishable from
        // the list having closed itself.
        if let Some(popup_response) = popup_response {
            ui.ctx()
                .set_sublayer(ui.layer_id(), popup_response.response.layer_id);
        }
    }

    fn draw_no_data(&self, ui: &mut egui::Ui, content_width: f32) {
        let handle_height = 18.0;
        let bar_height = 16.0;
        // Must match `draw_bar`'s `editor_height` so switching to/from a
        // variable with no usable range doesn't resize the legend.
        let editor_height = COLOR_PICKER_BUTTON_HEIGHT + 6.0;
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

    #[allow(clippy::too_many_arguments)]
    fn draw_bar(
        &self,
        ui: &mut egui::Ui,
        content_width: f32,
        bar_width: f32,
        min: f64,
        max: f64,
        model: &OpenBlockModel,
        editor: &mut EditorState,
        commands: &mut Vec<UiCommand>,
    ) {
        let handle_height = 18.0;
        let bar_height = 16.0;
        let editor_height = COLOR_PICKER_BUTTON_HEIGHT + 6.0;
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

        let mut stops = model.color_transfer.stops.clone();
        let mut changed = false;
        let mut remove_index = None;
        let selected_id = self.id.with(("selected_color_stop", model.id));
        let mut selected_stop = ui
            .data_mut(|data| data.get_persisted::<usize>(selected_id))
            .unwrap_or(0)
            .min(stops.len().saturating_sub(1));
        let value_popup_id = self.id.with(("stop_value_popup_open", model.id));
        let mut value_popup_stop = ui
            .data_mut(|data| data.get_persisted::<Option<usize>>(value_popup_id))
            .flatten()
            .filter(|index| *index < stops.len());
        // Set whenever a stop interaction this frame opens/keeps the value
        // popup, so the click-outside close below doesn't immediately undo it
        // (clicking a handle is "outside" the popup's own rect).
        let mut popup_kept_open = false;

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

        let bar_response = ui
            .interact(
                bar_rect,
                self.id.with("color_stop_bar"),
                egui::Sense::click(),
            )
            .on_hover_text("Double-click to add a stop here");
        // A double-click on either the bar or a handle requests an insert.
        // We record the target `t` and apply it *after* the handle loop so the
        // insertion never shifts indices mid-iteration, and so it lands in
        // sorted position (see `insert_stop_sorted`).
        let mut pending_insert: Option<f32> = None;
        if bar_response.double_clicked()
            && stops.len() < MAX_COLOR_STOPS
            && let Some(pos) = bar_response.interact_pointer_pos()
        {
            let raw_t = (pos.x - bar_rect.left()) / bar_rect.width().max(1.0);
            pending_insert = Some(nudge_away_from_existing(&stops, raw_t));
        }

        for i in 0..stops.len() {
            let x = bar_rect.left() + bar_rect.width() * stops[i].t;
            let handle_rect = egui::Rect::from_center_size(
                egui::pos2(x, handle_row_rect.center().y),
                egui::vec2(COLOR_STOP_HANDLE_WIDTH, handle_height),
            );
            let handle_id = self.id.with(("color_stop_handle", stops[i].id));
            let response = ui
                .interact(handle_rect, handle_id, egui::Sense::click_and_drag())
                .on_hover_text(if stops.len() > MIN_COLOR_STOPS {
                    "Drag to move · Right-click to remove"
                } else {
                    "Drag to move"
                });
            // Handles sit directly above the gradient strip, so a
            // double-click aimed at the bar near an existing stop (most
            // often the last one, at the visually prominent right edge)
            // lands on the handle instead. Without this, that click is
            // swallowed as an ordinary single click that just reselects the
            // handle, and no new stop is ever added.
            if response.double_clicked()
                && stops.len() < MAX_COLOR_STOPS
                && let Some(pos) = response.interact_pointer_pos()
            {
                let raw_t = (pos.x - bar_rect.left()) / bar_rect.width().max(1.0);
                pending_insert = Some(nudge_away_from_existing(&stops, raw_t));
            } else {
                if response.secondary_clicked() && stops.len() > MIN_COLOR_STOPS {
                    remove_index = Some(i);
                }
                if response.clicked() {
                    selected_stop = i;
                    value_popup_stop = Some(i);
                    popup_kept_open = true;
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
                        let t =
                            ((pos.x - bar_rect.left()) / bar_rect.width().max(1.0)).clamp(lo, hi);
                        if (stops[i].t - t).abs() > f32::EPSILON {
                            stops[i].t = t;
                            changed = true;
                        }
                    }
                    selected_stop = i;
                    if value_popup_stop.is_some() {
                        value_popup_stop = Some(i);
                        popup_kept_open = true;
                    }
                }
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

        if let Some(t) = pending_insert
            && stops.len() < MAX_COLOR_STOPS
        {
            let new_index = insert_stop_sorted(&mut stops, editor.allocate_color_stop_id(), t);
            selected_stop = new_index;
            value_popup_stop = Some(new_index);
            popup_kept_open = true;
            changed = true;
        }

        if let Some(index) = remove_index {
            stops.remove(index);
            value_popup_stop = match value_popup_stop {
                Some(open_index) if open_index == index => None,
                Some(open_index) if open_index > index => Some(open_index - 1),
                other => other,
            };
            selected_stop = selected_stop.min(stops.len().saturating_sub(1));
            changed = true;
        }

        if !stops.is_empty() {
            if value_popup_stop == Some(selected_stop) {
                let popup_response = self.draw_stop_value_popup(
                    ui,
                    &mut stops,
                    selected_stop,
                    bar_rect,
                    handle_row_rect,
                    min,
                    max,
                    &mut changed,
                );
                // Close the value input when the user clicks anywhere outside
                // it — unless this frame's click was on a stop handle, which
                // (re)opens it for that stop. Clicking inside the popup to edit
                // the value is not "elsewhere", so it stays open.
                if !popup_kept_open
                    && popup_response.is_some_and(|response| response.clicked_elsewhere())
                {
                    value_popup_stop = None;
                }
            }

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
                .scope_builder(
                    egui::UiBuilder::new()
                        .id_salt("color_stop_color_picker")
                        .max_rect(swatch_rect),
                    |ui| {
                        ui.spacing_mut().interact_size =
                            egui::vec2(COLOR_PICKER_BUTTON_WIDTH, COLOR_PICKER_BUTTON_HEIGHT);
                        // Pin the colour-picker popup above the self-promoting
                        // legend (see the sublayer comment in `show`). egui 0.35
                        // opens it under `ui.auto_id_with("popup")`, computed
                        // before the button is allocated; `auto_id_with` does not
                        // advance the id source, so recomputing it here yields
                        // the same id.
                        let picker_popup_layer =
                            egui::LayerId::new(egui::Order::Foreground, ui.auto_id_with("popup"));
                        ui.ctx().set_sublayer(ui.layer_id(), picker_popup_layer);
                        ui.color_edit_button_srgba_unmultiplied(&mut srgba)
                    },
                )
                .inner
                .on_hover_text("Click to edit color; right-click to remove");
            let picker_remove_clicked = (response.secondary_clicked()
                || (ui.rect_contains_pointer(swatch_rect)
                    && ui.input(|input| input.pointer.secondary_clicked())))
                && stops.len() > MIN_COLOR_STOPS;
            if picker_remove_clicked {
                value_popup_stop = None;
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
            let value_popup_t =
                value_popup_stop.and_then(|index| stops.get(index).map(|stop| stop.t));
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
            value_popup_stop = value_popup_t.and_then(|t| {
                stops
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| (a.t - t).abs().total_cmp(&(b.t - t).abs()))
                    .map(|(index, _)| index)
            });
            commands.push(UiCommand::SetBlockModelColorStops {
                id: model.id,
                stops,
            });
        }
        ui.data_mut(|data| data.insert_persisted(selected_id, selected_stop));
        ui.data_mut(|data| data.insert_persisted(value_popup_id, value_popup_stop));

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
            let value = normalized_to_value(t, min, max);
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

    #[allow(clippy::too_many_arguments)]
    fn draw_stop_value_popup(
        &self,
        ui: &mut egui::Ui,
        stops: &mut [ColorStop],
        selected_stop: usize,
        bar_rect: egui::Rect,
        handle_row_rect: egui::Rect,
        min: f64,
        max: f64,
        changed: &mut bool,
    ) -> Option<egui::Response> {
        let stop = stops.get(selected_stop).copied()?;
        let selected_x = bar_rect.left() + bar_rect.width() * stop.t;
        let popup_pos = egui::pos2(selected_x, handle_row_rect.top() - 6.0);
        let area_response = egui::Area::new(self.id.with("stop_value_popup"))
            .order(egui::Order::Foreground)
            .pivot(egui::Align2::CENTER_BOTTOM)
            .fixed_pos(popup_pos)
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    let lower_t = if selected_stop == 0 {
                        0.0
                    } else {
                        stops[selected_stop - 1].t + STOP_EPSILON
                    };
                    let upper_t = if selected_stop + 1 == stops.len() {
                        1.0
                    } else {
                        stops[selected_stop + 1].t - STOP_EPSILON
                    };
                    let mut value = normalized_to_value(stop.t, min, max);
                    let min_value = normalized_to_value(lower_t.min(upper_t), min, max);
                    let max_value = normalized_to_value(lower_t.max(upper_t), min, max);
                    // No forced width: the popup frame hugs the drag value
                    // instead of leaving empty space the number sat
                    // left-aligned against.
                    let response = ui.add(
                        egui::DragValue::new(&mut value)
                            .range(min_value..=max_value)
                            .speed(((max - min).abs() / 250.0).max(0.0001))
                            .max_decimals(6),
                    );
                    if response.changed() {
                        stops[selected_stop].t = value_to_normalized(value, min, max)
                            .clamp(lower_t.min(upper_t), lower_t.max(upper_t));
                        *changed = true;
                    }
                });
            });
        // Pin the value popup above the self-promoting legend (see the
        // sublayer comment in `show`); it overlaps the legend's own settings
        // controls, and promoting it via `move_to_top` cannot outrank a
        // layer that is itself re-promoted every frame.
        ui.ctx()
            .set_sublayer(ui.layer_id(), area_response.response.layer_id);
        Some(area_response.response)
    }
}

fn active_variable_range(editor: &mut EditorState, model: &OpenBlockModel) -> Option<(f64, f64)> {
    let name = model.active_numeric_variable.as_deref()?;
    cached_variable_range(editor, model, name)
}

/// Whether the legend can show this model: it already has an active variable,
/// or it has at least one non-special numeric variable the user could pick.
fn model_has_selectable_variable(model: &OpenBlockModel) -> bool {
    model.active_numeric_variable.is_some()
        || model
            .model
            .numeric_variables()
            .into_iter()
            .any(|variable| !variable.special)
}

fn cached_variable_range(
    editor: &mut EditorState,
    model: &OpenBlockModel,
    name: &str,
) -> Option<(f64, f64)> {
    let key = (model.id, name.to_owned());
    if let Some(range) = editor.block_model_variable_ranges.get(&key) {
        return *range;
    }
    let range = if model.active_numeric_variable.as_deref() == Some(name) {
        model.active_value_range()
    } else {
        model.model.variable(name).and_then(|variable| {
            let default = numeric_variable_default(variable);
            model.model.numeric_values(name).ok().and_then(|values| {
                render_value_range(&values, &model.renderable_block_indices, default)
            })
        })
    };
    editor.block_model_variable_ranges.insert(key, range);
    range
}

fn normalized_to_value(t: f32, min: f64, max: f64) -> f64 {
    min + (max - min) * t.clamp(0.0, 1.0) as f64
}

fn value_to_normalized(value: f64, min: f64, max: f64) -> f32 {
    if (max - min).abs() <= f64::EPSILON {
        0.0
    } else {
        ((value - min) / (max - min)).clamp(0.0, 1.0) as f32
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

/// Fixed, titleless shaded plan preview shown while slice mode is active.
pub(crate) struct ViewportMiniMap {
    id: egui::Id,
    viewport_rect: egui::Rect,
}

impl ViewportMiniMap {
    pub(crate) fn new(id_source: impl Hash + Debug, viewport_rect: egui::Rect) -> Self {
        Self {
            id: egui::Id::new(id_source),
            viewport_rect,
        }
    }

    pub(crate) fn show(
        self,
        ctx: &egui::Context,
        editor: &mut EditorState,
        commands: &mut Vec<crate::ui::state::UiCommand>,
    ) {
        if editor.slice_preview_detached {
            return;
        }

        // Keep the embedded preview proportional to the main viewport. It is
        // deliberately not user-resizable: resizing an egui window captures
        // pointer interaction and makes the following middle-drag feel as if
        // the 3D canvas needs to be focused again.
        let preview_size = egui::vec2(
            (self.viewport_rect.width() * 0.24).max(160.0),
            (self.viewport_rect.height() * 0.24).max(160.0),
        );
        let preview_pos = egui::pos2(
            (self.viewport_rect.right() - preview_size.x - 10.0).max(self.viewport_rect.left()),
            self.viewport_rect.top() + 104.0,
        );
        let frame = egui::Frame::window(&ctx.global_style()).inner_margin(egui::Margin::ZERO);
        egui::Window::new("")
            .id(self.id)
            .order(egui::Order::Foreground)
            .fixed_pos(preview_pos)
            .fixed_size(preview_size)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .title_bar(false)
            .frame(frame)
            .show(ctx, |ui| {
                let size = ui.available_size().max(egui::vec2(120.0, 120.0));
                let response = if let Some(texture_id) = editor.slice_preview_texture {
                    ui.add(
                        egui::Image::new(egui::load::SizedTexture::new(texture_id, size))
                            .fit_to_exact_size(size)
                            .sense(egui::Sense::click()),
                    )
                } else {
                    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
                    ui.painter()
                        .rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
                    response
                };
                let pixels_per_point = ctx.pixels_per_point();
                editor.slice_preview_size_px = [
                    (response.rect.width() * pixels_per_point).round().max(1.0) as u32,
                    (response.rect.height() * pixels_per_point).round().max(1.0) as u32,
                ];
                if response
                    .on_hover_text("Click to detach into a full-resolution window")
                    .clicked()
                {
                    commands.push(crate::ui::state::UiCommand::SetSlicePreviewDetached(true));
                }
            });
    }
}
