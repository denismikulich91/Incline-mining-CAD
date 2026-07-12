use std::{fmt::Debug, hash::Hash, path::PathBuf};

/// A non-resizable, non-collapsible floating menu with a draggable title bar.
///
/// Built on top of `egui::Window` but with fixed sizing and collapsibility
/// disabled.  Ideal for non-tool dialogs.
pub(crate) struct DragableMenu<'open> {
    title: egui::WidgetText,
    open: Option<&'open mut bool>,
    min_width: f32,
    default_pos: Option<egui::Pos2>,
    current_pos: Option<egui::Pos2>,
}

impl<'open> DragableMenu<'open> {
    pub(crate) fn new(title: impl Into<egui::WidgetText>) -> Self {
        Self {
            title: title.into().fallback_text_style(egui::TextStyle::Button),
            open: None,
            min_width: 0.0,
            default_pos: None,
            current_pos: None,
        }
    }

    pub(crate) fn open(mut self, open: &'open mut bool) -> Self {
        self.open = Some(open);
        self
    }

    pub(crate) fn min_width(mut self, min_width: f32) -> Self {
        self.min_width = min_width;
        self
    }

    pub(crate) fn default_pos(mut self, default_pos: egui::Pos2) -> Self {
        self.default_pos = Some(default_pos);
        self
    }

    pub(crate) fn show<R>(
        self,
        ctx: &egui::Context,
        add_contents: impl FnOnce(&mut egui::Ui) -> R,
    ) -> Option<egui::InnerResponse<Option<R>>> {
        let frame =
            egui::Frame::window(&ctx.global_style()).inner_margin(egui::Margin::symmetric(4, 1));
        let mut window = egui::Window::new(self.title)
            .auto_sized()
            .collapsible(false)
            .resizable(false)
            .frame(frame);
        window = if let Some(current_pos) = self.current_pos {
            window
                .pivot(egui::Align2::LEFT_TOP)
                .current_pos(current_pos)
        } else if let Some(default_pos) = self.default_pos {
            window
                .pivot(egui::Align2::LEFT_TOP)
                .default_pos(default_pos)
        } else {
            window
                .pivot(egui::Align2::CENTER_CENTER)
                .default_pos(ctx.content_rect().center())
        };
        if let Some(open) = self.open {
            window = window.open(open);
        }
        let min_width = self.min_width;
        window.show(ctx, |ui| {
            if min_width > 0.0 {
                ui.set_min_width(min_width);
            }
            // Default cap: prevent auto-sized dialogs from growing unconstrained.
            // Individual dialogs can call ui.set_max_width() in their content to override.
            ui.set_max_width(400.0);
            add_contents(ui)
        })
    }
}

const MENU_FIELD_WIDTH: f32 = 120.0;

/// A labelled menu row for controls that do not fit one of the standard field types.
pub(crate) struct MenuField {
    label: egui::WidgetText,
}

impl MenuField {
    pub(crate) fn new(label: impl Into<egui::WidgetText>) -> Self {
        Self {
            label: label.into(),
        }
    }

    pub(crate) fn show<R>(
        self,
        ui: &mut egui::Ui,
        add_field: impl FnOnce(&mut egui::Ui, f32) -> R,
    ) -> R {
        menu_field_row(ui, self.label, add_field)
    }
}

/// A labelled file-picker row showing the selected file count/name and a choose button.
pub(crate) struct MenuFieldFilePicker<'paths> {
    label: egui::WidgetText,
    paths: &'paths [PathBuf],
    empty_text: egui::WidgetText,
    button_text: egui::WidgetText,
    width: f32,
}

impl<'paths> MenuFieldFilePicker<'paths> {
    pub(crate) fn new(label: impl Into<egui::WidgetText>, paths: &'paths [PathBuf]) -> Self {
        Self {
            label: label.into(),
            paths,
            empty_text: "No file chosen".into(),
            button_text: "Choose...".into(),
            width: MENU_FIELD_WIDTH,
        }
    }

    pub(crate) fn empty_text(mut self, text: impl Into<egui::WidgetText>) -> Self {
        self.empty_text = text.into();
        self
    }

    pub(crate) fn button_text(mut self, text: impl Into<egui::WidgetText>) -> Self {
        self.button_text = text.into();
        self
    }

    pub(crate) fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    pub(crate) fn show(self, ui: &mut egui::Ui) -> egui::Response {
        let Self {
            label,
            paths,
            empty_text,
            button_text,
            width,
        } = self;
        let text = selected_file_label(paths, empty_text);
        menu_field_row(ui, label, |ui, row_height| {
            let inner = ui.horizontal(|ui| {
                let clicked = ui
                    .add_sized(
                        [row_height * 3.8, row_height],
                        egui::Button::new(button_text),
                    )
                    .clicked();
                ui.add_sized([width, row_height], egui::Label::new(text).truncate());
                clicked
            });
            let mut response = inner.response;
            if inner.inner {
                response.mark_changed();
            }
            response
        })
    }
}

fn selected_file_label(paths: &[PathBuf], empty_text: egui::WidgetText) -> egui::WidgetText {
    match paths {
        [] => empty_text,
        [path] => path
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned)
            .unwrap_or_else(|| path.to_string_lossy().into_owned())
            .into(),
        paths => format!("{} files selected", paths.len()).into(),
    }
}

fn menu_field_row<R>(
    ui: &mut egui::Ui,
    label: egui::WidgetText,
    add_field: impl FnOnce(&mut egui::Ui, f32) -> R,
) -> R {
    let row_height = ui.spacing().interact_size.y;
    let row_width = ui.available_width();

    ui.allocate_ui_with_layout(
        egui::vec2(row_width, row_height),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.label(label);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                add_field(ui, row_height)
            })
            .inner
        },
    )
    .inner
}

macro_rules! menu_field_float {
    ($name:ident, $value_type:ty) => {
        pub(crate) struct $name<'value> {
            label: egui::WidgetText,
            value: &'value mut $value_type,
            range: std::ops::RangeInclusive<$value_type>,
            width: f32,
            speed: f64,
            suffix: String,
            max_decimals: usize,
        }

        #[allow(dead_code)]
        impl<'value> $name<'value> {
            pub(crate) fn new(
                label: impl Into<egui::WidgetText>,
                value: &'value mut $value_type,
                range: std::ops::RangeInclusive<$value_type>,
            ) -> Self {
                Self {
                    label: label.into(),
                    value,
                    range,
                    width: MENU_FIELD_WIDTH,
                    speed: 0.1,
                    suffix: String::new(),
                    max_decimals: 2,
                }
            }

            pub(crate) fn width(mut self, width: f32) -> Self {
                self.width = width;
                self
            }

            pub(crate) fn speed(mut self, speed: f64) -> Self {
                self.speed = speed;
                self
            }

            pub(crate) fn suffix(mut self, suffix: impl Into<String>) -> Self {
                self.suffix = suffix.into();
                self
            }

            pub(crate) fn max_decimals(mut self, max_decimals: usize) -> Self {
                self.max_decimals = max_decimals;
                self
            }

            pub(crate) fn show(self, ui: &mut egui::Ui) -> egui::Response {
                let Self {
                    label,
                    value,
                    range,
                    width,
                    speed,
                    suffix,
                    max_decimals,
                } = self;
                menu_field_row(ui, label, |ui, row_height| {
                    ui.add_sized(
                        [width, row_height],
                        egui::DragValue::new(value)
                            .speed(speed)
                            .range(range)
                            .suffix(suffix)
                            .max_decimals(max_decimals),
                    )
                })
            }

            pub(crate) fn show_inline(self, ui: &mut egui::Ui) -> egui::Response {
                let Self {
                    label,
                    value,
                    range,
                    width,
                    speed,
                    suffix,
                    max_decimals,
                } = self;
                let row_height = ui.spacing().interact_size.y;
                ui.label(label);
                ui.add_sized(
                    [width, row_height],
                    egui::DragValue::new(value)
                        .speed(speed)
                        .range(range)
                        .suffix(suffix)
                        .max_decimals(max_decimals),
                )
            }
        }
    };
}

menu_field_float!(MenuFieldF32, f32);
menu_field_float!(MenuFieldF64, f64);

/// A consistently aligned sliding boolean toggle field.
pub(crate) struct MenuFieldBool<'value> {
    label: egui::WidgetText,
    value: &'value mut bool,
}

impl<'value> MenuFieldBool<'value> {
    pub(crate) fn new(label: impl Into<egui::WidgetText>, value: &'value mut bool) -> Self {
        Self {
            label: label.into(),
            value,
        }
    }

    pub(crate) fn show(self, ui: &mut egui::Ui) -> egui::Response {
        let Self { label, value } = self;
        menu_field_row(ui, label, |ui, row_height| {
            ui.add_sized([row_height * 1.8, row_height], SlidingToggle::new(value))
        })
    }
}

struct SlidingToggle<'value> {
    value: &'value mut bool,
}

impl<'value> SlidingToggle<'value> {
    fn new(value: &'value mut bool) -> Self {
        Self { value }
    }
}

impl egui::Widget for SlidingToggle<'_> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let desired_size = ui.available_size_before_wrap();
        let height = desired_size.y.max(ui.spacing().interact_size.y);
        let width = desired_size.x.max(height * 1.8);
        let (rect, mut response) =
            ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click());

        if response.clicked() {
            *self.value = !*self.value;
            response.mark_changed();
        }

        response.widget_info(|| {
            egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *self.value, "")
        });

        if ui.is_rect_visible(rect) {
            let visuals = ui.style().interact_selectable(&response, *self.value);
            let animation = ui
                .ctx()
                .animate_bool_responsive(response.id.with("sliding_toggle"), *self.value);
            let radius = rect.height() / 2.0;
            let track_rect = rect.shrink2(egui::vec2(1.0, 3.0));
            let off_fill = ui.visuals().widgets.inactive.bg_fill;
            let on_fill = ui.visuals().selection.bg_fill;
            let track_fill = if *self.value { on_fill } else { off_fill };
            let knob_radius = (track_rect.height() / 2.0 - 2.0).max(2.0);
            let knob_x = egui::lerp(
                (track_rect.left() + radius)..=(track_rect.right() - radius),
                animation,
            );
            let knob_center = egui::pos2(knob_x, track_rect.center().y);

            ui.painter().rect(
                track_rect,
                radius,
                track_fill,
                visuals.bg_stroke,
                egui::StrokeKind::Inside,
            );
            ui.painter()
                .circle_filled(knob_center, knob_radius, visuals.fg_stroke.color);
        }

        response
    }
}

pub(crate) struct MenuFieldU32<'value> {
    label: egui::WidgetText,
    value: &'value mut u32,
    range: std::ops::RangeInclusive<u32>,
    width: f32,
    speed: f32,
    suffix: String,
}

#[allow(dead_code)]
impl<'value> MenuFieldU32<'value> {
    pub(crate) fn new(
        label: impl Into<egui::WidgetText>,
        value: &'value mut u32,
        range: std::ops::RangeInclusive<u32>,
    ) -> Self {
        Self {
            label: label.into(),
            value,
            range,
            width: MENU_FIELD_WIDTH,
            speed: 1.,
            suffix: String::new(),
        }
    }

    pub(crate) fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    pub(crate) fn speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    pub(crate) fn suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }

    pub(crate) fn show(self, ui: &mut egui::Ui) -> egui::Response {
        let Self {
            label,
            value,
            range,
            width,
            speed,
            suffix,
        } = self;
        menu_field_row(ui, label, |ui, row_height| {
            ui.add_sized(
                [width, row_height],
                egui::DragValue::new(value)
                    .speed(speed)
                    .range(range)
                    .suffix(suffix),
            )
        })
    }
}

/// A consistently aligned single-line text field for draggable and docked menus.
pub(crate) struct MenuFieldText<'value> {
    label: egui::WidgetText,
    value: &'value mut String,
    width: f32,
    hint: egui::WidgetText,
}

impl<'value> MenuFieldText<'value> {
    pub(crate) fn new(label: impl Into<egui::WidgetText>, value: &'value mut String) -> Self {
        Self {
            label: label.into(),
            value,
            width: MENU_FIELD_WIDTH,
            hint: egui::WidgetText::default(),
        }
    }

    pub(crate) fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    pub(crate) fn hint_text(mut self, hint: impl Into<egui::WidgetText>) -> Self {
        self.hint = hint.into();
        self
    }

    pub(crate) fn show(self, ui: &mut egui::Ui) -> egui::Response {
        let Self {
            label,
            value,
            width,
            hint,
        } = self;
        menu_field_row(ui, label, |ui, row_height| {
            ui.add_sized(
                [width, row_height],
                egui::TextEdit::singleline(value).hint_text(hint),
            )
        })
    }
}

/// A consistently aligned sRGBA colour field for draggable and docked menus.
pub(crate) struct MenuFieldColor32<'value> {
    label: egui::WidgetText,
    value: &'value mut egui::Color32,
}

impl<'value> MenuFieldColor32<'value> {
    pub(crate) fn new(
        label: impl Into<egui::WidgetText>,
        value: &'value mut egui::Color32,
    ) -> Self {
        Self {
            label: label.into(),
            value,
        }
    }

    pub(crate) fn show(self, ui: &mut egui::Ui) -> egui::Response {
        let Self { label, value } = self;
        menu_field_row(ui, label, |ui, _| ui.color_edit_button_srgba(value))
    }
}

/// A consistently aligned premultiplied RGBA colour field.
pub(crate) struct MenuFieldRgba<'value> {
    label: egui::WidgetText,
    value: &'value mut [f32; 4],
}

impl<'value> MenuFieldRgba<'value> {
    pub(crate) fn new(label: impl Into<egui::WidgetText>, value: &'value mut [f32; 4]) -> Self {
        Self {
            label: label.into(),
            value,
        }
    }

    pub(crate) fn show(self, ui: &mut egui::Ui) -> egui::Response {
        let Self { label, value } = self;
        menu_field_row(ui, label, |ui, _| {
            ui.color_edit_button_rgba_premultiplied(value)
        })
    }
}

/// A consistently aligned combo-box field.
pub(crate) struct MenuFieldCombo<'value, T> {
    id: egui::Id,
    label: egui::WidgetText,
    value: &'value mut T,
    selected_text: egui::WidgetText,
    options: Vec<(T, egui::WidgetText)>,
    width: f32,
}

impl<'value, T: PartialEq> MenuFieldCombo<'value, T> {
    pub(crate) fn new(
        id_source: impl Hash + Debug,
        label: impl Into<egui::WidgetText>,
        value: &'value mut T,
        selected_text: impl Into<egui::WidgetText>,
        options: impl IntoIterator<Item = (T, egui::WidgetText)>,
    ) -> Self {
        Self {
            id: egui::Id::new(id_source),
            label: label.into(),
            value,
            selected_text: selected_text.into(),
            options: options.into_iter().collect(),
            width: MENU_FIELD_WIDTH,
        }
    }

    pub(crate) fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    pub(crate) fn show(self, ui: &mut egui::Ui) -> egui::Response {
        let Self {
            id,
            label,
            value,
            selected_text,
            options,
            width,
        } = self;
        menu_field_row(ui, label, |ui, _| {
            let selected_tooltip = selected_text.text().to_owned();
            egui::ComboBox::from_id_salt(id)
                .selected_text(selected_text)
                .width(width)
                .truncate()
                .show_ui(ui, |ui| {
                    for (option, text) in options {
                        ui.selectable_value(value, option, text);
                    }
                })
                .response
                .on_hover_text(selected_tooltip)
        })
    }
}

/// Used in preferences to select a category
pub(crate) struct PreferenceCategory {
    label: String,
    active: bool,
}

impl PreferenceCategory {
    pub(crate) fn new(label: String) -> Self {
        Self {
            label,
            active: false,
        }
    }

    pub(crate) fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    pub(crate) fn show(self, ui: &mut egui::Ui) -> egui::Response {
        let Self { label, active, .. } = self;
        let visuals = ui.visuals();
        let fill = if active {
            crate::ui::SELECTION_COLOR
        } else {
            visuals.code_bg_color
        };
        ui.add(
            egui::Button::new(crate::ui::fonts::bold(&label).color(visuals.strong_text_color()))
                .fill(fill)
                .min_size((170., 30.).into()),
        )
    }
}
