/// An icon-and-label row used for entries in the explorer tree.
pub(crate) struct ExplorerEntry {
    icon: egui::ImageSource<'static>,
    title: egui::WidgetText,
    selected: bool,
    icon_size: egui::Vec2,
}

impl ExplorerEntry {
    pub(crate) fn new(
        icon: egui::ImageSource<'static>,
        title: impl Into<egui::WidgetText>,
    ) -> Self {
        Self {
            icon,
            title: title.into(),
            selected: false,
            icon_size: egui::vec2(16.0, 16.0),
        }
    }

    pub(crate) fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }
}

impl egui::Widget for ExplorerEntry {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
        ui.horizontal(|ui| {
            let icon = ui.add(
                egui::Image::new(self.icon)
                    .fit_to_exact_size(self.icon_size)
                    .sense(egui::Sense::click()),
            );
            let label = ui.add(
                egui::Button::new("")
                    .left_text(self.title)
                    .frame(false)
                    .selected(self.selected)
                    .min_size((ui.available_width(), ui.available_height()).into()),
            );
            icon.union(label)
        })
        .inner
    }
}

/// A collapsible explorer section with an icon beside its heading.
pub(crate) struct ExplorerHeader {
    id: egui::Id,
    icon: egui::ImageSource<'static>,
    title: egui::WidgetText,
    default_open: bool,
    icon_size: egui::Vec2,
}

impl ExplorerHeader {
    pub(crate) fn new(
        id: egui::Id,
        icon: egui::ImageSource<'static>,
        title: impl Into<egui::WidgetText>,
    ) -> Self {
        Self {
            id,
            icon,
            title: title.into(),
            default_open: false,
            icon_size: egui::vec2(20.0, 20.0),
        }
    }

    pub(crate) fn default_open(mut self, default_open: bool) -> Self {
        self.default_open = default_open;
        self
    }

    pub(crate) fn show<R>(
        self,
        ui: &mut egui::Ui,
        add_contents: impl FnOnce(&mut egui::Ui) -> R,
    ) -> (
        egui::Response,
        egui::InnerResponse<egui::Response>,
        Option<egui::InnerResponse<R>>,
    ) {
        let (toggle_response, header_response, body_response) =
            egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                self.id,
                self.default_open,
            )
            .show_header(ui, |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                let icon = ui.add(
                    egui::Image::new(self.icon)
                        .fit_to_exact_size(self.icon_size)
                        .sense(egui::Sense::click()),
                );
                let label = ui.add(
                    egui::Button::new(self.title)
                        .frame(false)
                        .sense(egui::Sense::click()),
                );
                icon.union(label)
            })
            .body(add_contents);

        if header_response.inner.clicked() {
            let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                self.id,
                self.default_open,
            );
            state.toggle(ui);
            state.store(ui.ctx());
        }

        (toggle_response, header_response, body_response)
    }
}
