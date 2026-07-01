//! Tab bar: Preferences, Console, Documentation, Workspace.

use std::sync::Mutex;

use lazy_static::lazy_static;

use crate::{
    model::block_model::{BlockModelId, OpenBlockModel},
    ui::{themed_icon, unthemed_icon},
};

lazy_static! {
    static ref TAB_MANAGER: Mutex<TabManager> = Mutex::new(TabManager::new());
}

/// Semantic category of a tab.  Used to open the correct tab by identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TabClass {
    Workspace,
    Preferences,
    BlockModelTable(BlockModelId),
}

/// Single tab entry stored in the tab manager.
#[derive(Clone)]
struct Tab {
    id: usize,
    title: String,
    class: TabClass,
    closable: bool,
}

struct TabManager {
    tabs: Vec<Tab>,
    active_tab_id: usize,
    next_tab_id: usize,
}

impl TabManager {
    fn new() -> Self {
        let tabs = vec![
            Tab {
                id: 0,
                title: "Preferences".to_owned(),
                class: TabClass::Preferences,
                closable: false,
            },
            Tab {
                id: 1,
                title: "Workspace".to_owned(),
                class: TabClass::Workspace,
                closable: false,
            },
        ];
        let next_tab_id = tabs.len();

        Self {
            tabs,
            active_tab_id: 1,
            next_tab_id,
        }
    }

    fn open(&mut self, title: impl Into<String>, class: TabClass) {
        self.open_with_close_button(title, class, false);
    }

    fn open_with_close_button(
        &mut self,
        title: impl Into<String>,
        class: TabClass,
        closable: bool,
    ) {
        let title = title.into();
        if let Some(tab) = self.tabs.iter_mut().find(|tab| tab.class == class) {
            tab.title = title;
            tab.closable = closable;
            self.active_tab_id = tab.id;
            return;
        }

        let id = self.next_tab_id;
        self.next_tab_id += 1;
        self.tabs.push(Tab {
            id,
            title,
            class,
            closable,
        });
        self.active_tab_id = id;
    }

    fn close(&mut self, tab_id: usize) {
        let Some(index) = self
            .tabs
            .iter()
            .position(|tab| tab.id == tab_id && tab.closable)
        else {
            return;
        };
        let was_active = self.active_tab_id == tab_id;
        self.tabs.remove(index);
        if was_active {
            self.active_tab_id = 2;
        }
    }

    fn sync_block_model_tables(&mut self, block_models: &[OpenBlockModel]) {
        let loaded: std::collections::HashSet<_> =
            block_models.iter().map(|model| model.id).collect();
        let active_removed = self
            .tabs
            .iter()
            .find(|tab| tab.id == self.active_tab_id)
            .is_some_and(|tab| match tab.class {
                TabClass::BlockModelTable(id) => !loaded.contains(&id),
                _ => false,
            });

        self.tabs.retain(|tab| match tab.class {
            TabClass::BlockModelTable(id) => loaded.contains(&id),
            _ => true,
        });

        for model in block_models {
            let class = TabClass::BlockModelTable(model.id);
            if let Some(tab) = self.tabs.iter_mut().find(|tab| tab.class == class) {
                tab.title.clone_from(&model.name);
                tab.closable = true;
            }
        }

        if active_removed {
            self.active_tab_id = 2;
        }
    }

    /// Return the class of the currently active tab.
    fn active_class(&self) -> TabClass {
        self.tabs
            .iter()
            .find(|tab| tab.id == self.active_tab_id)
            .map_or(TabClass::Workspace, |tab| tab.class.clone())
    }
}

/// Open (or focus) the Preferences tab.
pub(crate) fn open_preferences() {
    TAB_MANAGER
        .lock()
        .expect("tab manager mutex poisoned")
        .open("Preferences", TabClass::Preferences);
}

pub(crate) fn open_block_model_table(id: BlockModelId, title: impl Into<String>) {
    let mut manager = TAB_MANAGER.lock().expect("tab manager mutex poisoned");
    manager.open_with_close_button(title, TabClass::BlockModelTable(id), true);
}

pub(crate) fn sync_block_model_table_tabs(block_models: &[OpenBlockModel]) {
    TAB_MANAGER
        .lock()
        .expect("tab manager mutex poisoned")
        .sync_block_model_tables(block_models);
}

/// Draw the tab selector bar panel.
///
/// Returns the panel's bounding rect and the class of the currently active tab.
pub(crate) fn draw_tabs(ui: &mut egui::Ui) -> (egui::Rect, TabClass) {
    let panel_height =
        ui.spacing().interact_size.y + egui::Frame::side_top_panel(ui.style()).inner_margin.sum().y;
    let response = egui::Panel::top("tab_selector")
        .resizable(false)
        .exact_size(panel_height)
        .show(ui, |ui| {
            let mut manager = TAB_MANAGER.lock().expect("tab manager mutex poisoned");
            let tabs = manager.tabs.clone();
            let available = ui.available_rect_before_wrap();
            let tab_height = ui.spacing().interact_size.y;
            let mut tab_x = available.left();

            for tab in tabs {
                let width =
                    40.0 + 7.8 * tab.title.len() as f32 + if tab.closable { 22.0 } else { 0.0 };
                let rect = egui::Rect::from_min_size(
                    egui::pos2(tab_x, available.top()),
                    egui::vec2(width, tab_height),
                );
                let selected = manager.active_tab_id == tab.id;
                let tab_response = tab_entry(ui, &tab, rect, selected);
                if tab_response.close_clicked {
                    manager.close(tab.id);
                } else if tab_response.response.clicked() {
                    manager.active_tab_id = tab.id;
                }
                tab_x = rect.right() + 1.0;
            }

            manager.active_class()
        });

    (response.response.rect, response.inner)
}

struct TabEntryResponse {
    response: egui::Response,
    close_clicked: bool,
}

fn tab_entry(ui: &mut egui::Ui, tab: &Tab, rect: egui::Rect, selected: bool) -> TabEntryResponse {
    let icon = match tab.class {
        TabClass::Workspace => unthemed_icon!("workspace.svg"),
        TabClass::Preferences => themed_icon!(ui, "gear.svg"),
        TabClass::BlockModelTable(_) => unthemed_icon!("block_model.svg"),
    };
    let close_width = if tab.closable { 22.0 } else { 0.0 };
    let tab_rect =
        egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x - close_width, rect.max.y));
    let response = ui.interact(
        tab_rect,
        ui.make_persistent_id(("tab", tab.id)),
        egui::Sense::click(),
    );

    let background = if !selected && response.hovered() {
        ui.visuals().widgets.hovered.bg_fill
    } else {
        egui::Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, 0.0, background);
    if selected {
        let fill_color = if ui.visuals().dark_mode {
            egui::Color32::from_rgb(21, 21, 21)
        } else {
            egui::Color32::from_rgb(240, 240, 240)
        };
        ui.painter().rect_filled(rect, 3., fill_color);
    }

    let icon_rect = egui::Align2::LEFT_CENTER
        .align_size_within_rect(egui::vec2(16.0, 16.0), rect.shrink2(egui::vec2(8.0, 0.0)));
    ui.put(
        icon_rect,
        egui::Image::new(icon).fit_to_exact_size(icon_rect.size()),
    );
    ui.painter().text(
        egui::pos2(rect.left() + 34.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        &tab.title,
        egui::TextStyle::Button.resolve(ui.style()),
        ui.visuals().text_color(),
    );

    let close_clicked = if tab.closable {
        let close_rect = egui::Rect::from_center_size(
            egui::pos2(rect.right() - 11.0, rect.center().y),
            egui::vec2(18.0, 18.0),
        );
        let close_response = ui.interact(
            close_rect,
            ui.make_persistent_id(("tab_close", tab.id)),
            egui::Sense::click(),
        );
        if close_response.hovered() {
            ui.painter()
                .rect_filled(close_rect, 3.0, ui.visuals().widgets.hovered.bg_fill);
        }
        let icon_rect =
            egui::Align2::CENTER_CENTER.align_size_within_rect(egui::vec2(12.0, 12.0), close_rect);
        ui.put(
            icon_rect,
            egui::Image::new(unthemed_icon!("exit.svg")).fit_to_exact_size(icon_rect.size()),
        );
        close_response.clicked()
    } else {
        false
    };

    TabEntryResponse {
        response,
        close_clicked,
    }
}
