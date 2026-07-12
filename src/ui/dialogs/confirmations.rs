//! Destructive-action and unsaved-work confirmation dialogs.

use crate::ui::{
    state::{EditorState, UiCommand, UiProjectView},
    widgets::menu::DragableMenu,
};

/// Draw the "Save before quit?" confirmation dialog.
pub(crate) fn draw_exit_confirm_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    _editor: &mut EditorState,
) {
    let mut open = true;
    DragableMenu::new("Exit: Unsaved Changes")
        .open(&mut open)
        .min_width(280.0)
        .show(ui.ctx(), |ui| {
            ui.set_max_width(200.);
            ui.label("Save all modified PIDBs and unsaved triangulations before exiting?");
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("Save and Exit").clicked() {
                    commands.push(UiCommand::SaveAndExit);
                }
                if ui.button("Exit Without Saving").clicked() {
                    commands.push(UiCommand::ExitWithoutSaving);
                }
                if ui.button("Cancel").clicked() {
                    commands.push(UiCommand::CancelExit);
                }
            });
        });
    if !open {
        commands.push(UiCommand::CancelExit);
    }
}

/// Draw the delete-selection confirmation dialog shown when Delete/Backspace is pressed.
pub(crate) fn draw_delete_confirm_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
) {
    if !editor.delete_confirm_open {
        return;
    }
    let count = editor
        .selected_handles
        .iter()
        .filter(|h| matches!(h, crate::model::SceneEntityId::Object(_)))
        .count();
    let mut open = true;
    DragableMenu::new("Delete Objects")
        .open(&mut open)
        .min_width(160.0)
        .show(ui.ctx(), |ui| {
            ui.set_max_width(160.);
            ui.label(format!(
                "Are you sure you want to delete {count} selected item{}?",
                if count == 1 { "" } else { "s" }
            ));
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("Delete").clicked() {
                    commands.push(UiCommand::ConfirmDeleteSelection);
                    editor.delete_confirm_open = false;
                }
                if ui.button("Cancel").clicked() {
                    editor.delete_confirm_open = false;
                }
            });
        });
    if !open {
        editor.delete_confirm_open = false;
    }
}

/// Draw the confirmation dialog shown before deleting a layer and its objects.
pub(crate) fn draw_delete_layer_confirm_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
) {
    let Some((layer_id, name)) = editor.pending_delete_layer.clone() else {
        return;
    };
    let mut open = true;
    DragableMenu::new("Delete Layer")
        .open(&mut open)
        .min_width(280.0)
        .show(ui.ctx(), |ui| {
            ui.label(format!(
                "Delete layer '{name}' and all objects on it?\nThis cannot be undone."
            ));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Delete Layer").clicked() {
                    commands.push(UiCommand::DeleteLayer(layer_id));
                }
                if ui.button("Cancel").clicked() {
                    editor.pending_delete_layer = None;
                }
            });
        });
    if !open {
        editor.pending_delete_layer = None;
    }
}

/// Draw the confirmation dialog shown before closing a dirty PIDB.
pub(crate) fn draw_close_pidb_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
    project: &UiProjectView,
) {
    let Some(runtime_id) = editor.pending_close_pidb else {
        return;
    };
    let name = project
        .projects
        .iter()
        .find(|entry| entry.runtime_id == runtime_id)
        .map(|entry| entry.name.as_str())
        .unwrap_or("this PIDB");
    let mut open = true;
    DragableMenu::new("Close PIDB: Unsaved Changes")
        .open(&mut open)
        .min_width(320.0)
        .show(ui.ctx(), |ui| {
            ui.set_max_width(320.);
            ui.label(format!("Save changes to '{name}' before closing it?"));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Save and Close").clicked() {
                    commands.push(UiCommand::SaveAndClosePidb(runtime_id));
                }
                if ui.button("Close Without Saving").clicked() {
                    commands.push(UiCommand::ClosePidbForce(runtime_id));
                }
                if ui.button("Cancel").clicked() {
                    editor.pending_close_pidb = None;
                }
            });
        });
    if !open {
        editor.pending_close_pidb = None;
    }
}

pub(crate) fn draw_confirm_load_all_folder_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
) {
    let folder_path = match &editor.confirm_load_all_folder {
        Some(p) => p.clone(),
        None => return,
    };
    let folder_name = folder_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("folder")
        .to_owned();
    let mut open = true;
    DragableMenu::new("Load All Triangulations")
        .open(&mut open)
        .min_width(270.0)
        .show(ui.ctx(), |ui| {
            ui.label(format!("Load all triangulations in \"{folder_name}\"?"));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Load All").clicked() {
                    commands.push(UiCommand::LoadAllTriangulationsInFolder(
                        folder_path.clone(),
                    ));
                    editor.confirm_load_all_folder = None;
                }
                if ui.button("Cancel").clicked() {
                    editor.confirm_load_all_folder = None;
                }
            });
        });
    if !open {
        editor.confirm_load_all_folder = None;
    }
}

pub(crate) fn draw_close_unsaved_tri_dialog(
    ui: &mut egui::Ui,
    commands: &mut Vec<UiCommand>,
    editor: &mut EditorState,
) {
    let Some(tri_id) = editor.tri_close_unsaved else {
        return;
    };
    let mut open = true;
    DragableMenu::new("Unsaved Triangulation")
        .open(&mut open)
        .min_width(270.0)
        .show(ui.ctx(), |ui| {
            ui.label("This triangulation is not saved to disk.\nSave now?");
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Save As...").clicked() {
                    commands.push(UiCommand::SaveAndCloseTriangulationAs(tri_id));
                }
                if ui.button("Close Without Saving").clicked() {
                    commands.push(UiCommand::CloseTriangulationForce(tri_id));
                }
                if ui.button("Cancel").clicked() {
                    editor.tri_close_unsaved = None;
                }
            });
        });
    if !open {
        editor.tri_close_unsaved = None;
    }
}
