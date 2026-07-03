use crate::{
    model::block_model::{
        BlockModelId, OpenBlockModel, numeric_variable_default, render_value_range,
    },
    ui::{
        EditorState, UiCommand,
        state::OreFilterMode,
        widgets::menu::{DragableMenu, MenuFieldCombo, MenuFieldText},
    },
};

const TABLE_PAGE_SIZE: usize = 100;

/// The active colour variable's name and render range for `model`, sharing
/// the model's own decoded-values cache with the renderer. `None` when
/// there's no active variable or no usable range (e.g. every block has the
/// sentinel/default value).
pub(crate) fn active_color_scale(model: &OpenBlockModel) -> Option<(String, f64, f64)> {
    let name = model.active_numeric_variable.clone()?;
    let default = model
        .model
        .variable(&name)
        .and_then(numeric_variable_default);
    let values = model.active_numeric_values()?;
    let (min, max) = render_value_range(&values, &model.renderable_block_indices, default)?;
    Some((name, min, max))
}

pub(crate) fn draw_block_model_table(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    block_models: &[OpenBlockModel],
    id: BlockModelId,
) {
    egui::CentralPanel::default().show(ui, |ui| {
        draw_block_model_table_contents(ui, editor, block_models, id);
    });
}

fn draw_block_model_table_contents(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    block_models: &[OpenBlockModel],
    id: BlockModelId,
) {
    let Some(model) = block_models.iter().find(|model| model.id == id) else {
        ui.centered_and_justified(|ui| {
            ui.label("This block model is no longer loaded");
        });
        return;
    };

    ui.heading(&model.name);
    ui.add_space(6.0);
    egui::Grid::new(("block_model_metadata", id))
        .num_columns(2)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            ui.label("Model");
            ui.add(egui::Label::new(model.source.bmf_path.display().to_string()).truncate());
            ui.end_row();
            ui.label("Definition");
            ui.add(
                egui::Label::new(
                    model
                        .source
                        .bdf_path
                        .as_ref()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "No .bdf attached".to_owned()),
                )
                .truncate(),
            );
            ui.end_row();
            ui.label("Blocks");
            ui.label(model.model.metadata.n_blocks.to_string());
            ui.end_row();
            ui.label("Origin");
            let origin = model.model.metadata.origin;
            ui.label(format!("{:.3}, {:.3}, {:.3}", origin.x, origin.y, origin.z));
            ui.end_row();
            ui.label("Orientation");
            let orientation = model.model.metadata.orientation;
            ui.label(format!(
                "{:.3}, {:.3}, {:.3}",
                orientation.x, orientation.y, orientation.z
            ));
            ui.end_row();
        });

    ui.separator();
    ui.heading("Schemas");
    egui::Grid::new(("block_model_schemas", id))
        .striped(true)
        .show(ui, |ui| {
            ui.strong("Name");
            ui.strong("Dims");
            ui.strong("Lower");
            ui.strong("Upper");
            ui.strong("Size range");
            ui.end_row();
            for schema in &model.model.metadata.schemas {
                ui.label(&schema.name);
                ui.label(format!(
                    "{} x {} x {}",
                    schema.dims[0], schema.dims[1], schema.dims[2]
                ));
                ui.label(format!(
                    "{:.3}, {:.3}, {:.3}",
                    schema.lower.x, schema.lower.y, schema.lower.z
                ));
                ui.label(format!(
                    "{:.3}, {:.3}, {:.3}",
                    schema.upper.x, schema.upper.y, schema.upper.z
                ));
                ui.label(format!(
                    "{:.3}-{:.3}, {:.3}-{:.3}, {:.3}-{:.3}",
                    schema.min_size.x,
                    schema.max_size.x,
                    schema.min_size.y,
                    schema.max_size.y,
                    schema.min_size.z,
                    schema.max_size.z
                ));
                ui.end_row();
            }
        });

    ui.separator();
    ui.heading("Variables");
    egui::ScrollArea::vertical()
        .id_salt(("block_model_vars_scroll", id))
        .max_height(160.0)
        .show(ui, |ui| {
            egui::Grid::new(("block_model_vars", id))
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Name");
                    ui.strong("Type");
                    ui.strong("Location");
                    ui.strong("Dictionary");
                    ui.strong("Description");
                    ui.end_row();
                    for variable in &model.model.metadata.variables {
                        ui.label(&variable.name);
                        if crate::model::formats::bmf::BmfModel::variable_type_supported(
                            &variable.physical_type,
                        ) {
                            ui.label(&variable.physical_type);
                        } else {
                            ui.colored_label(
                                egui::Color32::from_rgb(220, 150, 30),
                                format!("{} (unsupported)", variable.physical_type),
                            );
                        }
                        ui.label(variable.location.to_string());
                        ui.add(egui::Label::new(format!(
                            "{} entries",
                            variable.strings.len()
                        )));
                        ui.add(egui::Label::new(&variable.description).truncate());
                        ui.end_row();
                    }
                });
        });

    if let Some(bdf) = &model.bdf {
        ui.separator();
        ui.heading("Definition Sections");
        egui::ScrollArea::vertical()
            .id_salt(("block_model_definitions_scroll", id))
            .max_height(120.0)
            .show(ui, |ui| {
                for section in &bdf.sections {
                    ui.label(format!(
                        "{} ({} fields)",
                        section.name,
                        section.fields.len()
                    ));
                }
            });
    }

    ui.separator();
    ui.horizontal(|ui| {
        let max_page = model.model.metadata.n_blocks.saturating_sub(1) / TABLE_PAGE_SIZE;
        let page = editor.block_model_table_pages.entry(id).or_insert(0);
        *page = (*page).min(max_page);
        if ui.button("Previous").clicked() {
            *page = page.saturating_sub(1);
        }
        ui.label(format!("Page {} / {}", *page + 1, max_page + 1));
        if ui.button("Next").clicked() {
            *page = (*page + 1).min(max_page);
        }
    });

    let page = editor
        .block_model_table_pages
        .get(&id)
        .copied()
        .unwrap_or(0);
    let start = page * TABLE_PAGE_SIZE;
    let end = (start + TABLE_PAGE_SIZE).min(model.model.metadata.n_blocks);
    // Decode only the visible page of each variable; whole-variable decodes
    // of a large model just to show 100 rows made opening this tab slow.
    let decoded: Vec<(&str, Option<Vec<f64>>)> = model
        .model
        .numeric_variables()
        .iter()
        .map(|var| {
            (
                var.name.as_str(),
                model.model.numeric_values_range(&var.name, start, end).ok(),
            )
        })
        .collect();
    egui::ScrollArea::both()
        .id_salt(("block_model_values_scroll", id))
        .show(ui, |ui| {
            egui::Grid::new(("block_model_values", id))
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Block");
                    for (name, _) in &decoded {
                        ui.strong(*name);
                    }
                    ui.end_row();
                    for row in start..end {
                        ui.label(row.to_string());
                        for (_, values) in &decoded {
                            let text = values
                                .as_ref()
                                .and_then(|values| values.get(row - start))
                                .map(|value| format!("{value:.4}"))
                                .unwrap_or_else(|| "-".to_owned());
                            ui.label(text);
                        }
                        ui.end_row();
                    }
                });
        });
}

pub(crate) fn draw_ore_triangulation_dialog(
    ui: &mut egui::Ui,
    editor: &mut EditorState,
    block_models: &[OpenBlockModel],
    commands: &mut Vec<UiCommand>,
) {
    let mut open = true;
    DragableMenu::new("Create Ore Triangulation")
        .open(&mut open)
        .min_width(340.0)
        .show(ui.ctx(), |ui| {
            let selected_label = editor
                .ore_block_model_id
                .and_then(|id| block_models.iter().find(|model| model.id == id))
                .map(|model| model.name.as_str())
                .unwrap_or("Choose a block model");
            MenuFieldCombo::new(
                "ore_block_model",
                "Block model",
                &mut editor.ore_block_model_id,
                selected_label,
                block_models
                    .iter()
                    .map(|model| (Some(model.id), model.name.clone().into())),
            )
            .width(230.0)
            .show(ui);

            let variables: Vec<String> = editor
                .ore_block_model_id
                .and_then(|id| block_models.iter().find(|model| model.id == id))
                .map(|model| {
                    model
                        .model
                        .numeric_variables()
                        .into_iter()
                        .filter(|var| !var.special)
                        .map(|var| var.name.clone())
                        .collect()
                })
                .unwrap_or_default();
            if !variables.is_empty() && !variables.contains(&editor.ore_variable) {
                editor.ore_variable = variables[0].clone();
            }
            let variable_label = if editor.ore_variable.is_empty() {
                "Choose a numeric variable".to_owned()
            } else {
                editor.ore_variable.clone()
            };
            MenuFieldCombo::new(
                "ore_variable",
                "Variable",
                &mut editor.ore_variable,
                variable_label.as_str(),
                variables
                    .iter()
                    .map(|name| (name.clone(), name.clone().into())),
            )
            .width(230.0)
            .show(ui);

            let mode_label = match editor.ore_filter_mode {
                OreFilterMode::GreaterOrEqual => ">= threshold",
                OreFilterMode::LessOrEqual => "<= threshold",
                OreFilterMode::Between => "Between",
            };
            MenuFieldCombo::new(
                "ore_filter_mode",
                "Filter",
                &mut editor.ore_filter_mode,
                mode_label,
                [
                    (OreFilterMode::GreaterOrEqual, ">= threshold".into()),
                    (OreFilterMode::LessOrEqual, "<= threshold".into()),
                    (OreFilterMode::Between, "Between".into()),
                ],
            )
            .width(160.0)
            .show(ui);

            MenuFieldText::new("Threshold / min", &mut editor.ore_min_input)
                .width(120.0)
                .show(ui);
            if editor.ore_filter_mode == OreFilterMode::Between {
                MenuFieldText::new("Max", &mut editor.ore_max_input)
                    .width(120.0)
                    .show(ui);
            }
            MenuFieldText::new("Output name", &mut editor.ore_name_input)
                .width(220.0)
                .show(ui);

            let min = editor.ore_min_input.trim().parse::<f64>().ok();
            let max = editor.ore_max_input.trim().parse::<f64>().ok();
            let ready = editor.ore_block_model_id.is_some()
                && !editor.ore_variable.is_empty()
                && !editor.ore_name_input.trim().is_empty()
                && min.is_some()
                && (editor.ore_filter_mode != OreFilterMode::Between || max.is_some());
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    editor.ore_triangulation_open = false;
                }
                if ui.add_enabled(ready, egui::Button::new("Create")).clicked() {
                    commands.push(UiCommand::ExecuteCreateOreTriangulation {
                        block_model_id: editor.ore_block_model_id.unwrap(),
                        variable: editor.ore_variable.clone(),
                        mode: editor.ore_filter_mode,
                        min: min.unwrap(),
                        max: max.unwrap_or(min.unwrap()),
                        name: editor.ore_name_input.trim().to_owned(),
                    });
                }
            });
        });
    if !open {
        editor.ore_triangulation_open = false;
    }
}
