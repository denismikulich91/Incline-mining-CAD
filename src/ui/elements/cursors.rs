//! Canvas overlay helpers: orbit marker, cursor highlights, and view gizmos.

/// Draw the orbit indicator (compass rose) on the canvas.
///
/// The marker is clipped to `clip_rect` so it doesn't bleed over panels.
/// Returns early if the position is outside the clip area.
/// Draw a screen-space world orientation gizmo in the bottom-left of the viewport.
pub(crate) fn draw_orientation_gizmo(
    ui: &mut egui::Ui,
    canvas_rect: egui::Rect,
    camera_forward: [f32; 3],
    camera_up: [f32; 3],
) {
    if canvas_rect.width() < 80.0 || canvas_rect.height() < 80.0 {
        return;
    }

    let forward = normalize3(camera_forward).unwrap_or([0.0, 0.0, -1.0]);
    let up = normalize3(camera_up).unwrap_or([0.0, 1.0, 0.0]);
    let right = normalize3(cross3(forward, up)).unwrap_or([1.0, 0.0, 0.0]);

    const MARGIN: f32 = 22.0;
    const AXIS_LEN: f32 = 34.0;
    const MIN_AXIS_LEN: f32 = 8.0;
    let origin = egui::pos2(
        canvas_rect.left() + MARGIN + AXIS_LEN,
        canvas_rect.bottom() - MARGIN - AXIS_LEN,
    );

    let axes = [
        ([1.0, 0.0, 0.0], "X", egui::Color32::from_rgb(235, 60, 55)),
        ([0.0, 1.0, 0.0], "Y", egui::Color32::from_rgb(90, 220, 90)),
        ([0.0, 0.0, 1.0], "Z", egui::Color32::from_rgb(70, 120, 245)),
    ];

    let mut projected: Vec<_> = axes
        .into_iter()
        .map(|(axis, label, color)| {
            let screen = egui::vec2(dot3(axis, right), -dot3(axis, up));
            let depth = dot3(axis, forward);
            (axis, label, color, screen, depth)
        })
        .collect();
    projected.sort_by(|a, b| a.4.total_cmp(&b.4));

    let mut painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Middle,
        egui::Id::new("world_orientation_gizmo"),
    ));
    painter.set_clip_rect(canvas_rect);

    for (_axis, label, color, screen, depth) in projected {
        let length = screen.length();
        let toward_camera = depth < -0.35;
        let dir = if length >= 0.001 {
            screen / length
        } else {
            egui::vec2(0.0, -1.0)
        };
        let visible_len = (AXIS_LEN * length).max(if toward_camera { MIN_AXIS_LEN } else { 0.0 });
        let tip = origin + dir * visible_len;
        let alpha = if length < 0.08 { 150 } else { 235 };
        let draw_color =
            egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);

        if visible_len >= 1.0 {
            painter.line_segment([origin, tip], egui::Stroke::new(2.25, draw_color));
            painter.circle_filled(tip, 4.0, draw_color);
        } else {
            painter.circle_filled(origin, 4.0, draw_color);
        }

        let label_pos = if visible_len >= 1.0 {
            tip + dir * 7.0 + egui::vec2(0.0, -5.0)
        } else {
            origin + egui::vec2(6.0, -16.0)
        };
        painter.text(
            label_pos,
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(13.0),
            draw_color,
        );
    }

    painter.circle_filled(
        origin,
        4.0,
        egui::Color32::from_rgba_unmultiplied(245, 245, 245, 230),
    );
}

pub(crate) fn draw_view_cube(
    ui: &mut egui::Ui,
    canvas_rect: egui::Rect,
    camera_forward: [f32; 3],
    camera_up: [f32; 3],
    commands: &mut Vec<crate::ui::state::UiCommand>,
) {
    if canvas_rect.width() < 180.0 || canvas_rect.height() < 96.0 {
        return;
    }

    const SIZE: f32 = 72.0;
    const MARGIN: f32 = 22.0;
    const GIZMO_AXIS_LEN: f32 = 34.0;
    const GAP: f32 = 18.0;
    let pos = egui::pos2(
        canvas_rect.left() + MARGIN + GIZMO_AXIS_LEN - SIZE * 0.5,
        canvas_rect.bottom() - MARGIN - GIZMO_AXIS_LEN * 2.0 - GAP - SIZE,
    );
    egui::Area::new(egui::Id::new("view_cube_overlay"))
        .order(egui::Order::Foreground)
        .fixed_pos(pos)
        .show(ui.ctx(), |ui| {
            let (rect, response) =
                ui.allocate_exact_size(egui::vec2(SIZE, SIZE), egui::Sense::click());
            let forward = normalize3(camera_forward).unwrap_or([0.0, 0.0, -1.0]);
            let up = normalize3(camera_up).unwrap_or([0.0, 1.0, 0.0]);
            let right = normalize3(cross3(forward, up)).unwrap_or([1.0, 0.0, 0.0]);
            let center = rect.center();
            let scale = 20.5;
            let project = |point: [f32; 3]| -> egui::Pos2 {
                center + egui::vec2(dot3(point, right) * scale, -dot3(point, up) * scale)
            };

            let vertices = [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ];
            let faces = [
                ViewCubeFace {
                    indices: [4, 5, 6, 7],
                    normal: [0.0, 0.0, 1.0],
                    u: [1.0, 0.0, 0.0],
                    v: [0.0, 1.0, 0.0],
                    label: "UP",
                    view: crate::ui::state::StandardView::Up,
                },
                ViewCubeFace {
                    indices: [0, 3, 2, 1],
                    normal: [0.0, 0.0, -1.0],
                    u: [1.0, 0.0, 0.0],
                    v: [0.0, -1.0, 0.0],
                    label: "DOWN",
                    view: crate::ui::state::StandardView::Down,
                },
                ViewCubeFace {
                    indices: [2, 3, 7, 6],
                    normal: [0.0, 1.0, 0.0],
                    u: [-1.0, 0.0, 0.0],
                    v: [0.0, 0.0, 1.0],
                    label: "NORTH",
                    view: crate::ui::state::StandardView::North,
                },
                ViewCubeFace {
                    indices: [0, 1, 5, 4],
                    normal: [0.0, -1.0, 0.0],
                    u: [1.0, 0.0, 0.0],
                    v: [0.0, 0.0, 1.0],
                    label: "SOUTH",
                    view: crate::ui::state::StandardView::South,
                },
                ViewCubeFace {
                    indices: [0, 4, 7, 3],
                    normal: [-1.0, 0.0, 0.0],
                    u: [0.0, -1.0, 0.0],
                    v: [0.0, 0.0, 1.0],
                    label: "WEST",
                    view: crate::ui::state::StandardView::West,
                },
                ViewCubeFace {
                    indices: [1, 2, 6, 5],
                    normal: [1.0, 0.0, 0.0],
                    u: [0.0, 1.0, 0.0],
                    v: [0.0, 0.0, 1.0],
                    label: "EAST",
                    view: crate::ui::state::StandardView::East,
                },
            ];

            let hover_pos = response.hover_pos();
            let mut drawn_faces: Vec<_> = faces
                .into_iter()
                .map(|face| {
                    let depth = dot3(face.normal, forward);
                    let points = face.indices.map(|idx| project(vertices[idx]));
                    (face, depth, points)
                })
                .collect();
            drawn_faces.sort_by(|a, b| b.1.total_cmp(&a.1));

            let hovered_view = hover_pos.and_then(|pos| {
                drawn_faces
                    .iter()
                    .rev()
                    .find(|(_, depth, points)| *depth < -0.02 && point_in_polygon(pos, points))
                    .map(|(face, _, _)| face.view)
            });

            let painter = ui.painter_at(rect);

            for (face, depth, points) in drawn_faces {
                if depth >= -0.02 {
                    continue;
                }
                let is_hovered = hovered_view == Some(face.view);
                let fill = view_cube_face_color(face.normal, depth, is_hovered);
                painter.add(egui::Shape::convex_polygon(
                    points.to_vec(),
                    fill,
                    egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(25, 30, 36, 150)),
                ));
                draw_projected_label(&painter, face, &project);
            }

            if response.clicked()
                && let Some(view) = hovered_view
            {
                commands.push(crate::ui::state::UiCommand::SetStandardView(view));
            }
        });
}

pub(crate) fn draw_orbit_marker(ui: &mut egui::Ui, ox: f32, oy: f32, clip_rect: egui::Rect) {
    let ppp = ui.ctx().pixels_per_point();
    let pos = egui::pos2(ox / ppp, oy / ppp);
    if !clip_rect.contains(pos) {
        return;
    }
    let mut painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("orbit_marker"),
    ));
    painter.set_clip_rect(clip_rect);
    let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(255, 180, 0, 220));
    let r = 4.0;
    painter.circle_stroke(pos, r, stroke);
    painter.line_segment(
        [
            pos - egui::vec2(r + 4.0, 0.0),
            pos + egui::vec2(r + 4.0, 0.0),
        ],
        stroke,
    );
    painter.line_segment(
        [
            pos - egui::vec2(0.0, r + 4.0),
            pos + egui::vec2(0.0, r + 4.0),
        ],
        stroke,
    );
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = dot3(v, v).sqrt();
    (len > f32::EPSILON).then(|| [v[0] / len, v[1] / len, v[2] / len])
}

#[derive(Clone, Copy)]
struct ViewCubeFace {
    indices: [usize; 4],
    normal: [f32; 3],
    u: [f32; 3],
    v: [f32; 3],
    label: &'static str,
    view: crate::ui::state::StandardView,
}

fn view_cube_face_color(normal: [f32; 3], depth: f32, hovered: bool) -> egui::Color32 {
    let base = if normal[2].abs() > 0.5 {
        [70_u8, 120, 245]
    } else if normal[0].abs() > 0.5 {
        [235_u8, 60, 55]
    } else {
        [90_u8, 220, 90]
    };
    let shade =
        (0.62 + (-depth).clamp(0.0, 1.0) * 0.28 + if hovered { 0.16 } else { 0.0 }).clamp(0.0, 1.0);
    egui::Color32::from_rgb(
        ((base[0] as f32 * shade).min(255.0)) as u8,
        ((base[1] as f32 * shade).min(255.0)) as u8,
        ((base[2] as f32 * shade).min(255.0)) as u8,
    )
}

fn draw_projected_label(
    painter: &egui::Painter,
    face: ViewCubeFace,
    project: &impl Fn([f32; 3]) -> egui::Pos2,
) {
    let chars: Vec<_> = face.label.chars().collect();
    let char_width = 0.17_f32;
    let char_height = 0.28_f32;
    let spacing = 0.055_f32;
    let total_width =
        chars.len() as f32 * char_width + chars.len().saturating_sub(1) as f32 * spacing;
    let left = -total_width * 0.5;
    let stroke = egui::Stroke::new(1.45, egui::Color32::from_rgba_unmultiplied(26, 30, 36, 155));

    for (index, ch) in chars.into_iter().enumerate() {
        let x_offset = left + index as f32 * (char_width + spacing);
        for (from, to) in glyph_segments(ch) {
            let local_from = [
                x_offset + from[0] * char_width,
                (0.5 - from[1]) * char_height,
            ];
            let local_to = [x_offset + to[0] * char_width, (0.5 - to[1]) * char_height];
            painter.line_segment(
                [
                    project(face_point(face, local_from)),
                    project(face_point(face, local_to)),
                ],
                stroke,
            );
        }
    }
}

fn face_point(face: ViewCubeFace, local: [f32; 2]) -> [f32; 3] {
    [
        face.normal[0] * 1.012 + face.u[0] * local[0] + face.v[0] * local[1],
        face.normal[1] * 1.012 + face.u[1] * local[0] + face.v[1] * local[1],
        face.normal[2] * 1.012 + face.u[2] * local[0] + face.v[2] * local[1],
    ]
}

fn glyph_segments(ch: char) -> &'static [([f32; 2], [f32; 2])] {
    match ch {
        'A' => &[
            ([0.0, 1.0], [0.5, 0.0]),
            ([0.5, 0.0], [1.0, 1.0]),
            ([0.22, 0.58], [0.78, 0.58]),
        ],
        'B' => &[
            ([0.0, 0.0], [0.0, 1.0]),
            ([0.0, 0.0], [0.72, 0.0]),
            ([0.72, 0.0], [0.9, 0.2]),
            ([0.9, 0.2], [0.72, 0.43]),
            ([0.0, 0.43], [0.72, 0.43]),
            ([0.72, 0.43], [0.95, 0.68]),
            ([0.95, 0.68], [0.72, 1.0]),
            ([0.0, 1.0], [0.72, 1.0]),
        ],
        'C' => &[
            ([0.92, 0.08], [0.72, 0.0]),
            ([0.72, 0.0], [0.18, 0.0]),
            ([0.18, 0.0], [0.0, 0.22]),
            ([0.0, 0.22], [0.0, 0.78]),
            ([0.0, 0.78], [0.18, 1.0]),
            ([0.18, 1.0], [0.72, 1.0]),
            ([0.72, 1.0], [0.92, 0.9]),
        ],
        'D' => &[
            ([0.0, 0.0], [0.0, 1.0]),
            ([0.0, 0.0], [0.68, 0.0]),
            ([0.68, 0.0], [0.95, 0.28]),
            ([0.95, 0.28], [0.95, 0.72]),
            ([0.95, 0.72], [0.68, 1.0]),
            ([0.68, 1.0], [0.0, 1.0]),
        ],
        'E' => &[
            ([0.9, 0.0], [0.0, 0.0]),
            ([0.0, 0.0], [0.0, 1.0]),
            ([0.0, 0.5], [0.72, 0.5]),
            ([0.0, 1.0], [0.9, 1.0]),
        ],
        'F' => &[
            ([0.0, 0.0], [0.0, 1.0]),
            ([0.0, 0.0], [0.9, 0.0]),
            ([0.0, 0.5], [0.72, 0.5]),
        ],
        'G' => &[
            ([0.92, 0.12], [0.72, 0.0]),
            ([0.72, 0.0], [0.18, 0.0]),
            ([0.18, 0.0], [0.0, 0.22]),
            ([0.0, 0.22], [0.0, 0.78]),
            ([0.0, 0.78], [0.18, 1.0]),
            ([0.18, 1.0], [0.78, 1.0]),
            ([0.78, 1.0], [0.94, 0.78]),
            ([0.94, 0.78], [0.94, 0.58]),
            ([0.94, 0.58], [0.55, 0.58]),
        ],
        'H' => &[
            ([0.0, 0.0], [0.0, 1.0]),
            ([1.0, 0.0], [1.0, 1.0]),
            ([0.0, 0.5], [1.0, 0.5]),
        ],
        'I' => &[
            ([0.12, 0.0], [0.88, 0.0]),
            ([0.5, 0.0], [0.5, 1.0]),
            ([0.12, 1.0], [0.88, 1.0]),
        ],
        'K' => &[
            ([0.0, 0.0], [0.0, 1.0]),
            ([1.0, 0.0], [0.0, 0.52]),
            ([0.0, 0.52], [1.0, 1.0]),
        ],
        'L' => &[([0.0, 0.0], [0.0, 1.0]), ([0.0, 1.0], [0.9, 1.0])],
        'M' => &[
            ([0.0, 1.0], [0.0, 0.0]),
            ([0.0, 0.0], [0.5, 0.52]),
            ([0.5, 0.52], [1.0, 0.0]),
            ([1.0, 0.0], [1.0, 1.0]),
        ],
        'N' => &[
            ([0.0, 1.0], [0.0, 0.0]),
            ([0.0, 0.0], [1.0, 1.0]),
            ([1.0, 1.0], [1.0, 0.0]),
        ],
        'O' => &[
            ([0.2, 0.0], [0.8, 0.0]),
            ([0.8, 0.0], [1.0, 0.22]),
            ([1.0, 0.22], [1.0, 0.78]),
            ([1.0, 0.78], [0.8, 1.0]),
            ([0.8, 1.0], [0.2, 1.0]),
            ([0.2, 1.0], [0.0, 0.78]),
            ([0.0, 0.78], [0.0, 0.22]),
            ([0.0, 0.22], [0.2, 0.0]),
        ],
        'P' => &[
            ([0.0, 1.0], [0.0, 0.0]),
            ([0.0, 0.0], [0.75, 0.0]),
            ([0.75, 0.0], [0.95, 0.24]),
            ([0.95, 0.24], [0.75, 0.5]),
            ([0.75, 0.5], [0.0, 0.5]),
        ],
        'R' => &[
            ([0.0, 1.0], [0.0, 0.0]),
            ([0.0, 0.0], [0.75, 0.0]),
            ([0.75, 0.0], [0.95, 0.24]),
            ([0.95, 0.24], [0.75, 0.5]),
            ([0.75, 0.5], [0.0, 0.5]),
            ([0.45, 0.5], [1.0, 1.0]),
        ],
        'T' => &[([0.0, 0.0], [1.0, 0.0]), ([0.5, 0.0], [0.5, 1.0])],
        'S' => &[
            ([0.92, 0.08], [0.72, 0.0]),
            ([0.72, 0.0], [0.18, 0.0]),
            ([0.18, 0.0], [0.0, 0.2]),
            ([0.0, 0.2], [0.0, 0.42]),
            ([0.0, 0.42], [0.22, 0.5]),
            ([0.22, 0.5], [0.72, 0.5]),
            ([0.72, 0.5], [0.95, 0.62]),
            ([0.95, 0.62], [0.95, 0.82]),
            ([0.95, 0.82], [0.72, 1.0]),
            ([0.72, 1.0], [0.18, 1.0]),
            ([0.18, 1.0], [0.0, 0.9]),
        ],
        'U' => &[
            ([0.0, 0.0], [0.0, 0.78]),
            ([0.0, 0.78], [0.2, 1.0]),
            ([0.2, 1.0], [0.8, 1.0]),
            ([0.8, 1.0], [1.0, 0.78]),
            ([1.0, 0.78], [1.0, 0.0]),
        ],
        'W' => &[
            ([0.0, 0.0], [0.0, 1.0]),
            ([0.0, 1.0], [0.5, 0.48]),
            ([0.5, 0.48], [1.0, 1.0]),
            ([1.0, 1.0], [1.0, 0.0]),
        ],
        _ => &[],
    }
}

fn point_in_polygon(point: egui::Pos2, polygon: &[egui::Pos2; 4]) -> bool {
    let mut inside = false;
    let mut previous = polygon.len() - 1;
    for current in 0..polygon.len() {
        let a = polygon[current];
        let b = polygon[previous];
        if (a.y > point.y) != (b.y > point.y) {
            let x = (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x;
            if point.x < x {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}
