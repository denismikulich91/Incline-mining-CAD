//! World-to-screen projection updates for UI/tool overlays.

use super::*;

impl<'a> Graphics<'a> {
    pub(super) fn update_tool_projections(&self, editor: &mut EditorState, document: &Document) {
        if editor.offset_awaiting_side_pick {
            let vp = self.view_proj();
            let screen = self.screen_size();
            editor.offset_source_screen_px = editor
                .offset_source_world
                .iter()
                .filter_map(|&p| crate::rendering::pick::world_to_screen(&vp, p, screen))
                .map(|sp| (sp.x as f32, sp.y as f32))
                .collect();
            editor.offset_preview_screen_px = editor
                .offset_preview_world
                .iter()
                .filter_map(|&p| crate::rendering::pick::world_to_screen(&vp, p, screen))
                .map(|sp| (sp.x as f32, sp.y as f32))
                .collect();
        } else {
            editor.offset_source_screen_px.clear();
            editor.offset_preview_screen_px.clear();
        }

        if editor.batter_berm_dialog_open && !editor.batter_berm_rings_world.is_empty() {
            let vp = self.view_proj();
            let screen = self.screen_size();
            editor.batter_berm_source_screen_px = editor
                .batter_berm_source_world
                .iter()
                .map(|&p| {
                    crate::rendering::pick::world_to_screen(&vp, p, screen)
                        .map(|sp| (sp.x as f32, sp.y as f32))
                })
                .collect();
            editor.batter_berm_rings_screen_px = editor
                .batter_berm_rings_world
                .iter()
                .map(|ring| {
                    ring.iter()
                        .map(|&p| {
                            crate::rendering::pick::world_to_screen(&vp, p, screen)
                                .map(|sp| (sp.x as f32, sp.y as f32))
                        })
                        .collect()
                })
                .collect();
            editor.batter_berm_guides_screen_px = editor
                .batter_berm_guides_world
                .iter()
                .map(|&(from, to)| {
                    (
                        crate::rendering::pick::world_to_screen(&vp, from, screen)
                            .map(|sp| (sp.x as f32, sp.y as f32)),
                        crate::rendering::pick::world_to_screen(&vp, to, screen)
                            .map(|sp| (sp.x as f32, sp.y as f32)),
                    )
                })
                .collect();
        } else {
            editor.batter_berm_source_screen_px.clear();
            editor.batter_berm_rings_screen_px.clear();
            editor.batter_berm_guides_screen_px.clear();
        }

        use crate::ui::state::ActiveTool;
        if editor.active_tool == ActiveTool::MakeRoad {
            let vp = self.view_proj();
            let screen = self.screen_size();
            let project = |p: DVec3| {
                crate::rendering::pick::world_to_screen(&vp, p, screen)
                    .map(|sp| (sp.x as f32, sp.y as f32))
            };
            // Prefer the resolved ghost centerline (kept fresh by
            // update_road_preview); fall back to the raw stroke when the
            // current cursor position is invalid.
            let mut center_preview: Vec<DVec3> = editor.road_preview_center_world.clone();
            if center_preview.is_empty() {
                center_preview = editor.pending_stroke.clone();
                if let Some(cursor) = editor.cursor_world {
                    center_preview.push(cursor);
                }
            }
            editor.road_preview_center_screen_px =
                center_preview.iter().map(|&p| project(p)).collect();
            editor.road_preview_left_screen_px = editor
                .road_preview_left_world
                .iter()
                .map(|&p| project(p))
                .collect();
            editor.road_preview_right_screen_px = editor
                .road_preview_right_world
                .iter()
                .map(|&p| project(p))
                .collect();
        } else {
            editor.road_preview_center_screen_px.clear();
            editor.road_preview_left_screen_px.clear();
            editor.road_preview_right_screen_px.clear();
        }

        if editor.active_tool == ActiveTool::Move
            && (!editor.selected_handles.is_empty() || editor.move_vertex_target.is_some())
        {
            let mut sum = DVec3::ZERO;
            let mut count = 0usize;
            if let Some((target_id, vertex_index)) = editor.move_vertex_target {
                if let Some(obj) = document.get_object(target_id) {
                    match obj {
                        Object::Polyline { verts, .. } => {
                            if let Some(vertex) = verts.get(vertex_index) {
                                sum += vertex.pos;
                                count += 1;
                            }
                        }
                        Object::Point { pos, .. } if vertex_index == 0 => {
                            sum += *pos;
                            count += 1;
                        }
                        _ => {}
                    }
                }
            } else {
                for &handle in &editor.selected_handles {
                    if let SceneEntityId::Object(id) = handle
                        && let Some(obj) = document.get_object(id)
                    {
                        match obj {
                            Object::Polyline { verts, .. }
                            | Object::Road {
                                centerline: verts, ..
                            } => {
                                for vertex in verts {
                                    sum += vertex.pos;
                                    count += 1;
                                }
                            }
                            Object::Point { pos, .. } | Object::Text { pos, .. } => {
                                sum += *pos;
                                count += 1;
                            }
                        }
                    }
                }
            }
            if count > 0 {
                let c = sum / count as f64;
                let vp = self.view_proj();
                let sz = self.screen_size();
                const GIZMO_PX: f64 = 70.0;
                let project = |w: DVec3| -> Option<(f32, f32)> {
                    crate::rendering::pick::world_to_screen(&vp, w, sz)
                        .map(|s| (s.x as f32, s.y as f32))
                };
                let raw_px = |axis: DVec3| -> f64 {
                    if let (Some(c_px), Some(t_px)) = (project(c), project(c + axis)) {
                        let dx = (t_px.0 - c_px.0) as f64;
                        let dy = (t_px.1 - c_px.1) as f64;
                        (dx * dx + dy * dy).sqrt()
                    } else {
                        1.0
                    }
                };
                let tip_px = |axis: DVec3| -> Option<(f32, f32)> {
                    let c_px = project(c)?;
                    let t_px = project(c + axis)?;
                    let dx = (t_px.0 - c_px.0) as f64;
                    let dy = (t_px.1 - c_px.1) as f64;
                    let len = (dx * dx + dy * dy).sqrt();
                    if len < 1e-4 {
                        return None;
                    }
                    let nx = (dx / len * GIZMO_PX) as f32;
                    let ny = (dy / len * GIZMO_PX) as f32;
                    Some((c_px.0 + nx, c_px.1 + ny))
                };
                editor.move_gizmo_center_px = project(c);
                editor.move_gizmo_x_tip_px = tip_px(DVec3::X);
                editor.move_gizmo_y_tip_px = tip_px(DVec3::Y);
                editor.move_gizmo_z_tip_px = tip_px(DVec3::Z);
                editor.move_gizmo_x_px_per_world = raw_px(DVec3::X).max(0.001);
                editor.move_gizmo_y_px_per_world = raw_px(DVec3::Y).max(0.001);
                editor.move_gizmo_z_px_per_world = raw_px(DVec3::Z).max(0.001);
            } else {
                editor.move_gizmo_center_px = None;
                editor.move_gizmo_x_tip_px = None;
                editor.move_gizmo_y_tip_px = None;
                editor.move_gizmo_z_tip_px = None;
            }
        } else {
            editor.move_gizmo_center_px = None;
            editor.move_gizmo_x_tip_px = None;
            editor.move_gizmo_y_tip_px = None;
            editor.move_gizmo_z_tip_px = None;
        }

        if editor.active_tool == ActiveTool::Chamfer {
            use crate::app::commands::drawing::chamfer::chamfer_corner;
            let vp = self.view_proj();
            let screen = self.screen_size();
            let project = |w: DVec3| -> Option<(f32, f32)> {
                crate::rendering::pick::world_to_screen(&vp, w, screen)
                    .map(|s| (s.x as f32, s.y as f32))
            };

            let corner_data = editor.chamfer_poly_id.and_then(|oid| {
                let ci = editor.chamfer_corner_index?;
                if let Some(Object::Polyline {
                    verts,
                    closed: true,
                    ..
                }) = document.get_object(oid)
                {
                    Some((verts.clone(), ci))
                } else {
                    None
                }
            });

            if let Some((ref verts, ci)) = corner_data {
                use crate::app::commands::drawing::chamfer::chamfer_max_radius;
                editor.chamfer_max_radius = chamfer_max_radius(verts, ci);
                editor.chamfer_radius = editor.chamfer_radius.min(editor.chamfer_max_radius);
                let chamfered =
                    chamfer_corner(verts, ci, editor.chamfer_radius, editor.chamfer_segments);
                editor.chamfer_preview_screen_px =
                    chamfered.iter().filter_map(|v| project(v.pos)).collect();
                editor.chamfer_hover_corner_px = None;

                let n = verts.len();
                let corner = verts[ci].pos;
                let next = verts[(ci + 1) % n].pos;
                let c_px = project(corner);
                let n_px = project(next);
                let edge_stub_dir = if let (Some(cp), Some(np)) = (c_px, n_px) {
                    let dx = np.0 - cp.0;
                    let dy = np.1 - cp.1;
                    let l = (dx * dx + dy * dy).sqrt().max(1e-6);
                    Some((dx / l, dy / l))
                } else {
                    None
                };
                editor.chamfer_gizmo_bisector_px = edge_stub_dir;

                let edge_2d = glam::DVec2::new(next.x - corner.x, next.y - corner.y);
                let edge_len = edge_2d.length();
                if edge_len > 1e-10 {
                    let edge_dir = edge_2d / edge_len;
                    let handle_world = DVec3::new(
                        corner.x + edge_dir.x * editor.chamfer_radius,
                        corner.y + edge_dir.y * editor.chamfer_radius,
                        corner.z,
                    );
                    editor.chamfer_gizmo_corner_px = c_px;
                    editor.chamfer_gizmo_handle_px = project(handle_world);
                    if let (Some(cp), Some(np)) = (c_px, n_px) {
                        let dx = np.0 - cp.0;
                        let dy = np.1 - cp.1;
                        let screen_edge_len = (dx * dx + dy * dy).sqrt();
                        if screen_edge_len > 1e-4 {
                            editor.chamfer_gizmo_edge_screen_dir =
                                Some((dx / screen_edge_len, dy / screen_edge_len));
                            editor.chamfer_gizmo_px_per_world = screen_edge_len as f64 / edge_len;
                        }
                    }
                } else {
                    editor.chamfer_gizmo_corner_px = None;
                    editor.chamfer_gizmo_handle_px = None;
                }
            } else {
                editor.chamfer_preview_screen_px.clear();
                editor.chamfer_gizmo_corner_px = None;
                editor.chamfer_gizmo_handle_px = None;
                editor.chamfer_gizmo_bisector_px = None;
                editor.chamfer_gizmo_edge_screen_dir = None;
                editor.chamfer_max_radius = f64::MAX;
            }
        } else {
            editor.chamfer_preview_screen_px.clear();
            editor.chamfer_hover_corner_px = None;
            editor.chamfer_gizmo_corner_px = None;
            editor.chamfer_gizmo_handle_px = None;
            editor.chamfer_gizmo_bisector_px = None;
            editor.chamfer_gizmo_edge_screen_dir = None;
            editor.chamfer_gizmo_drag_start_px = None;
            editor.chamfer_gizmo_hovered = false;
            editor.chamfer_max_radius = f64::MAX;
        }

        if editor.active_tool == ActiveTool::Bezier {
            use crate::app::commands::drawing::bezier::bezier_eval;
            let vp = self.view_proj();
            let screen = self.screen_size();
            let project = |w: DVec3| -> Option<(f32, f32)> {
                crate::rendering::pick::world_to_screen(&vp, w, screen)
                    .map(|s| (s.x as f32, s.y as f32))
            };

            if let Some(oid) = editor.bezier_poly_id {
                if let Some(Object::Polyline {
                    verts,
                    closed: true,
                    ..
                }) = document.get_object(oid)
                {
                    let verts = verts.clone();
                    let n = verts.len();

                    editor.bezier_poly_verts_screen_px =
                        verts.iter().filter_map(|v| project(v.pos)).collect();

                    if let [Some(vi), Some(vj)] = editor.bezier_selected_verts {
                        let cp1 = DVec3::from(editor.bezier_cp1);
                        let cp2 = DVec3::from(editor.bezier_cp2);
                        editor.bezier_cp1_screen_px = project(cp1);
                        editor.bezier_cp2_screen_px = project(cp2);

                        let v_start = verts[vi].pos;
                        let v_end = verts[vj].pos;
                        let segs = editor.bezier_segments.max(2) as usize;

                        let mut preview: Vec<DVec3> = Vec::with_capacity(n + segs);
                        for (k, vert) in verts.iter().enumerate().take(n) {
                            preview.push(vert.pos);
                            if k == vi {
                                for s in 1..segs {
                                    let t = s as f64 / segs as f64;
                                    preview.push(bezier_eval(v_start, cp1, cp2, v_end, t));
                                }
                            }
                        }
                        editor.bezier_preview_screen_px =
                            preview.iter().filter_map(|&p| project(p)).collect();
                    } else {
                        editor.bezier_cp1_screen_px = None;
                        editor.bezier_cp2_screen_px = None;
                        editor.bezier_preview_screen_px.clear();
                    }
                } else {
                    editor.bezier_poly_verts_screen_px.clear();
                    editor.bezier_cp1_screen_px = None;
                    editor.bezier_cp2_screen_px = None;
                    editor.bezier_preview_screen_px.clear();
                }
            } else {
                editor.bezier_poly_verts_screen_px.clear();
                editor.bezier_cp1_screen_px = None;
                editor.bezier_cp2_screen_px = None;
                editor.bezier_preview_screen_px.clear();
            }
        } else {
            editor.bezier_poly_verts_screen_px.clear();
            editor.bezier_cp1_screen_px = None;
            editor.bezier_cp2_screen_px = None;
            editor.bezier_preview_screen_px.clear();
            editor.bezier_hover_cp = None;
            editor.bezier_dragging_cp = None;
        }

        {
            use crate::ui::state::{RelimitMode, TrimEnd};
            let show = editor.relimit_dialog_open
                && matches!(
                    editor.relimit_mode,
                    RelimitMode::AbsoluteLength | RelimitMode::RelativeLength
                );
            if show {
                editor.relimit_resize_end_px = editor.relimit_source_id.and_then(|oid| {
                    if let Some(Object::Polyline { verts, .. }) = document.get_object(oid) {
                        let vp = self.view_proj();
                        let screen = self.screen_size();
                        let v = match editor.relimit_resize_end {
                            TrimEnd::Start => verts.first()?.pos,
                            TrimEnd::End => verts.last()?.pos,
                        };
                        crate::rendering::pick::world_to_screen(&vp, v, screen)
                            .map(|s| (s.x as f32, s.y as f32))
                    } else {
                        None
                    }
                });
            } else {
                editor.relimit_resize_end_px = None;
            }
        }
    }
}
