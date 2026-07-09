//! Independent CPU port of `block_model_volume.wgsl`, used as a numerical
//! reference for the GPU raycaster's nonuniform DDA traversal and
//! Beer–Lambert accumulation.
//!
//! This deliberately re-derives the shader's math from scratch instead of
//! sharing code with the asset builder in `gpu_cache.rs`. A reference that
//! reused the code it validates could not catch a divergence between the two;
//! the whole value here is that it is a second, independent implementation.
//! Every constant and formula mirrors a specific part of the WGSL and is
//! annotated as such — if the shader changes, this must change with it and the
//! tests below will flag the difference.
//!
//! Scope: the pure volume integral. The scene-depth occlusion test in
//! `fs_main` (which depends on external opaque geometry) is intentionally
//! omitted; everything else — cell lookup, segment lengths, alpha from
//! `sigma_t`, premultiplied accumulation, early termination and the boundary
//! highlight — is ported faithfully.

use glam::{Vec3, vec3};

/// Mirrors `EMPTY_CELL_PAYLOAD` in the WGSL.
const EMPTY_CELL_PAYLOAD: u32 = 0xffff_ffff;
/// Mirrors `FALLBACK_CELL_FLAG` in the WGSL.
const FALLBACK_CELL_FLAG: u32 = 0x8000_0000;
/// Mirrors `VISIBLE_ALPHA_EPSILON` in the WGSL.
const VISIBLE_ALPHA_EPSILON: f32 = 0.004;
/// Mirrors `VOLUME_AMBIENT_INTENSITY` in the WGSL.
const VOLUME_AMBIENT_INTENSITY: f32 = 0.68;
/// Mirrors `VOLUME_BOUNDARY_STRENGTH` in the WGSL.
const VOLUME_BOUNDARY_STRENGTH: f32 = 0.35;
/// Mirrors the WGSL `INF` constant (`3.402823e+38`, i.e. `f32::MAX`).
const INF: f32 = f32::MAX;

/// A single colour-transfer stop: straight RGBA plus its position `t`.
#[derive(Clone, Copy)]
pub(crate) struct Stop {
    pub color: [f32; 4],
    pub t: f32,
}

/// A dense coordinate-compressed block volume, matching the buffers the GPU
/// path uploads (`x/y/z_planes`, `cells`, `dims`, ramp stops and options).
pub(crate) struct VolumeReference {
    pub x_planes: Vec<f32>,
    pub y_planes: Vec<f32>,
    pub z_planes: Vec<f32>,
    /// Cell count per axis; `cells.len() == dims.x * dims.y * dims.z`.
    pub dims: [u32; 3],
    pub cells: Vec<u32>,
    pub stops: Vec<Stop>,
    pub fallback_color: [f32; 4],
    /// `options.y`: reference cell length feeding `sigma_for_alpha`.
    pub reference_len: f32,
    /// `options.z`: accumulated-opacity cutoff for early ray termination.
    pub opacity_cutoff: f32,
    /// `options.w`: maximum DDA steps.
    pub max_steps: u32,
    /// Mirrors `volume.brick_dims.w` (`BRICK_SIZE`): the macro-brick edge
    /// length in cells, used only by the brick skip.
    pub brick_size: u32,
    /// Mirrors the shader's brick fast path for empty *and* uniform bricks.
    /// `false` steps every cell — the two modes must produce identical output.
    pub skip_uniform_bricks: bool,
}

/// Result of a reference raycast: premultiplied `rgb` plus resolved `alpha`,
/// or `None` when the ray misses the volume or accumulates nothing visible
/// (the shader `discard`s in both cases).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RayResult {
    pub rgb: [f32; 3],
    pub alpha: f32,
    /// DDA loop iterations taken; lets tests assert the empty/uniform brick skip
    /// actually reduces traversal work, not just that outputs match.
    pub steps: u32,
}

impl VolumeReference {
    fn bounds_min(&self) -> Vec3 {
        vec3(self.x_planes[0], self.y_planes[0], self.z_planes[0])
    }

    fn bounds_max(&self) -> Vec3 {
        vec3(
            *self.x_planes.last().unwrap(),
            *self.y_planes.last().unwrap(),
            *self.z_planes.last().unwrap(),
        )
    }

    /// Port of `ramp_color`: hard-cutoff stops. Below the first stop is fully
    /// transparent; each stop's colour holds until the next.
    fn ramp_color(&self, t: f32) -> [f32; 4] {
        let stop_count = self.stops.len().max(2);
        let last = self.stops[stop_count - 1];
        if t < self.stops[0].t {
            return [0.0; 4];
        }
        if t >= last.t {
            return last.color;
        }
        for i in 1..stop_count {
            if t < self.stops[i].t {
                return self.stops[i - 1].color;
            }
        }
        last.color
    }

    /// Port of `color_for_payload`.
    fn color_for_payload(&self, payload: u32) -> [f32; 4] {
        if payload & FALLBACK_CELL_FLAG != 0 {
            return self.fallback_color;
        }
        let grade = (payload & 0xffff) as f32 / 65535.0;
        self.ramp_color(grade)
    }

    /// Port of `sigma_for_alpha`: invert a per-reference-cell alpha into an
    /// absorption coefficient so opacity scales with path length.
    fn sigma_for_alpha(&self, alpha: f32) -> f32 {
        let ref_len = self.reference_len.max(1.0e-6);
        if alpha >= 0.98 {
            return -(0.001_f32.ln()) / ref_len;
        }
        -((1.0 - alpha).max(0.001).ln()) / ref_len
    }

    fn cell_payload(&self, i: u32, j: u32, k: u32) -> u32 {
        let index = (k * self.dims[1] + j) * self.dims[0] + i;
        self.cells[index as usize]
    }

    /// The cell-index range `[lo, hi)` of the brick containing `cell` on one
    /// axis, clipped to the grid like the shader's `min(bx0 + bs, dims)`.
    fn brick_range(&self, cell: u32, dim: u32) -> (u32, u32) {
        let lo = (cell / self.brick_size) * self.brick_size;
        (lo, (lo + self.brick_size).min(dim))
    }

    /// The appearance class a payload resolves to for uniformity purposes:
    /// `None` when the cell is invisible (empty, or ramp alpha below the
    /// epsilon), otherwise its exact resolved RGBA. Mirrors the builder's
    /// brick-flag rule: two cells are interchangeable to the march exactly
    /// when these classes are equal — same accumulation, and crossings
    /// between them can never emit a highlight.
    fn appearance_class(&self, payload: u32) -> Option<[f32; 4]> {
        if payload == EMPTY_CELL_PAYLOAD {
            return None;
        }
        let color = self.color_for_payload(payload);
        if color[3] < VISIBLE_ALPHA_EPSILON {
            return None;
        }
        Some(color)
    }

    /// Whether every cell in the brick containing `(i, j, k)` resolves to one
    /// appearance class. The GPU path answers this with the per-occupied-brick
    /// `UNIFORM_BRICK_FLAG` in `brick_info` (plus `EMPTY_BRICK` for unoccupied
    /// bricks); scanning the dense grid is the independent equivalent.
    fn brick_is_uniform(&self, i: u32, j: u32, k: u32) -> bool {
        let (x0, x1) = self.brick_range(i, self.dims[0]);
        let (y0, y1) = self.brick_range(j, self.dims[1]);
        let (z0, z1) = self.brick_range(k, self.dims[2]);
        let first = self.appearance_class(self.cell_payload(x0, y0, z0));
        (z0..z1).all(|bk| {
            (y0..y1).all(|bj| {
                (x0..x1).all(|bi| self.appearance_class(self.cell_payload(bi, bj, bk)) == first)
            })
        })
    }

    /// Port of `intersect_aabb`; returns `(t_enter, t_exit)`.
    fn intersect_aabb(&self, origin: Vec3, dir: Vec3) -> (f32, f32) {
        let inv = Vec3::ONE / dir;
        let t0 = (self.bounds_min() - origin) * inv;
        let t1 = (self.bounds_max() - origin) * inv;
        let lo = t0.min(t1);
        let hi = t0.max(t1);
        (lo.x.max(lo.y).max(lo.z), hi.x.min(hi.y).min(hi.z))
    }

    /// Port of `fs_main`'s ray march. `origin`/`dir` are in the volume's local
    /// space; `dir` need not be normalized.
    pub(crate) fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<RayResult> {
        let dir = dir.normalize();
        let (t_enter, t_exit) = self.intersect_aabb(origin, dir);
        if t_exit < t_enter.max(0.0) {
            return None;
        }

        let mut t = t_enter.max(0.0);
        let seed = origin + dir * (t + 1.0e-5);
        let mut i = find_cell(&self.x_planes, self.dims[0], seed.x);
        let mut j = find_cell(&self.y_planes, self.dims[1], seed.y);
        let mut k = find_cell(&self.z_planes, self.dims[2], seed.z);
        let mut color = Vec3::ZERO;
        let mut transmittance = 1.0f32;
        let mut steps = 0u32;

        for _ in 0..self.max_steps {
            if t >= t_exit || (1.0 - transmittance) >= self.opacity_cutoff {
                break;
            }
            steps += 1;
            let p = origin + dir * t;

            // Port of the shader's brick-granular fast path for uniform
            // bricks (which includes all-empty bricks: every cell resolves to
            // the invisible class). One iteration consumes the whole brick:
            // its single appearance integrates in one exp over the span —
            // exact, because Beer–Lambert transmittance is multiplicative —
            // then a rim highlight fires at the brick's exit face exactly
            // where per-cell stepping would emit its only highlight (interior
            // same-appearance crossings never rim), and the march advances
            // across the face. The shader reads the appearance from the
            // brick's aggregate; for a uniform brick the aggregate is the
            // appearance, so deriving it from the class here is equivalent
            // and independent.
            if self.skip_uniform_bricks && self.brick_is_uniform(i, j, k) {
                let (bx0, bx1) = self.brick_range(i, self.dims[0]);
                let (by0, by1) = self.brick_range(j, self.dims[1]);
                let (bz0, bz1) = self.brick_range(k, self.dims[2]);
                let class = self.appearance_class(self.cell_payload(bx0, by0, bz0));
                let exit_plane = |planes: &[f32], lo: u32, hi: u32, d: f32| {
                    planes[if d > 0.0 { hi } else { lo } as usize]
                };
                let axis_t = |planes: &[f32], lo: u32, hi: u32, p_axis: f32, d: f32| {
                    if d.abs() <= 1.0e-8 {
                        INF
                    } else {
                        (exit_plane(planes, lo, hi, d) - p_axis) / d
                    }
                };
                let btx = axis_t(&self.x_planes, bx0, bx1, p.x, dir.x);
                let bty = axis_t(&self.y_planes, by0, by1, p.y, dir.y);
                let btz = axis_t(&self.z_planes, bz0, bz1, p.z, dir.z);
                let t_brick_exit = t + btx.min(bty).min(btz).max(0.0);

                if let Some(rgba) = class {
                    let sigma = self.sigma_for_alpha(rgba[3].clamp(0.0, 1.0));
                    let segment_len = (t_brick_exit.min(t_exit) - t).max(0.0);
                    if sigma > 0.0 && segment_len > 0.0 {
                        let alpha = (1.0 - (-sigma * segment_len).exp()).clamp(0.0, 1.0);
                        color += transmittance
                            * alpha
                            * vec3(rgba[0], rgba[1], rgba[2])
                            * VOLUME_AMBIENT_INTENSITY;
                        transmittance *= 1.0 - alpha;
                    }
                }

                if t_brick_exit >= t_exit {
                    t = t_exit;
                    continue;
                }

                // Advance across the brick face: crossed axes step explicitly
                // (guaranteeing progress), the others re-derive their cell
                // index from the exit point; leaving the grid ends the march.
                let bcx = btx <= bty + 1.0e-6 && btx <= btz + 1.0e-6;
                let bcy = bty <= btx + 1.0e-6 && bty <= btz + 1.0e-6;
                let bcz = btz <= btx + 1.0e-6 && btz <= bty + 1.0e-6;
                let q = origin + dir * (t_brick_exit + 1.0e-5);
                let mut departed = false;
                let mut advance =
                    |crossed: bool, d: f32, lo: u32, hi: u32, dim: u32, fallback: u32| -> u32 {
                        if !crossed {
                            return fallback;
                        }
                        if d > 0.0 {
                            if hi >= dim {
                                departed = true;
                                return 0;
                            }
                            hi
                        } else if lo == 0 {
                            departed = true;
                            0
                        } else {
                            lo - 1
                        }
                    };
                let ni = advance(
                    bcx,
                    dir.x,
                    bx0,
                    bx1,
                    self.dims[0],
                    find_cell(&self.x_planes, self.dims[0], q.x),
                );
                let nj = advance(
                    bcy,
                    dir.y,
                    by0,
                    by1,
                    self.dims[1],
                    find_cell(&self.y_planes, self.dims[1], q.y),
                );
                let nk = advance(
                    bcz,
                    dir.z,
                    bz0,
                    bz1,
                    self.dims[2],
                    find_cell(&self.z_planes, self.dims[2], q.z),
                );
                if departed {
                    t = t_exit;
                    continue;
                }

                // Rim highlight at the brick's exit face; mirrors the guards
                // and colour rules of the shader's brick path.
                if (1.0 - transmittance) < self.opacity_cutoff && t_brick_exit < t_exit {
                    let neighbor = self.appearance_class(self.cell_payload(ni, nj, nk));
                    let highlight_rgb = match (class, neighbor) {
                        (Some(cur), Some(nbr)) => {
                            let cur = vec3(cur[0], cur[1], cur[2]);
                            let nbr = vec3(nbr[0], nbr[1], nbr[2]);
                            if cur.distance(nbr) > 0.05 {
                                Some((cur + nbr) * 0.5)
                            } else {
                                None
                            }
                        }
                        (Some(cur), None) => Some(vec3(cur[0], cur[1], cur[2])),
                        (None, Some(nbr)) => Some(vec3(nbr[0], nbr[1], nbr[2])),
                        (None, None) => None,
                    };
                    if let Some(highlight_rgb) = highlight_rgb {
                        let boundary_normal = if bcx {
                            vec3(if dir.x > 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0)
                        } else if bcy {
                            vec3(0.0, if dir.y > 0.0 { 1.0 } else { -1.0 }, 0.0)
                        } else {
                            vec3(0.0, 0.0, if dir.z > 0.0 { 1.0 } else { -1.0 })
                        };
                        let fresnel = (1.0 - (-dir).dot(boundary_normal).abs())
                            .clamp(0.0, 1.0)
                            .powf(4.0);
                        let highlight =
                            (highlight_rgb + Vec3::splat(0.15)).clamp(Vec3::ZERO, Vec3::ONE);
                        color += transmittance * fresnel * VOLUME_BOUNDARY_STRENGTH * highlight;
                    }
                }

                i = ni;
                j = nj;
                k = nk;
                t = t_brick_exit.max(t + 1.0e-6);
                continue;
            }

            let tx = next_plane_t(&self.x_planes, i, p.x, dir.x);
            let ty = next_plane_t(&self.y_planes, j, p.y, dir.y);
            let tz = next_plane_t(&self.z_planes, k, p.z, dir.z);
            let step_t = tx.min(ty).min(tz).max(0.0);
            let t_next = (t + step_t).min(t_exit);
            let segment_len = (t_next - t).max(0.0);

            // Fetched once per step; the boundary-highlight call reuses it
            // (mirrors the shader's `current_payload`/`current_color` hoist).
            let current_payload = self.cell_payload(i, j, k);
            let current_color = self.color_for_payload(current_payload);

            if segment_len > 0.0
                && current_payload != EMPTY_CELL_PAYLOAD
                && current_color[3] >= VISIBLE_ALPHA_EPSILON
            {
                let sigma = self.sigma_for_alpha(current_color[3].clamp(0.0, 1.0));
                let alpha = (1.0 - (-sigma * segment_len).exp()).clamp(0.0, 1.0);
                let contribution = transmittance * alpha;
                color += contribution
                    * vec3(current_color[0], current_color[1], current_color[2])
                    * VOLUME_AMBIENT_INTENSITY;
                transmittance *= 1.0 - alpha;
            }

            let crossed_x = tx <= ty + 1.0e-6 && tx <= tz + 1.0e-6;
            let crossed_y = ty <= tx + 1.0e-6 && ty <= tz + 1.0e-6;
            let crossed_z = tz <= tx + 1.0e-6 && tz <= ty + 1.0e-6;

            // The `t + step_t < t_exit` guard mirrors the shader: crossings
            // at/behind the march end are not visible boundaries.
            if (1.0 - transmittance) < self.opacity_cutoff && t + step_t < t_exit {
                self.accumulate_boundary_highlight(
                    &mut color,
                    transmittance,
                    dir,
                    current_payload,
                    current_color,
                    i,
                    j,
                    k,
                    crossed_x,
                    crossed_y,
                    crossed_z,
                );
            }

            if crossed_x {
                if dir.x > 0.0 {
                    if i + 1 >= self.dims[0] {
                        t = t_exit;
                    } else {
                        i += 1;
                    }
                } else if i == 0 {
                    t = t_exit;
                } else {
                    i -= 1;
                }
            }
            if crossed_y {
                if dir.y > 0.0 {
                    if j + 1 >= self.dims[1] {
                        t = t_exit;
                    } else {
                        j += 1;
                    }
                } else if j == 0 {
                    t = t_exit;
                } else {
                    j -= 1;
                }
            }
            if crossed_z {
                if dir.z > 0.0 {
                    if k + 1 >= self.dims[2] {
                        t = t_exit;
                    } else {
                        k += 1;
                    }
                } else if k == 0 {
                    t = t_exit;
                } else {
                    k -= 1;
                }
            }
            if !crossed_x && !crossed_y && !crossed_z {
                t += 1.0e-5;
            } else {
                t = t_next.max(t + 1.0e-6);
            }
        }

        let mut alpha_out = 1.0 - transmittance;
        if alpha_out <= 0.0001 {
            return None;
        }
        // Mirrors the shader's opaque snap: a march cut short by the opacity
        // cutoff *inside* the medium unpremultiplies its colour and reports
        // fully opaque, so no background bleeds through solid models. Rays
        // that exit the volume keep their true partial alpha.
        if alpha_out >= self.opacity_cutoff && t < t_exit {
            color /= alpha_out.max(1.0e-4);
            alpha_out = 1.0;
        }
        Some(RayResult {
            rgb: [color.x, color.y, color.z],
            alpha: alpha_out,
            steps,
        })
    }

    /// Port of the boundary-highlight block: a Fresnel-weighted rim added when
    /// the ray crosses into a differently-coloured neighbour, or across the
    /// model's glass surface (a visible↔invisible crossing), which rims in the
    /// visible side's own colour.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_boundary_highlight(
        &self,
        color: &mut Vec3,
        transmittance: f32,
        dir: Vec3,
        current_payload: u32,
        current: [f32; 4],
        i: u32,
        j: u32,
        k: u32,
        crossed_x: bool,
        crossed_y: bool,
        crossed_z: bool,
    ) {
        let mut boundary_normal = Vec3::ZERO;
        let mut neighbor: Option<u32> = None;
        if crossed_x {
            if dir.x > 0.0 {
                if i + 1 < self.dims[0] {
                    neighbor = Some(self.cell_payload(i + 1, j, k));
                }
            } else if i > 0 {
                neighbor = Some(self.cell_payload(i - 1, j, k));
            }
            boundary_normal = vec3(if dir.x > 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0);
        } else if crossed_y {
            if dir.y > 0.0 {
                if j + 1 < self.dims[1] {
                    neighbor = Some(self.cell_payload(i, j + 1, k));
                }
            } else if j > 0 {
                neighbor = Some(self.cell_payload(i, j - 1, k));
            }
            boundary_normal = vec3(0.0, if dir.y > 0.0 { 1.0 } else { -1.0 }, 0.0);
        } else if crossed_z {
            if dir.z > 0.0 {
                if k + 1 < self.dims[2] {
                    neighbor = Some(self.cell_payload(i, j, k + 1));
                }
            } else if k > 0 {
                neighbor = Some(self.cell_payload(i, j, k - 1));
            }
            boundary_normal = vec3(0.0, 0.0, if dir.z > 0.0 { 1.0 } else { -1.0 });
        }

        let Some(neighbor_payload) = neighbor else {
            return;
        };
        let neighbor_color = if neighbor_payload == EMPTY_CELL_PAYLOAD {
            [0.0; 4]
        } else {
            self.color_for_payload(neighbor_payload)
        };
        let cur_rgb = vec3(current[0], current[1], current[2]);
        let nbr_rgb = vec3(neighbor_color[0], neighbor_color[1], neighbor_color[2]);
        let current_visible =
            current_payload != EMPTY_CELL_PAYLOAD && current[3] >= VISIBLE_ALPHA_EPSILON;
        let neighbor_visible =
            neighbor_payload != EMPTY_CELL_PAYLOAD && neighbor_color[3] >= VISIBLE_ALPHA_EPSILON;
        let highlight_rgb = match (current_visible, neighbor_visible) {
            (true, true) => {
                if cur_rgb.distance(nbr_rgb) <= 0.05 {
                    return;
                }
                (cur_rgb + nbr_rgb) * 0.5
            }
            (true, false) => cur_rgb,
            (false, true) => nbr_rgb,
            (false, false) => return,
        };
        let fresnel = (1.0 - (-dir).dot(boundary_normal).abs())
            .clamp(0.0, 1.0)
            .powf(4.0);
        let highlight = (highlight_rgb + Vec3::splat(0.15)).clamp(Vec3::ZERO, Vec3::ONE);
        *color += transmittance * fresnel * VOLUME_BOUNDARY_STRENGTH * highlight;
    }
}

/// Port of `find_x_cell`/`find_y_cell`/`find_z_cell`: index of the last plane
/// `<= value`, clamped to a valid cell index.
fn find_cell(planes: &[f32], dim: u32, value: f32) -> u32 {
    let mut lo = 0u32;
    let mut hi = dim + 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if planes[mid as usize] <= value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        return 0;
    }
    (lo - 1).min(dim - 1)
}

/// Distance along the ray to the next plane crossed on one axis, or `INF` when
/// the ray is parallel to that axis. Mirrors the `select(...)` expressions in
/// `fs_main`.
fn next_plane_t(planes: &[f32], cell: u32, p_axis: f32, dir_axis: f32) -> f32 {
    if dir_axis.abs() <= 1.0e-8 {
        return INF;
    }
    let plane = if dir_axis > 0.0 {
        planes[(cell + 1) as usize]
    } else {
        planes[cell as usize]
    };
    (plane - p_axis) / dir_axis
}