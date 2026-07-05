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

#![cfg(test)]

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
    /// DDA loop iterations taken; lets tests assert the empty-space skip
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid with one material filling every cell at the given straight-alpha,
    /// laid out as `dims` cells spanning `[0, extent]` on each axis.
    fn uniform_grid(
        dims: [u32; 3],
        extent: f32,
        alpha: f32,
        reference_len: f32,
    ) -> VolumeReference {
        let planes = |n: u32| {
            (0..=n)
                .map(|c| extent * c as f32 / n as f32)
                .collect::<Vec<_>>()
        };
        let cell_count = (dims[0] * dims[1] * dims[2]) as usize;
        VolumeReference {
            x_planes: planes(dims[0]),
            y_planes: planes(dims[1]),
            z_planes: planes(dims[2]),
            dims,
            // grade 1.0 -> quantized 65535 -> ramp lookup returns the (single)
            // stop colour, whose alpha we set below.
            cells: vec![0xffff; cell_count],
            stops: vec![
                Stop {
                    color: [1.0, 0.0, 0.0, alpha],
                    t: 0.0,
                },
                Stop {
                    color: [1.0, 0.0, 0.0, alpha],
                    t: 1.0,
                },
            ],
            fallback_color: [0.5, 0.5, 0.5, 1.0],
            reference_len,
            opacity_cutoff: 0.999,
            max_steps: 4096,
            brick_size: 8,
            skip_uniform_bricks: false,
        }
    }

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn ray_missing_the_volume_returns_none() {
        let grid = uniform_grid([1, 1, 1], 10.0, 1.0, 10.0);
        // Parallel to the box, well outside it on y.
        let result = grid.raycast(vec3(-5.0, 50.0, 5.0), vec3(1.0, 0.0, 0.0));
        assert!(
            result.is_none(),
            "a ray that never enters the box must miss"
        );
    }

    #[test]
    fn single_block_matches_beer_lambert_analytically() {
        // One 10-unit cell, fully opaque material, reference length 10.
        let grid = uniform_grid([1, 1, 1], 10.0, 1.0, 10.0);
        let result = grid
            .raycast(vec3(-5.0, 5.0, 5.0), vec3(1.0, 0.0, 0.0))
            .expect("axis ray through the cube must hit");

        // sigma = -ln(0.001)/10; alpha = 1 - exp(-sigma * 10) = 0.999.
        let sigma = -(0.001_f32.ln()) / 10.0;
        let expected_alpha = 1.0 - (-sigma * 10.0).exp();
        assert!(
            approx(result.alpha, expected_alpha, 1.0e-4),
            "alpha {} != expected {}",
            result.alpha,
            expected_alpha
        );
        // Premultiplied red channel: contribution * 1.0 * ambient.
        let expected_r = expected_alpha * 1.0 * VOLUME_AMBIENT_INTENSITY;
        assert!(
            approx(result.rgb[0], expected_r, 1.0e-4),
            "red {} != expected {}",
            result.rgb[0],
            expected_r
        );
        assert!(approx(result.rgb[1], 0.0, 1.0e-6));
    }

    #[test]
    fn opacity_increases_with_path_length() {
        // Partially transparent so it never saturates. Compare an axis ray
        // (path = 10) against a body-diagonal ray (path = 10*sqrt(3)).
        let grid = uniform_grid([1, 1, 1], 10.0, 0.5, 10.0);
        let axis = grid
            .raycast(vec3(-5.0, 5.0, 5.0), vec3(1.0, 0.0, 0.0))
            .expect("axis ray hits");
        let diagonal = grid
            .raycast(vec3(-5.0, -5.0, -5.0), vec3(1.0, 1.0, 1.0))
            .expect("diagonal ray hits");
        assert!(
            diagonal.alpha > axis.alpha + 0.05,
            "longer path should be more opaque: diagonal {} vs axis {}",
            diagonal.alpha,
            axis.alpha
        );

        // The axis case is analytic: sigma = -ln(0.5)/10, path 10.
        let sigma = -(0.5_f32.ln()) / 10.0;
        let expected_axis = 1.0 - (-sigma * 10.0).exp();
        assert!(
            approx(axis.alpha, expected_axis, 1.0e-4),
            "axis alpha {} != expected {}",
            axis.alpha,
            expected_axis
        );
    }

    #[test]
    fn subdividing_a_homogeneous_block_is_invariant() {
        // Beer–Lambert transmittance is multiplicative, so splitting one cell
        // into N cells of the same material (holding reference_len fixed) must
        // give an identical result — no seams from the cell subdivision.
        let one = uniform_grid([1, 1, 1], 12.0, 0.5, 6.0);
        let three = uniform_grid([3, 1, 1], 12.0, 0.5, 6.0);
        let a = one
            .raycast(vec3(-5.0, 6.0, 6.0), vec3(1.0, 0.0, 0.0))
            .expect("single-cell hit");
        let b = three
            .raycast(vec3(-5.0, 6.0, 6.0), vec3(1.0, 0.0, 0.0))
            .expect("three-cell hit");
        assert!(
            approx(a.alpha, b.alpha, 1.0e-4),
            "subdivision changed alpha: {} vs {}",
            a.alpha,
            b.alpha
        );
        assert!(
            approx(a.rgb[0], b.rgb[0], 1.0e-4),
            "subdivision changed colour: {} vs {}",
            a.rgb[0],
            b.rgb[0]
        );
    }

    #[test]
    fn empty_cells_contribute_nothing() {
        // Two identical grids of 3 cells along x. In `gap`, the middle cell is
        // empty; in `solid`, it is filled. The empty cell adds geometric
        // distance but no absorption, so `gap` must be strictly less opaque.
        let mut gap = uniform_grid([3, 1, 1], 12.0, 0.5, 4.0);
        gap.cells[1] = EMPTY_CELL_PAYLOAD;
        let solid = uniform_grid([3, 1, 1], 12.0, 0.5, 4.0);

        let gap_hit = gap
            .raycast(vec3(-5.0, 6.0, 6.0), vec3(1.0, 0.0, 0.0))
            .expect("gap hit");
        let solid_hit = solid
            .raycast(vec3(-5.0, 6.0, 6.0), vec3(1.0, 0.0, 0.0))
            .expect("solid hit");
        assert!(
            gap_hit.alpha < solid_hit.alpha - 0.05,
            "empty cell should absorb nothing: gap {} vs solid {}",
            gap_hit.alpha,
            solid_hit.alpha
        );

        // The gap result must equal two material cells of length 4 each:
        // transmittance = exp(-sigma*4)^2.
        let sigma = -(0.5_f32.ln()) / 4.0;
        let expected_alpha = 1.0 - (-sigma * 4.0).exp().powi(2);
        assert!(
            approx(gap_hit.alpha, expected_alpha, 1.0e-4),
            "gap alpha {} != two-segment expectation {}",
            gap_hit.alpha,
            expected_alpha
        );
    }

    /// Deterministic xorshift so the randomized grids/rays are reproducible.
    fn xorshift(state: &mut u32) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    /// A mixed grid for the skip tests: unit cells, a fully-empty brick
    /// column (x in [8, 16)), and elsewhere a random mix of empty cells,
    /// grade payloads and fallback-flagged payloads under a 3-colour ramp.
    fn mixed_brick_grid(seed: u32) -> VolumeReference {
        let dims = [24u32, 12, 10];
        let planes = |n: u32| (0..=n).map(|c| c as f32).collect::<Vec<_>>();
        let mut state = seed;
        let mut cells = Vec::with_capacity((dims[0] * dims[1] * dims[2]) as usize);
        for _k in 0..dims[2] {
            for _j in 0..dims[1] {
                for i in 0..dims[0] {
                    let r = xorshift(&mut state);
                    let payload = if (8..16).contains(&i) || r % 10 < 4 {
                        EMPTY_CELL_PAYLOAD
                    } else if r % 10 == 9 {
                        FALLBACK_CELL_FLAG
                    } else {
                        xorshift(&mut state) & 0xffff
                    };
                    cells.push(payload);
                }
            }
        }
        VolumeReference {
            x_planes: planes(dims[0]),
            y_planes: planes(dims[1]),
            z_planes: planes(dims[2]),
            dims,
            cells,
            stops: vec![
                Stop {
                    color: [1.0, 0.1, 0.1, 0.55],
                    t: 0.0,
                },
                Stop {
                    color: [0.1, 1.0, 0.1, 0.4],
                    t: 0.4,
                },
                Stop {
                    color: [0.1, 0.1, 1.0, 0.7],
                    t: 0.7,
                },
            ],
            fallback_color: [0.6, 0.6, 0.6, 0.5],
            reference_len: 1.0,
            opacity_cutoff: 0.999,
            max_steps: 4096,
            brick_size: 8,
            skip_uniform_bricks: false,
        }
    }

    #[test]
    fn empty_brick_skipping_is_output_preserving() {
        // The skip must be invisible: for every ray, traversal with
        // empty-brick skipping produces the same colour, alpha and hit/miss
        // as stepping every cell — including the Fresnel highlights at
        // empty↔filled brick faces, which the skip hands to the normal
        // per-cell step at the brick's exit cell.
        let no_skip = mixed_brick_grid(0x9e37_79b9);
        let mut skip = mixed_brick_grid(0x9e37_79b9);
        skip.skip_uniform_bricks = true;

        let mut state = 0x1234_5678u32;
        let unit = |s: &mut u32| xorshift(s) as f32 / u32::MAX as f32;
        let mut hits = 0u32;
        let mut skipped_steps = 0u64;
        let mut full_steps = 0u64;
        for ray in 0..500 {
            // Random interior target, origin pushed outside a random face.
            let target = vec3(
                unit(&mut state) * 24.0,
                unit(&mut state) * 12.0,
                unit(&mut state) * 10.0,
            );
            let mut origin = vec3(
                unit(&mut state) * 44.0 - 10.0,
                unit(&mut state) * 32.0 - 10.0,
                unit(&mut state) * 30.0 - 10.0,
            );
            match xorshift(&mut state) % 6 {
                0 => origin.x = -5.0,
                1 => origin.x = 29.0,
                2 => origin.y = -5.0,
                3 => origin.y = 17.0,
                4 => origin.z = -5.0,
                _ => origin.z = 15.0,
            }
            let dir = target - origin;
            if dir.length() < 1.0e-3 {
                continue;
            }

            let a = no_skip.raycast(origin, dir);
            let b = skip.raycast(origin, dir);
            match (a, b) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    hits += 1;
                    full_steps += u64::from(a.steps);
                    skipped_steps += u64::from(b.steps);
                    assert!(
                        approx(a.alpha, b.alpha, 1.0e-4),
                        "ray {ray}: alpha diverged with skipping: {} vs {}",
                        a.alpha,
                        b.alpha
                    );
                    for c in 0..3 {
                        assert!(
                            approx(a.rgb[c], b.rgb[c], 1.0e-4),
                            "ray {ray}: rgb[{c}] diverged with skipping: {} vs {}",
                            a.rgb[c],
                            b.rgb[c]
                        );
                    }
                }
                (a, b) => panic!("ray {ray}: hit/miss diverged with skipping: {a:?} vs {b:?}"),
            }
        }
        assert!(hits > 100, "test needs real coverage; only {hits} rays hit");
        assert!(
            skipped_steps < full_steps,
            "skipping should reduce total steps: {skipped_steps} vs {full_steps}"
        );
    }

    #[test]
    fn empty_brick_skipping_jumps_the_gap_in_constant_steps() {
        // 8 bricks along x; only the first and last are occupied. Cell-by-cell
        // traversal pays one step per empty cell in the 48-cell gap; the skip
        // must cross each empty brick in O(1) steps with identical output.
        let dims = [64u32, 8, 8];
        let planes = |n: u32| (0..=n).map(|c| c as f32).collect::<Vec<_>>();
        let mut cells = vec![EMPTY_CELL_PAYLOAD; (dims[0] * dims[1] * dims[2]) as usize];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    if !(8..56).contains(&i) {
                        cells[((k * dims[1] + j) * dims[0] + i) as usize] = 0xffff;
                    }
                }
            }
        }
        let grid = |skip: bool| VolumeReference {
            x_planes: planes(dims[0]),
            y_planes: planes(dims[1]),
            z_planes: planes(dims[2]),
            dims,
            cells: cells.clone(),
            stops: vec![
                Stop {
                    color: [1.0, 0.0, 0.0, 0.5],
                    t: 0.0,
                },
                Stop {
                    color: [1.0, 0.0, 0.0, 0.5],
                    t: 1.0,
                },
            ],
            fallback_color: [0.5, 0.5, 0.5, 1.0],
            reference_len: 1.0,
            opacity_cutoff: 0.9999,
            max_steps: 4096,
            brick_size: 8,
            skip_uniform_bricks: skip,
        };

        let origin = vec3(-5.0, 4.2, 4.3);
        let dir = vec3(1.0, 0.0, 0.0);
        let full = grid(false).raycast(origin, dir).expect("full hit");
        let skipped = grid(true).raycast(origin, dir).expect("skip hit");

        assert!(
            approx(full.alpha, skipped.alpha, 1.0e-4),
            "alpha diverged: {} vs {}",
            full.alpha,
            skipped.alpha
        );
        assert!(
            approx(full.rgb[0], skipped.rgb[0], 1.0e-4),
            "red diverged: {} vs {}",
            full.rgb[0],
            skipped.rgb[0]
        );
        // Cell-by-cell pays for the first brick, the whole 48-cell gap, and
        // the far cells up to early termination; the skip must cut the step
        // count by more than half by crossing each empty brick in O(1).
        assert!(
            full.steps >= 56,
            "cell-by-cell traversal should step every gap cell, got {}",
            full.steps
        );
        assert!(
            skipped.steps < full.steps / 2,
            "skip did not reduce steps: {} vs {}",
            skipped.steps,
            full.steps
        );
    }

    #[test]
    fn uniform_brick_skipping_is_output_preserving() {
        // Exercises the uniform-brick fast path specifically: a solid region
        // whose cells all resolve to one ramp colour (different grades, same
        // stop bin), a region of invisible cells mixing EMPTY payloads with
        // filled-but-below-first-stop grades (one appearance class, so the
        // skip may treat them interchangeably), and a mixed region that must
        // never be skipped. Skipping must reproduce cell-by-cell output
        // exactly — including surface rims at visible↔invisible faces and the
        // whole-span accumulation of jumped visible bricks.
        let dims = [32u32, 16, 16];
        let planes = |n: u32| (0..=n).map(|c| c as f32).collect::<Vec<_>>();
        let mut state = 0x0badc0deu32;
        let mut cells = Vec::with_capacity((dims[0] * dims[1] * dims[2]) as usize);
        for _k in 0..dims[2] {
            for _j in 0..dims[1] {
                for i in 0..dims[0] {
                    let r = xorshift(&mut state);
                    let payload = if i < 16 {
                        // Same stop bin (grades in [0.5, 0.6)): uniform colour.
                        (0.5 * 65535.0) as u32 + r % 6000
                    } else if i < 24 {
                        // Mixed: random grades across bins plus empties.
                        if r.is_multiple_of(4) {
                            EMPTY_CELL_PAYLOAD
                        } else {
                            xorshift(&mut state) & 0xffff
                        }
                    } else {
                        // Invisible: EMPTY and below-first-stop filled cells
                        // share one appearance class.
                        if r.is_multiple_of(2) {
                            EMPTY_CELL_PAYLOAD
                        } else {
                            r % 3000 // grade < 0.1 -> below the first stop
                        }
                    };
                    cells.push(payload);
                }
            }
        }
        let grid = |skip: bool| VolumeReference {
            x_planes: planes(dims[0]),
            y_planes: planes(dims[1]),
            z_planes: planes(dims[2]),
            dims,
            cells: cells.clone(),
            stops: vec![
                Stop {
                    color: [1.0, 0.2, 0.1, 0.5],
                    t: 0.1,
                },
                Stop {
                    color: [0.1, 0.9, 0.2, 0.35],
                    t: 0.45,
                },
                Stop {
                    color: [0.2, 0.2, 1.0, 0.6],
                    t: 0.62,
                },
            ],
            fallback_color: [0.6, 0.6, 0.6, 0.5],
            reference_len: 1.0,
            // Tight cutoff: past the cutoff the jump over-integrates by up to
            // a brick where per-cell stepping over-integrates by up to a cell.
            // Both snap to opaque, so the residual divergence is bounded by
            // (1 - cutoff) — kept far below the comparison tolerance here.
            opacity_cutoff: 0.99999,
            max_steps: 4096,
            brick_size: 8,
            skip_uniform_bricks: skip,
        };
        let no_skip = grid(false);
        let skip = grid(true);

        let mut state = 0xfeed_beefu32;
        let unit = |s: &mut u32| xorshift(s) as f32 / u32::MAX as f32;
        let mut hits = 0u32;
        let mut skipped_steps = 0u64;
        let mut full_steps = 0u64;
        for ray in 0..400 {
            let target = vec3(
                unit(&mut state) * 32.0,
                unit(&mut state) * 16.0,
                unit(&mut state) * 16.0,
            );
            let mut origin = vec3(
                unit(&mut state) * 52.0 - 10.0,
                unit(&mut state) * 36.0 - 10.0,
                unit(&mut state) * 36.0 - 10.0,
            );
            match xorshift(&mut state) % 6 {
                0 => origin.x = -5.0,
                1 => origin.x = 37.0,
                2 => origin.y = -5.0,
                3 => origin.y = 21.0,
                4 => origin.z = -5.0,
                _ => origin.z = 21.0,
            }
            let dir = target - origin;
            if dir.length() < 1.0e-3 {
                continue;
            }

            let a = no_skip.raycast(origin, dir);
            let b = skip.raycast(origin, dir);
            match (a, b) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    hits += 1;
                    full_steps += u64::from(a.steps);
                    skipped_steps += u64::from(b.steps);
                    assert!(
                        approx(a.alpha, b.alpha, 1.0e-4),
                        "ray {ray}: alpha diverged with uniform skipping: {} vs {}",
                        a.alpha,
                        b.alpha
                    );
                    for c in 0..3 {
                        assert!(
                            approx(a.rgb[c], b.rgb[c], 1.0e-4),
                            "ray {ray}: rgb[{c}] diverged with uniform skipping: {} vs {}",
                            a.rgb[c],
                            b.rgb[c]
                        );
                    }
                }
                (a, b) => {
                    panic!("ray {ray}: hit/miss diverged with uniform skipping: {a:?} vs {b:?}")
                }
            }
        }
        assert!(hits > 100, "test needs real coverage; only {hits} rays hit");
        assert!(
            skipped_steps * 2 < full_steps,
            "uniform skipping should cut steps by more than half: {skipped_steps} vs {full_steps}"
        );
    }

    #[test]
    fn head_on_boundary_adds_no_highlight_energy() {
        // A ray crossing a colour boundary perpendicular has Fresnel = 0, so
        // the boundary highlight must contribute nothing: the result equals
        // pure two-segment accumulation. Cell 0 red, cell 1 blue.
        let mut grid = uniform_grid([2, 1, 1], 8.0, 1.0, 4.0);
        // Distinct grades so the two cells resolve to different ramp colours.
        grid.stops = vec![
            Stop {
                color: [1.0, 0.0, 0.0, 1.0],
                t: 0.0,
            },
            Stop {
                color: [0.0, 0.0, 1.0, 1.0],
                t: 0.75,
            },
        ];
        grid.cells = vec![0x0000, 0xffff]; // grade 0 -> red, grade 1 -> blue

        let hit = grid
            .raycast(vec3(-5.0, 4.0, 4.0), vec3(1.0, 0.0, 0.0))
            .expect("boundary ray hits");

        // First cell (red, opaque, len 4): alpha1 = 1 - exp(-sigma*4).
        let sigma = -(0.001_f32.ln()) / 4.0;
        let alpha1 = 1.0 - (-sigma * 4.0).exp();
        // Red channel only comes from cell 0; no green/blue leak from a rim.
        let expected_r = alpha1 * 1.0 * VOLUME_AMBIENT_INTENSITY;
        assert!(
            approx(hit.rgb[0], expected_r, 2.0e-3),
            "head-on boundary leaked highlight energy into red: {} vs {}",
            hit.rgb[0],
            expected_r
        );
    }
}
