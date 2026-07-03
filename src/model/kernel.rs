//! Shared robust 2D geometry kernel.
//!
//! Every intersection/sidedness decision in the app should come through here
//! rather than reimplementing `denom.abs() < 1e-10` locally, because those
//! ad-hoc gates are scale-dependent: the cross product of two unnormalized
//! direction vectors grows with segment length, so a fixed epsilon means a
//! different *angle* for a 1 m edge than for a 1 km haul road, and at
//! UTM-magnitude coordinates (~1e6) many such epsilons fall below f64 ulp and
//! silently become exact-equality tests on inexact data.
//!
//! Policy:
//! - **Signs are exact.** Which side of a line a point lies on, and whether
//!   two segments cross, are decided with adaptive-precision predicates
//!   ([`robust::orient2d`]), never with an epsilon.
//! - **Tolerances are metric.** "Close enough to coincide" is a distance in
//!   world metres ([`XY_TOL`], [`Z_TOL`]), applied to points — never to
//!   parameters (`t`, `u`) or to raw cross products, whose scale varies with
//!   the input.
//! - **Kernel classifies, callers filter.** [`segment_segment`] reports the
//!   full configuration (crossing / endpoint touch / collinear overlap /
//!   disjoint); a call site that only wants strict interior crossings matches
//!   on [`SegSeg::Crossing`] instead of re-deriving epsilon math.

use glam::DVec2;

/// World-space XY coincidence tolerance in metres (0.1 mm).
///
/// This is the app-wide answer to "are these the same point on the ground?".
/// Survey data is good to ~1 mm at best; anything closer than 0.1 mm apart is
/// noise from prior floating-point construction, not intent.
pub(crate) const XY_TOL: f64 = 1e-4;

/// World-space Z coincidence tolerance in metres (5 cm).
///
/// Deliberately much looser than [`XY_TOL`]: elevations come from
/// triangulated surfaces and design strings whose vertical agreement is far
/// coarser than horizontal survey precision.
pub(crate) const Z_TOL: f64 = 0.05;

/// Exact sign of the cross product `(b - a) × (c - a)`.
///
/// Positive when `c` lies to the left of the directed line `a -> b`, negative
/// to the right, exactly zero only when the three points are exactly
/// collinear. The magnitude is approximate (twice the triangle area); only
/// the sign is guaranteed.
pub(crate) fn orient2d(a: DVec2, b: DVec2, c: DVec2) -> f64 {
    robust::orient2d(
        robust::Coord { x: a.x, y: a.y },
        robust::Coord { x: b.x, y: b.y },
        robust::Coord { x: c.x, y: c.y },
    )
}

/// Intersect two infinite lines given as point + direction.
///
/// Returns `None` when the directions are parallel (exact sign test) or so
/// nearly parallel that the intersection point is numerically meaningless
/// (relative angular guard, scale-invariant in both segment length and
/// coordinate magnitude).
pub(crate) fn line_line(a: DVec2, da: DVec2, b: DVec2, db: DVec2) -> Option<DVec2> {
    // Exact sign of da × db: orient2d(0, da, db) computes it adaptively.
    let cross = orient2d(DVec2::ZERO, da, db);
    // Conditioning guard: |da × db| = |da||db| sin(angle). Requiring
    // sin(angle) > 1e-12 rejects intersections whose position error would be
    // amplified ~1e12×, independent of how long the direction vectors are.
    if cross == 0.0 || cross.abs() <= 1e-12 * da.length() * db.length() {
        return None;
    }
    let t = (b - a).perp_dot(db) / cross;
    Some(a + da * t)
}

/// Intersect the infinite line `point + t * dir` with the segment `a -> b`.
///
/// Returns the intersection point and the segment parameter. `slack` is a
/// metric distance: the hit may lie up to `slack` metres beyond either
/// segment endpoint and still count (the returned parameter is clamped to
/// `[0, 1]`). Pass `0.0` for a hard segment test, [`XY_TOL`] to accept
/// touch points that drifted off the end by floating-point noise.
pub(crate) fn line_segment(
    point: DVec2,
    dir: DVec2,
    a: DVec2,
    b: DVec2,
    slack: f64,
) -> Option<(DVec2, f64)> {
    let seg = b - a;
    let cross = orient2d(DVec2::ZERO, dir, seg);
    if cross == 0.0 || cross.abs() <= 1e-12 * dir.length() * seg.length() {
        return None;
    }
    let delta = a - point;
    let line_t = delta.perp_dot(seg) / cross;
    let seg_t = delta.perp_dot(dir) / cross;
    // Convert the metric slack into parameter space on *this* segment.
    let seg_len = seg.length();
    let param_slack = if seg_len > 0.0 { slack / seg_len } else { 0.0 };
    if (-param_slack..=1.0 + param_slack).contains(&seg_t) {
        Some((point + dir * line_t, seg_t.clamp(0.0, 1.0)))
    } else {
        None
    }
}

/// Full configuration of two XY segments, from [`segment_segment`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SegSeg {
    /// The segments do not come within [`XY_TOL`] of each other.
    Disjoint,
    /// Interiors cross at a single point strictly further than [`XY_TOL`]
    /// from every endpoint. `t` parametrizes `a -> b`, `u` parametrizes
    /// `c -> d`.
    Crossing { point: DVec2, t: f64, u: f64 },
    /// The segments meet within [`XY_TOL`] of at least one endpoint —
    /// a shared vertex or a T-junction.
    Touching { point: DVec2, t: f64, u: f64 },
    /// The segments run along the same line (within [`XY_TOL`] laterally)
    /// and their projections overlap by more than [`XY_TOL`]. `t0..t1` is
    /// the overlapping parameter range on `a -> b`.
    CollinearOverlap { t0: f64, t1: f64 },
}

/// Classify how segments `a -> b` and `c -> d` relate in the XY plane.
///
/// Crossing vs not is decided by exact orientation signs; crossing vs
/// touching, and near-miss touches, by metric distance ([`XY_TOL`]).
pub(crate) fn segment_segment(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> SegSeg {
    // Collinear-overlap first: both far endpoints within XY_TOL laterally of
    // the other segment's line, with overlapping extents. Exact sign tests
    // can't see this case on digitized data (nothing is *exactly* collinear).
    if let Some(overlap) = collinear_overlap(a, b, c, d) {
        return overlap;
    }

    let o1 = orient2d(c, d, a);
    let o2 = orient2d(c, d, b);
    let o3 = orient2d(a, b, c);
    let o4 = orient2d(a, b, d);

    // Exact proper-crossing test: each segment's endpoints straddle the
    // other's supporting line (allowing exactly-on-line as straddling so a
    // vertex sitting exactly on the other segment still yields a point).
    let straddles = (o1 >= 0.0) != (o2 >= 0.0) || (o1 > 0.0) != (o2 > 0.0);
    let straddled = (o3 >= 0.0) != (o4 >= 0.0) || (o3 > 0.0) != (o4 > 0.0);
    if straddles && straddled {
        let r = b - a;
        let s = d - c;
        let denom = r.perp_dot(s);
        // A true crossing guarantees denom != 0; the division is
        // well-conditioned because the straddle test already bounded the
        // configuration away from parallel-and-disjoint.
        let t = ((c - a).perp_dot(s) / denom).clamp(0.0, 1.0);
        let u = ((c - a).perp_dot(r) / denom).clamp(0.0, 1.0);
        let point = a + r * t;
        let near_endpoint = point.distance(a) <= XY_TOL
            || point.distance(b) <= XY_TOL
            || point.distance(c) <= XY_TOL
            || point.distance(d) <= XY_TOL;
        return if near_endpoint {
            SegSeg::Touching { point, t, u }
        } else {
            SegSeg::Crossing { point, t, u }
        };
    }

    // No exact crossing: the segments may still pass within XY_TOL of each
    // other at an endpoint (a touch that drifted apart by float noise).
    let candidates = [
        (a, c, d, 0.0, None),
        (b, c, d, 1.0, None),
        (c, a, b, 0.0, Some(())),
        (d, a, b, 1.0, Some(())),
    ];
    let mut best: Option<(f64, DVec2, f64, f64)> = None;
    for (p, s0, s1, fixed_param, swapped) in candidates {
        let (proj, param) = project_onto_segment(p, s0, s1);
        let dist = p.distance(proj);
        if dist <= XY_TOL && best.is_none_or(|(bd, ..)| dist < bd) {
            let (t, u) = if swapped.is_some() {
                (param, fixed_param)
            } else {
                (fixed_param, param)
            };
            best = Some((dist, proj, t, u));
        }
    }
    if let Some((_, point, t, u)) = best {
        return SegSeg::Touching { point, t, u };
    }
    SegSeg::Disjoint
}

/// Signed perpendicular distance in metres from `point` to the directed line
/// `a -> b`. Positive on the left of the line; sign is exact (via
/// [`orient2d`]), magnitude is approximate. Returns `0.0` for a degenerate
/// (zero-length) edge.
pub(crate) fn signed_distance_to_line(point: DVec2, a: DVec2, b: DVec2) -> f64 {
    let len = a.distance(b);
    if len == 0.0 {
        return 0.0;
    }
    orient2d(a, b, point) / len
}

/// Closest point on segment `a -> b` to `p`, with its clamped parameter.
pub(crate) fn project_onto_segment(p: DVec2, a: DVec2, b: DVec2) -> (DVec2, f64) {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq == 0.0 {
        return (a, 0.0);
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    (a + ab * t, t)
}

fn collinear_overlap(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> Option<SegSeg> {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq == 0.0 {
        return None;
    }
    let len = len_sq.sqrt();
    // Lateral deviation of c and d from the line through a -> b, in metres.
    let dev_c = orient2d(a, b, c).abs() / len;
    let dev_d = orient2d(a, b, d).abs() / len;
    if dev_c > XY_TOL || dev_d > XY_TOL {
        return None;
    }
    let t_c = (c - a).dot(ab) / len_sq;
    let t_d = (d - a).dot(ab) / len_sq;
    let (lo, hi) = (t_c.min(t_d), t_c.max(t_d));
    let t0 = lo.max(0.0);
    let t1 = hi.min(1.0);
    // Overlap must be a real shared extent (> XY_TOL metres), not just a
    // shared endpoint.
    if (t1 - t0) * len > XY_TOL {
        Some(SegSeg::CollinearOverlap { t0, t1 })
    } else {
        None
    }
}

/// True when `point` lies within [`XY_TOL`] of the *interior* of segment
/// `a -> b` (endpoints excluded by more than [`XY_TOL`]) and within
/// [`Z_TOL`] of the segment's interpolated elevation there.
pub(crate) fn point_on_segment_interior_3d(
    point: glam::DVec3,
    a: glam::DVec3,
    b: glam::DVec3,
) -> bool {
    let (proj, t) = project_onto_segment(point.truncate(), a.truncate(), b.truncate());
    if point.truncate().distance(proj) > XY_TOL {
        return false;
    }
    // Endpoint exclusion, metric: the projection must sit further than
    // XY_TOL along the segment from both ends.
    let len = a.truncate().distance(b.truncate());
    if len <= 2.0 * XY_TOL || t * len <= XY_TOL || (1.0 - t) * len <= XY_TOL {
        return false;
    }
    let z = a.z + (b.z - a.z) * t;
    (z - point.z).abs() <= Z_TOL
}

/// True when two points coincide within [`XY_TOL`] horizontally and
/// [`Z_TOL`] vertically.
pub(crate) fn points_coincident_3d(a: glam::DVec3, b: glam::DVec3) -> bool {
    a.truncate().distance_squared(b.truncate()) <= XY_TOL * XY_TOL && (a.z - b.z).abs() <= Z_TOL
}

/// Where a point sits relative to a closed ring in the XY plane.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PolyContainment {
    Inside,
    Outside,
    /// Within [`XY_TOL`] of an edge (or vertex) of the ring.
    OnBoundary,
}

/// Robust point-in-polygon: exact-sign crossing count ([`orient2d`] decides
/// which side of an edge the ray passes), with points within [`XY_TOL`] of
/// the ring classified [`PolyContainment::OnBoundary`] first so callers make
/// the borderline call explicitly instead of inheriting float noise.
///
/// The ring is closed implicitly (last vertex connects to the first);
/// zero-length edges are skipped. Fewer than 3 distinct vertices → `Outside`.
pub(crate) fn point_in_polygon(
    point: DVec2,
    ring: impl IntoIterator<Item = DVec2>,
) -> PolyContainment {
    let mut iter = ring.into_iter();
    let Some(first) = iter.next() else {
        return PolyContainment::Outside;
    };

    let mut inside = false;
    let mut edges = 0usize;
    let mut prev = first;
    let mut process_edge = |a: DVec2, b: DVec2| -> Option<PolyContainment> {
        if a == b {
            return None;
        }
        let (proj, _) = project_onto_segment(point, a, b);
        if point.distance(proj) <= XY_TOL {
            return Some(PolyContainment::OnBoundary);
        }
        // Ray cast toward +X. The half-open `y` comparison counts an edge
        // exactly once when the ray passes through a shared vertex, and
        // orient2d decides the crossing side exactly.
        let upward = a.y <= point.y && point.y < b.y;
        let downward = b.y <= point.y && point.y < a.y;
        if (upward && orient2d(a, b, point) > 0.0) || (downward && orient2d(a, b, point) < 0.0) {
            inside = !inside;
        }
        edges += 1;
        None
    };

    for cur in iter {
        if let Some(on_boundary) = process_edge(prev, cur) {
            return on_boundary;
        }
        prev = cur;
    }
    if let Some(on_boundary) = process_edge(prev, first) {
        return on_boundary;
    }

    if edges < 3 {
        PolyContainment::Outside
    } else if inside {
        PolyContainment::Inside
    } else {
        PolyContainment::Outside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // All positions use UTM-magnitude coordinates on purpose: the epsilons
    // these tests replace only failed at real mine coordinates.
    const E: f64 = 476_000.0;
    const N: f64 = 7_654_000.0;

    fn p(x: f64, y: f64) -> DVec2 {
        DVec2::new(E + x, N + y)
    }

    #[test]
    fn point_in_polygon_classifies_inside_outside_and_boundary_at_utm_magnitude() {
        let square = [p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0), p(0.0, 10.0)];
        let ring = || square.iter().copied();
        assert_eq!(
            point_in_polygon(p(5.0, 5.0), ring()),
            PolyContainment::Inside
        );
        assert_eq!(
            point_in_polygon(p(15.0, 5.0), ring()),
            PolyContainment::Outside
        );
        // 0.05 mm outside the right edge: metrically on the boundary.
        assert_eq!(
            point_in_polygon(p(10.0 + 0.00005, 5.0), ring()),
            PolyContainment::OnBoundary
        );
        assert_eq!(point_in_polygon(p(10.0, 10.0), ring()), {
            PolyContainment::OnBoundary
        });
    }

    #[test]
    fn point_in_polygon_counts_a_vertex_on_the_ray_exactly_once() {
        // A diamond whose left/right vertices sit at the probe's Y: the +X ray
        // from an inside probe passes exactly through the vertex at (10, 5).
        // Naive implementations double-count it and flip to Outside.
        let diamond = [p(5.0, 0.0), p(10.0, 5.0), p(5.0, 10.0), p(0.0, 5.0)];
        assert_eq!(
            point_in_polygon(p(5.0, 5.0), diamond.iter().copied()),
            PolyContainment::Inside
        );
        // Outside probe at the same Y stays outside.
        assert_eq!(
            point_in_polygon(p(-5.0, 5.0), diamond.iter().copied()),
            PolyContainment::Outside
        );
    }

    #[test]
    fn crossing_at_utm_magnitude() {
        let result = segment_segment(p(0.0, 0.0), p(10.0, 0.0), p(5.0, -5.0), p(5.0, 5.0));
        let SegSeg::Crossing { point, t, u } = result else {
            panic!("expected Crossing, got {result:?}");
        };
        assert!((point - p(5.0, 0.0)).length() < 1e-6);
        assert!((t - 0.5).abs() < 1e-9 && (u - 0.5).abs() < 1e-9);
    }

    #[test]
    fn touch_within_metric_tolerance_regardless_of_segment_length() {
        // Endpoint 0.05 mm off a 10 km segment: parameter-space epsilons of
        // 1e-8 rejected this; metric tolerance accepts it.
        let result = segment_segment(
            p(0.0, 0.0),
            p(10_000.0, 0.0),
            p(5_000.0, 0.00005),
            p(5_000.0, 5.0),
        );
        assert!(
            matches!(result, SegSeg::Touching { .. }),
            "expected Touching, got {result:?}"
        );
    }

    #[test]
    fn crossing_near_endpoint_classifies_as_touch_not_crossing() {
        let result = segment_segment(p(0.0, 0.0), p(10.0, 0.0), p(0.00002, -5.0), p(0.00002, 5.0));
        assert!(
            matches!(result, SegSeg::Touching { .. }),
            "expected Touching, got {result:?}"
        );
    }

    #[test]
    fn collinear_overlap_detected_on_digitized_data() {
        // Independently digitized breaklines along the same wall: parallel to
        // within survey noise, overlapping for 4 metres.
        let result = segment_segment(p(0.0, 0.0), p(10.0, 0.0), p(6.0, 0.00003), p(14.0, 0.00006));
        let SegSeg::CollinearOverlap { t0, t1 } = result else {
            panic!("expected CollinearOverlap, got {result:?}");
        };
        assert!((t0 - 0.6).abs() < 1e-3 && (t1 - 1.0).abs() < 1e-3);
    }

    #[test]
    fn shared_endpoint_is_touch_not_overlap() {
        let result = segment_segment(p(0.0, 0.0), p(10.0, 0.0), p(10.0, 0.0), p(20.0, 0.0));
        assert!(
            matches!(result, SegSeg::Touching { .. }),
            "expected Touching, got {result:?}"
        );
    }

    #[test]
    fn clearly_disjoint() {
        let result = segment_segment(p(0.0, 0.0), p(10.0, 0.0), p(0.0, 1.0), p(10.0, 1.0));
        assert_eq!(result, SegSeg::Disjoint);
    }

    #[test]
    fn line_line_rejects_parallel_exactly_and_near() {
        assert!(line_line(p(0.0, 0.0), DVec2::X, p(0.0, 1.0), DVec2::X).is_none());
        // Nearly parallel long directions: same angle must be rejected
        // regardless of direction-vector length.
        let almost = DVec2::new(1000.0, 1e-10);
        assert!(line_line(p(0.0, 0.0), DVec2::X * 1000.0, p(0.0, 1.0), almost).is_none());
    }

    #[test]
    fn line_segment_metric_slack() {
        // Hit drifted 0.05 mm past the segment end.
        let hit = line_segment(p(0.0, 0.0), DVec2::X, p(5.0, 0.00005), p(5.0, 5.0), XY_TOL);
        assert!(hit.is_some());
        let (_, t) = hit.unwrap();
        assert_eq!(t, 0.0);
        // Hard test rejects the same input.
        assert!(line_segment(p(0.0, 0.0), DVec2::X, p(5.0, 0.00005), p(5.0, 5.0), 0.0).is_none());
    }
}
