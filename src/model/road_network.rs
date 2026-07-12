//! Road network resolution — the single source of truth for derived road
//! geometry.
//!
//! `resolve` turns stored road centerlines (user intent) plus an optional
//! in-progress "ghost" road into final side-line geometry. Nothing here
//! mutates the document: junction pads, flat approaches and camber blending
//! are derived per resolution. Every point shared between two roads (a
//! junction corner, a seam miter) is computed exactly once by the node that
//! owns it, so side lines meet exactly by construction.

use std::collections::HashMap;

use glam::{DVec2, DVec3};

use super::{Document, Object, ObjectId, RoadShape, geometry::ROAD_INTERSECTION_FLAT_CLEARANCE_M};

/// XY tolerance for two points to be considered the same network node.
const NODE_XY_EPS: f64 = crate::model::kernel::XY_TOL;
/// Max elevation difference for coincident-in-plan points to join one node.
/// Larger separations are crossings (overpass) and do not connect.
const NODE_Z_TOL: f64 = crate::model::kernel::Z_TOL;
/// Two ports within this many degrees of a straight continuation form a seam
/// (settings blend across) rather than a junction pad (camber runs off to 0).
const SEAM_STRAIGHT_TOL_DEGREES: f64 = 15.0;
/// Length over which width and camber blend across a seam between roads with
/// different settings.
const SEAM_BLEND_M: f64 = 10.0;
/// Mitre guard for corner points and vertex offsets, in multiples of the
/// half-width; beyond this corners are bevelled.
const MITER_LIMIT: f64 = 4.0;
/// The junction pad solve assumes every edge runs straight (at its port
/// heading) through its approach zone. An edge whose centerline deviates
/// laterally by more than this fraction of the half-width inside the zone is
/// compromised: pad corners would land off the road and the side-line trim
/// would drop the wrong samples.
const JUNCTION_APPROACH_DEVIATION_FRAC: f64 = 0.25;
/// Minimum interior angle at any centerline vertex or between any two
/// branches meeting at a node.
pub(crate) const MIN_ROAD_TURN_ANGLE_DEGREES: f64 = 30.0;

/// Identifies the source of an edge: a committed document object or the
/// in-progress preview road.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum RoadKey {
    Object(ObjectId),
    Ghost,
}

/// The road currently being drawn (pending stroke + cursor), resolved exactly
/// like a committed road so the preview IS the committed result.
pub(crate) struct GhostRoad {
    pub(crate) centerline: Vec<DVec3>,
    pub(crate) width: f64,
    pub(crate) camber_degrees: f64,
    pub(crate) shape: RoadShape,
}

/// Final resolved geometry for one span of road between two network nodes.
/// `left`/`right` are ready to draw: trimmed to shared junction corners, with
/// camber blending and junction flattening already applied.
pub(crate) struct EdgeGeom {
    pub(crate) road: RoadKey,
    pub(crate) center: Vec<DVec3>,
    pub(crate) left: Vec<DVec3>,
    pub(crate) right: Vec<DVec3>,
    /// Draw a square cap across the road at the start/end (dead ends only).
    pub(crate) start_cap: bool,
    pub(crate) end_cap: bool,
}

#[derive(Default)]
pub(crate) struct ResolvedNetwork {
    pub(crate) edges: Vec<EdgeGeom>,
    /// Committed roads whose resolved geometry depends on the ghost: the
    /// transitive closure of node-sharing from the ghost. A ghost junction
    /// reshapes an edge's centre profile, and that edge in turn feeds the pad
    /// solve at the road's other nodes, so influence propagates road-to-road.
    pub(crate) ghost_affected: Vec<ObjectId>,
}

impl ResolvedNetwork {
    pub(crate) fn edges_for(&self, key: RoadKey) -> impl Iterator<Item = &EdgeGeom> {
        self.edges.iter().filter(move |edge| edge.road == key)
    }
}

// ---------------------------------------------------------------------------
// Internal working types
// ---------------------------------------------------------------------------

struct SourceRoad {
    key: RoadKey,
    centerline: Vec<DVec3>,
    width: f64,
    /// Cross slope of each side as rise per unit half-width (signed; negative
    /// drops the edge below the centerline).
    slope_left: f64,
    slope_right: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NodeKind {
    DeadEnd,
    Seam,
    Junction,
    Attachment,
}

struct Node {
    pos: DVec3,
    z_sum: f64,
    z_count: usize,
    /// Segments of attached polylines near this node (side rays terminate on
    /// these).
    attachment_segments: Vec<[DVec3; 2]>,
    /// Roads this node captures beyond the default `NODE_XY_EPS`: when a
    /// crossing snaps onto a centerline vertex, the non-vertex road passes up
    /// to this far from the node in plan and must still be cut here.
    capture_roads: Vec<(usize, f64)>,
    ports: Vec<Port>,
    kind: NodeKind,
}

impl Node {
    fn new(pos: DVec3) -> Self {
        Self {
            pos,
            z_sum: pos.z,
            z_count: 1,
            attachment_segments: Vec::new(),
            capture_roads: Vec::new(),
            ports: Vec::new(),
            kind: NodeKind::DeadEnd,
        }
    }
}

struct Port {
    edge: usize,
    at_start: bool,
    heading: DVec2,
    /// Half-width the edge presents at this node (after seam averaging).
    hw: f64,
    /// Flat/blend clearance along the edge from this node.
    clearance: f64,
    /// Cross slope of the side that lies counter-clockwise / clockwise of the
    /// outward heading, at the node (after seam averaging).
    slope_ccw: f64,
    slope_cw: f64,
}

/// Per-end blending parameters applied while sampling an edge cross-section.
#[derive(Clone, Copy)]
struct EndBlend {
    zone: f64,
    target_w: f64,
    target_slope_left: f64,
    target_slope_right: f64,
    /// Flatten the centerline to `flatten_z` over `zone` (junctions and
    /// attachments).
    flatten_z: Option<f64>,
}

struct WorkEdge {
    road: usize,
    start_node: usize,
    end_node: usize,
    center: Vec<DVec3>,
    left: Vec<DVec3>,
    right: Vec<DVec3>,
    start_blend: EndBlend,
    end_blend: EndBlend,
    compromised: Option<(f64, f64)>,
    /// The centerline bends away from a straight junction approach inside the
    /// approach zone: `(required straight run, lateral deviation)`.
    bend_compromised: Option<(f64, f64)>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn resolve(document: &Document, ghost: Option<&GhostRoad>) -> ResolvedNetwork {
    resolve_prepared(prepare(document, ghost))
}

/// A built network topology (stages 0–4), reusable across validation and
/// resolution so per-cursor-move callers pay for `build_topology` once.
pub(crate) struct PreparedNetwork {
    topology: Option<Topology>,
}

pub(crate) fn prepare(document: &Document, ghost: Option<&GhostRoad>) -> PreparedNetwork {
    PreparedNetwork {
        topology: build_topology(document, ghost),
    }
}

impl PreparedNetwork {
    /// Keys of edges that violate clearance/approach rules in this topology.
    /// The ghost-free set grandfathers legacy violations during validation.
    pub(crate) fn compromised_keys(&self) -> std::collections::HashSet<RoadKey> {
        let Some(topology) = &self.topology else {
            return Default::default();
        };
        topology
            .edges
            .iter()
            .filter(|edge| edge.compromised.is_some() || edge.bend_compromised.is_some())
            .map(|edge| topology.sources[edge.road].key)
            .collect()
    }
}

pub(crate) fn resolve_prepared(prepared: PreparedNetwork) -> ResolvedNetwork {
    let Some(Topology {
        sources,
        nodes,
        mut edges,
    }) = prepared.topology
    else {
        return ResolvedNetwork::default();
    };
    sample_cross_sections(&sources, &mut edges);
    solve_junctions(&mut edges, &nodes);
    for edge in &mut edges {
        remove_side_line_folds(&mut edge.left);
        remove_side_line_folds(&mut edge.right);
    }
    let ghost_affected = ghost_affected_roads(&sources, &nodes, &edges);

    ResolvedNetwork {
        ghost_affected,
        edges: edges
            .into_iter()
            .enumerate()
            .map(|(index, edge)| {
                let start_cap = nodes[edge.start_node].kind == NodeKind::DeadEnd
                    && port_of(&nodes[edge.start_node], index, true).is_some();
                let end_cap = nodes[edge.end_node].kind == NodeKind::DeadEnd
                    && port_of(&nodes[edge.end_node], index, false).is_some();
                EdgeGeom {
                    road: sources[edge.road].key,
                    center: edge.center,
                    left: edge.left,
                    right: edge.right,
                    start_cap,
                    end_cap,
                }
            })
            .collect(),
    }
}

/// Committed roads reachable from the ghost through shared nodes (fixpoint
/// over the node/port graph). Sorted by id so callers can compare sets cheaply.
fn ghost_affected_roads(
    sources: &[SourceRoad],
    nodes: &[Node],
    edges: &[WorkEdge],
) -> Vec<ObjectId> {
    if !sources.iter().any(|source| source.key == RoadKey::Ghost) {
        return Vec::new();
    }
    let mut affected: Vec<bool> = sources
        .iter()
        .map(|source| source.key == RoadKey::Ghost)
        .collect();
    loop {
        let mut changed = false;
        for node in nodes {
            if !node
                .ports
                .iter()
                .any(|port| affected[edges[port.edge].road])
            {
                continue;
            }
            for port in &node.ports {
                let road = edges[port.edge].road;
                if !affected[road] {
                    affected[road] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut out: Vec<ObjectId> = sources
        .iter()
        .zip(&affected)
        .filter(|&(_, &is_affected)| is_affected)
        .filter_map(|(source, _)| match source.key {
            RoadKey::Object(id) => Some(id),
            RoadKey::Ghost => None,
        })
        .collect();
    out.sort_unstable_by_key(|id| id.0);
    out
}

struct Topology {
    sources: Vec<SourceRoad>,
    nodes: Vec<Node>,
    edges: Vec<WorkEdge>,
}

/// Stages 0–4: everything up to (and including) center profiles — enough to
/// know the network's nodes, ports and clearance conflicts, without paying
/// for cross-section sampling and junction solving.
fn build_topology(document: &Document, ghost: Option<&GhostRoad>) -> Option<Topology> {
    let sources = collect_sources(document, ghost);
    if sources.is_empty() {
        return None;
    }
    let mut nodes = build_nodes(&sources, document);
    let mut edges = split_roads_at_nodes(&sources, &mut nodes);
    classify_nodes(&mut nodes);
    build_center_profiles(&sources, &mut edges, &nodes);
    Some(Topology {
        sources,
        nodes,
        edges,
    })
}

fn port_of(node: &Node, edge: usize, at_start: bool) -> Option<&Port> {
    node.ports
        .iter()
        .find(|port| port.edge == edge && port.at_start == at_start)
}

// ---------------------------------------------------------------------------
// Placement validation
// ---------------------------------------------------------------------------

/// A road-placement rule the ghost stroke violates. Produced by
/// [`validate_ghost`]; `Display` gives the user-facing message.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RoadRuleViolation {
    SegmentTooSteep {
        maximum_degrees: f64,
        actual_degrees: f64,
    },
    TurnTooSharp {
        minimum_degrees: f64,
        actual_degrees: f64,
    },
    /// A network edge the stroke creates cannot fit a required flat zone
    /// (junction approach or grade-break pocket).
    ClearanceTooTight {
        required: f64,
        actual: f64,
    },
    /// A network edge the stroke creates bends inside a junction's
    /// straight-approach zone (e.g. crossing another road just before its
    /// corner), which the junction pad geometry cannot represent.
    TurnTooCloseToJunction {
        required_straight: f64,
        deviation: f64,
    },
    DegenerateSegment,
}

impl std::fmt::Display for RoadRuleViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoadRuleViolation::SegmentTooSteep {
                maximum_degrees,
                actual_degrees,
            } => write!(
                f,
                "Road segment angle is too steep: {actual_degrees:.1}° exceeds the {maximum_degrees:.1}° maximum"
            ),
            RoadRuleViolation::TurnTooSharp {
                minimum_degrees,
                actual_degrees,
            } => write!(
                f,
                "Road turn is too sharp: {actual_degrees:.1}° is below the {minimum_degrees:.0}° minimum"
            ),
            RoadRuleViolation::ClearanceTooTight { required, actual } => write!(
                f,
                "Not enough room for {required:.1} m of flat approach: {actual:.2} m available"
            ),
            RoadRuleViolation::TurnTooCloseToJunction {
                required_straight,
                deviation,
            } => write!(
                f,
                "Road turns too close to a junction: it deviates {deviation:.1} m from a straight approach within the {required_straight:.1} m junction zone"
            ),
            RoadRuleViolation::DegenerateSegment => f.write_str("Road segment is too short"),
        }
    }
}

/// Check every placement rule against the ghost stroke: per-segment grade,
/// turn angles (both at its own vertices and between branches at every node
/// it joins — including mid-segment crossings), and flat-zone clearances.
/// Edges that were already compromised without the ghost (legacy documents)
/// are grandfathered; only newly created conflicts are refused.
pub(crate) fn validate_ghost(
    document: &Document,
    ghost: &GhostRoad,
    max_grade_degrees: f64,
) -> Result<(), RoadRuleViolation> {
    // Fast-fail on the stroke-local rules before paying for topology builds.
    validate_centerline_grades(&ghost.centerline, max_grade_degrees)?;
    validate_centerline_turns(&ghost.centerline)?;
    let preexisting = prepare(document, None).compromised_keys();
    validate_ghost_prepared(
        &prepare(document, Some(ghost)),
        ghost,
        max_grade_degrees,
        &preexisting,
    )
}

/// [`validate_ghost`] against an already-built ghost-inclusive topology and a
/// (cacheable) ghost-free `preexisting` compromised set.
pub(crate) fn validate_ghost_prepared(
    prepared: &PreparedNetwork,
    ghost: &GhostRoad,
    max_grade_degrees: f64,
    preexisting: &std::collections::HashSet<RoadKey>,
) -> Result<(), RoadRuleViolation> {
    validate_centerline_grades(&ghost.centerline, max_grade_degrees)?;
    validate_centerline_turns(&ghost.centerline)?;

    let Some(topology) = &prepared.topology else {
        return Ok(());
    };
    let is_ghost = |edge: usize| topology.sources[topology.edges[edge].road].key == RoadKey::Ghost;

    // Interior angle between two branches is the angle between their outward
    // headings; only pairs involving the ghost are checked so legacy
    // geometry is never retroactively refused.
    for node in &topology.nodes {
        for (i, a) in node.ports.iter().enumerate() {
            for b in &node.ports[i + 1..] {
                if !is_ghost(a.edge) && !is_ghost(b.edge) {
                    continue;
                }
                let angle_degrees = a
                    .heading
                    .dot(b.heading)
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees();
                if angle_degrees + 1e-6 < MIN_ROAD_TURN_ANGLE_DEGREES {
                    return Err(RoadRuleViolation::TurnTooSharp {
                        minimum_degrees: MIN_ROAD_TURN_ANGLE_DEGREES,
                        actual_degrees: angle_degrees,
                    });
                }
            }
        }
    }

    for edge in &topology.edges {
        if preexisting.contains(&topology.sources[edge.road].key) {
            continue;
        }
        if let Some((required, actual)) = edge.compromised {
            return Err(RoadRuleViolation::ClearanceTooTight { required, actual });
        }
        if let Some((required_straight, deviation)) = edge.bend_compromised {
            return Err(RoadRuleViolation::TurnTooCloseToJunction {
                required_straight,
                deviation,
            });
        }
    }
    Ok(())
}

fn validate_centerline_grades(
    centerline: &[DVec3],
    max_degrees: f64,
) -> Result<(), RoadRuleViolation> {
    let max_degrees = max_degrees.clamp(0.0, 89.9);
    for segment in centerline.windows(2) {
        let delta = segment[1] - segment[0];
        let horizontal = delta.truncate().length();
        let vertical = delta.z.abs();
        if horizontal < 1e-9 {
            if vertical < 1e-9 {
                continue;
            }
            return Err(RoadRuleViolation::SegmentTooSteep {
                maximum_degrees: max_degrees,
                actual_degrees: 90.0,
            });
        }
        let angle_degrees = vertical.atan2(horizontal).to_degrees();
        if angle_degrees > max_degrees + 1e-6 {
            return Err(RoadRuleViolation::SegmentTooSteep {
                maximum_degrees: max_degrees,
                actual_degrees: angle_degrees,
            });
        }
    }
    Ok(())
}

fn validate_centerline_turns(centerline: &[DVec3]) -> Result<(), RoadRuleViolation> {
    for window in centerline.windows(3) {
        let a = window[0].truncate() - window[1].truncate();
        let b = window[2].truncate() - window[1].truncate();
        let (a_len, b_len) = (a.length(), b.length());
        if a_len < 1e-9 || b_len < 1e-9 {
            return Err(RoadRuleViolation::DegenerateSegment);
        }
        let angle_degrees = (a.dot(b) / (a_len * b_len))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        if angle_degrees + 1e-6 < MIN_ROAD_TURN_ANGLE_DEGREES {
            return Err(RoadRuleViolation::TurnTooSharp {
                minimum_degrees: MIN_ROAD_TURN_ANGLE_DEGREES,
                actual_degrees: angle_degrees,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 0 — collect sources
// ---------------------------------------------------------------------------

fn collect_sources(document: &Document, ghost: Option<&GhostRoad>) -> Vec<SourceRoad> {
    let mut sources = Vec::new();
    for object in document.objects() {
        if let Object::Road {
            id,
            centerline,
            width,
            camber_degrees,
            shape,
            ..
        } = object
        {
            let pts = dedup_centerline(centerline.iter().map(|v| v.pos));
            if pts.len() >= 2 {
                let (slope_left, slope_right) = side_slopes(*shape, *camber_degrees);
                sources.push(SourceRoad {
                    key: RoadKey::Object(*id),
                    centerline: pts,
                    width: *width,
                    slope_left,
                    slope_right,
                });
            }
        }
    }
    if let Some(ghost) = ghost {
        let pts = dedup_centerline(ghost.centerline.iter().copied());
        if pts.len() >= 2 {
            let (slope_left, slope_right) = side_slopes(ghost.shape, ghost.camber_degrees);
            sources.push(SourceRoad {
                key: RoadKey::Ghost,
                centerline: pts,
                width: ghost.width,
                slope_left,
                slope_right,
            });
        }
    }
    sources
}

fn side_slopes(shape: RoadShape, camber_degrees: f64) -> (f64, f64) {
    // z_offsets is signed per side for a unit half-width of `width / 2`;
    // normalize to rise-per-unit-half-width so widths can vary along an edge.
    let probe_width = 2.0;
    let (left, right) = shape.z_offsets(probe_width, camber_degrees);
    (left, right)
}

fn dedup_centerline(points: impl Iterator<Item = DVec3>) -> Vec<DVec3> {
    let mut out: Vec<DVec3> = Vec::new();
    for point in points {
        if out
            .last()
            .is_some_and(|last| (last.truncate() - point.truncate()).length() < 1e-6)
        {
            continue;
        }
        out.push(point);
    }
    out
}

type XyBounds = (DVec2, DVec2);

/// Small immutable 2D BVH used only while resolving a road network. Query
/// results are sorted by their original item index at call sites whenever the
/// legacy scan order is observable (nearest-point ties and node creation).
struct XyBvh {
    item_bounds: Vec<XyBounds>,
    order: Vec<usize>,
    nodes: Vec<XyBvhNode>,
}

#[derive(Clone, Copy)]
struct XyBvhNode {
    bounds: XyBounds,
    left: usize,
    right: usize,
    start: usize,
    count: usize,
}

impl XyBvh {
    const LEAF_ITEMS: usize = 8;

    fn build(item_bounds: Vec<XyBounds>) -> Self {
        let mut index = Self {
            order: (0..item_bounds.len()).collect(),
            item_bounds,
            nodes: Vec::new(),
        };
        if !index.order.is_empty() {
            index.build_node(0, index.order.len());
        }
        index
    }

    fn build_node(&mut self, start: usize, end: usize) -> usize {
        let mut min = DVec2::splat(f64::INFINITY);
        let mut max = DVec2::splat(f64::NEG_INFINITY);
        let mut center_min = DVec2::splat(f64::INFINITY);
        let mut center_max = DVec2::splat(f64::NEG_INFINITY);
        for &item in &self.order[start..end] {
            let bounds = self.item_bounds[item];
            min = min.min(bounds.0);
            max = max.max(bounds.1);
            let center = (bounds.0 + bounds.1) * 0.5;
            center_min = center_min.min(center);
            center_max = center_max.max(center);
        }

        let node_index = self.nodes.len();
        self.nodes.push(XyBvhNode {
            bounds: (min, max),
            left: 0,
            right: 0,
            start,
            count: end - start,
        });
        if end - start <= Self::LEAF_ITEMS {
            return node_index;
        }

        let axis = usize::from((center_max.y - center_min.y) > (center_max.x - center_min.x));
        let middle = start + (end - start) / 2;
        let item_bounds = &self.item_bounds;
        self.order[start..end].select_nth_unstable_by(middle - start, |&a, &b| {
            let ca = (item_bounds[a].0 + item_bounds[a].1)[axis];
            let cb = (item_bounds[b].0 + item_bounds[b].1)[axis];
            ca.total_cmp(&cb).then_with(|| a.cmp(&b))
        });
        let left = self.build_node(start, middle);
        let right = self.build_node(middle, end);
        self.nodes[node_index].left = left;
        self.nodes[node_index].right = right;
        self.nodes[node_index].count = 0;
        node_index
    }

    /// Fill `matches` with original item indices whose bounds overlap `query`.
    /// Returns the number of leaf items examined, useful for scaling tests.
    fn query(
        &self,
        query: XyBounds,
        margin: f64,
        matches: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) -> usize {
        matches.clear();
        stack.clear();
        if self.nodes.is_empty() {
            return 0;
        }
        stack.push(0);
        let mut examined = 0;
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index];
            if !bounds_overlap_xy(node.bounds, query, margin) {
                continue;
            }
            if node.count == 0 {
                stack.push(node.left);
                stack.push(node.right);
                continue;
            }
            for &item in &self.order[node.start..node.start + node.count] {
                examined += 1;
                if bounds_overlap_xy(self.item_bounds[item], query, margin) {
                    matches.push(item);
                }
            }
        }
        examined
    }
}

#[derive(Clone, Copy)]
struct RoadSegmentRef {
    road: usize,
    segment: usize,
}

#[derive(Clone, Copy)]
struct RoadVertexRef {
    road: usize,
    vertex: usize,
}

/// Reusable broad-phase indexes over source bounds, per-road segments, all
/// segments, and vertices. The per-road indexes preserve the original
/// source-pair/segment-pair iteration order after candidate sorting.
struct RoadSpatialIndex {
    source_bounds: Vec<XyBounds>,
    sources: XyBvh,
    road_segments: Vec<XyBvh>,
    interior_vertices: Vec<XyBvh>,
    all_segments: Vec<RoadSegmentRef>,
    all_segment_index: XyBvh,
    all_vertices: Vec<RoadVertexRef>,
    all_vertex_index: XyBvh,
    max_half_width: f64,
}

impl RoadSpatialIndex {
    fn build(sources: &[SourceRoad]) -> Self {
        let source_bounds: Vec<XyBounds> = sources
            .iter()
            .map(|source| points_bounds_xy(source.centerline.iter().copied()))
            .collect();
        let source_index = XyBvh::build(source_bounds.clone());
        let mut road_segments = Vec::with_capacity(sources.len());
        let mut interior_vertices = Vec::with_capacity(sources.len());
        let mut all_segments = Vec::new();
        let mut all_segment_bounds = Vec::new();
        let mut all_vertices = Vec::new();
        let mut all_vertex_bounds = Vec::new();

        for (road, source) in sources.iter().enumerate() {
            let segment_bounds: Vec<XyBounds> = source
                .centerline
                .windows(2)
                .enumerate()
                .map(|(segment, points)| {
                    let bounds = points_bounds_xy(points.iter().copied());
                    all_segments.push(RoadSegmentRef { road, segment });
                    all_segment_bounds.push(bounds);
                    bounds
                })
                .collect();
            road_segments.push(XyBvh::build(segment_bounds));

            let interior_bounds: Vec<XyBounds> = source.centerline[1..source.centerline.len() - 1]
                .iter()
                .map(|point| {
                    let xy = point.truncate();
                    (xy, xy)
                })
                .collect();
            interior_vertices.push(XyBvh::build(interior_bounds));

            for (vertex, point) in source.centerline.iter().enumerate() {
                let xy = point.truncate();
                all_vertices.push(RoadVertexRef { road, vertex });
                all_vertex_bounds.push((xy, xy));
            }
        }

        Self {
            source_bounds,
            sources: source_index,
            road_segments,
            interior_vertices,
            all_segments,
            all_segment_index: XyBvh::build(all_segment_bounds),
            all_vertices,
            all_vertex_index: XyBvh::build(all_vertex_bounds),
            max_half_width: sources
                .iter()
                .map(|source| source.width * 0.5)
                .fold(0.0_f64, f64::max),
        }
    }
}

#[derive(Default)]
struct NodeGrid {
    cells: HashMap<(i64, i64), Vec<usize>>,
    candidates: Vec<usize>,
}

impl NodeGrid {
    fn cell(pos: DVec2) -> (i64, i64) {
        (
            (pos.x / NODE_XY_EPS).floor() as i64,
            (pos.y / NODE_XY_EPS).floor() as i64,
        )
    }

    fn find_or_create(&mut self, nodes: &mut Vec<Node>, pos: DVec3) -> usize {
        self.candidates.clear();
        let cell = Self::cell(pos.truncate());
        for dx in -1..=1 {
            let Some(x) = cell.0.checked_add(dx) else {
                continue;
            };
            for dy in -1..=1 {
                let Some(y) = cell.1.checked_add(dy) else {
                    continue;
                };
                if let Some(indices) = self.cells.get(&(x, y)) {
                    self.candidates.extend_from_slice(indices);
                }
            }
        }
        // The old linear scan always selected the earliest compatible node.
        self.candidates.sort_unstable();
        self.candidates.dedup();
        for &index in &self.candidates {
            let node = &mut nodes[index];
            if (node.pos.truncate() - pos.truncate()).length() < NODE_XY_EPS
                && (node.pos.z - pos.z).abs() <= NODE_Z_TOL
            {
                node.z_sum += pos.z;
                node.z_count += 1;
                node.pos.z = node.z_sum / node.z_count as f64;
                return index;
            }
        }

        let index = nodes.len();
        nodes.push(Node::new(pos));
        self.cells.entry(cell).or_default().push(index);
        index
    }
}

#[derive(Clone, Copy)]
struct AttachmentPolyline {
    first_segment: usize,
    segment_count: usize,
    closed: bool,
}

struct PolylineAttachmentIndex {
    polylines: Vec<AttachmentPolyline>,
    segment_polylines: Vec<usize>,
    segments: Vec<[DVec3; 2]>,
    spatial: XyBvh,
}

impl PolylineAttachmentIndex {
    fn build(document: &Document) -> Self {
        let mut polylines = Vec::new();
        let mut segment_polylines = Vec::new();
        let mut segments = Vec::new();
        let mut bounds = Vec::new();
        for object in document.objects() {
            let Object::Polyline { verts, closed, .. } = object else {
                continue;
            };
            if verts.len() < 2 {
                continue;
            }
            let closed = *closed && verts.len() >= 3;
            let segment_count = if closed { verts.len() } else { verts.len() - 1 };
            let polyline = polylines.len();
            let first_segment = segments.len();
            for segment in 0..segment_count {
                let points = [verts[segment].pos, verts[(segment + 1) % verts.len()].pos];
                segment_polylines.push(polyline);
                segments.push(points);
                bounds.push(points_bounds_xy(points.into_iter()));
            }
            polylines.push(AttachmentPolyline {
                first_segment,
                segment_count,
                closed,
            });
        }
        Self {
            polylines,
            segment_polylines,
            segments,
            spatial: XyBvh::build(bounds),
        }
    }

    fn attached_segments(
        &self,
        junction: DVec3,
        matches: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) -> Vec<[DVec3; 2]> {
        let xy = junction.truncate();
        self.spatial.query((xy, xy), NODE_XY_EPS, matches, stack);
        matches.sort_unstable();

        let mut attached = Vec::new();
        let mut cursor = 0;
        while cursor < matches.len() {
            let polyline_index = self.segment_polylines[matches[cursor]];
            let polyline = self.polylines[polyline_index];
            let mut wanted = Vec::new();
            while cursor < matches.len()
                && self.segment_polylines[matches[cursor]] == polyline_index
            {
                let flat_segment = matches[cursor];
                let [a, b] = self.segments[flat_segment];
                if point_on_segment_xy(junction, a, b)
                    || points_coincident_3d(junction, a)
                    || points_coincident_3d(junction, b)
                {
                    let local_segment = flat_segment - polyline.first_segment;
                    for candidate in [
                        local_segment.wrapping_sub(1),
                        local_segment,
                        local_segment + 1,
                    ] {
                        let local = if polyline.closed {
                            candidate % polyline.segment_count
                        } else if candidate < polyline.segment_count {
                            candidate
                        } else {
                            continue;
                        };
                        let wanted_segment = polyline.first_segment + local;
                        if !wanted.contains(&wanted_segment) {
                            wanted.push(wanted_segment);
                        }
                    }
                }
                cursor += 1;
            }
            attached.extend(wanted.into_iter().map(|segment| self.segments[segment]));
        }
        attached
    }
}

// ---------------------------------------------------------------------------
// Stage 1 — node discovery
// ---------------------------------------------------------------------------

fn build_nodes(sources: &[SourceRoad], document: &Document) -> Vec<Node> {
    let mut nodes: Vec<Node> = Vec::new();
    let spatial = RoadSpatialIndex::build(sources);
    let mut node_grid = NodeGrid::default();
    let mut source_matches = Vec::new();
    let mut segment_matches = Vec::new();
    let mut vertex_matches = Vec::new();
    let mut query_stack = Vec::new();

    // Road endpoints. An endpoint landing inside another road's body (within
    // the wider half-width of its stored centerline in plan, at compatible
    // elevation) attaches to that road. Roads draw from *derived*
    // centerlines, which deviate from the stored ones near junction reroutes,
    // so a stroke ended on the drawn road can sit slightly off the stored
    // line; demanding exact contact would leave it dangling. The node goes
    // onto the target centerline and captures the ending road so the split
    // stage still terminates it here.
    let mut attachments: HashMap<(usize, usize), Vec<(DVec2, f64)>> = HashMap::new();
    for (road_index, source) in sources.iter().enumerate() {
        let first = source.centerline[0];
        let last = *source.centerline.last().expect("len >= 2");
        for endpoint in [first, last] {
            let Some((pos, dist, other)) = attach_endpoint_target(
                sources,
                &spatial,
                road_index,
                endpoint,
                &mut segment_matches,
                &mut query_stack,
            ) else {
                node_grid.find_or_create(&mut nodes, endpoint);
                continue;
            };
            let node = node_grid.find_or_create(&mut nodes, pos);
            nodes[node]
                .capture_roads
                .push((road_index, dist + NODE_XY_EPS));
            let tol = (source.width * 0.5).max(sources[other].width * 0.5);
            let pair = if road_index < other {
                (road_index, other)
            } else {
                (other, road_index)
            };
            attachments
                .entry(pair)
                .or_default()
                .push((pos.truncate(), tol));
        }
    }

    // Mid-segment crossings between roads at compatible elevation form real
    // junction nodes (X-junctions). Incompatible elevations are overpasses
    // and stay disconnected.
    //
    // A crossing landing within the wider road's half-width of an interior
    // centerline vertex snaps onto that vertex: a stroke drawn "through a
    // corner" almost never hits the vertex exactly, and a node just before
    // the bend leaves the bend inside the junction pad where the pad solve
    // (which assumes straight approaches) breaks down. Snapping instead
    // yields the clean shared-vertex junction the stroke intends. The
    // non-vertex road no longer passes through the node exactly, so the node
    // records a capture radius for it and the split stage cuts it at its
    // closest approach.
    let mut crossing_positions: Vec<(DVec3, Option<(usize, f64)>)> = Vec::new();
    for (ai, a) in sources.iter().enumerate() {
        spatial.sources.query(
            spatial.source_bounds[ai],
            NODE_XY_EPS,
            &mut source_matches,
            &mut query_stack,
        );
        source_matches.sort_unstable();
        for &bi in &source_matches {
            if bi <= ai {
                continue;
            }
            let b = &sources[bi];
            for seg_a in a.centerline.windows(2) {
                let seg_a_bounds = points_bounds_xy(seg_a.iter().copied());
                spatial.road_segments[bi].query(
                    seg_a_bounds,
                    NODE_XY_EPS,
                    &mut segment_matches,
                    &mut query_stack,
                );
                segment_matches.sort_unstable();
                for &segment_b in &segment_matches {
                    let seg_b = &b.centerline[segment_b..=segment_b + 1];
                    let Some((xy, ta, tb)) = segment_intersection_xy(
                        seg_a[0].truncate(),
                        seg_a[1].truncate(),
                        seg_b[0].truncate(),
                        seg_b[1].truncate(),
                    ) else {
                        continue;
                    };
                    let za = seg_a[0].z + (seg_a[1].z - seg_a[0].z) * ta;
                    let zb = seg_b[0].z + (seg_b[1].z - seg_b[0].z) * tb;
                    if (za - zb).abs() > NODE_Z_TOL {
                        continue;
                    }
                    // A crossing right at an endpoint attachment between the
                    // same two roads IS the attachment: the endpoint rests a
                    // hair past the centerline it attached to. Skip it so the
                    // junction stays one node instead of gaining a sliver
                    // twin (whose near-parallel ports read as a 0° turn).
                    if attachments.get(&(ai, bi)).is_some_and(|entries| {
                        entries.iter().any(|&(pos, tol)| (xy - pos).length() < tol)
                    }) {
                        continue;
                    }
                    let cross = DVec3::new(xy.x, xy.y, (za + zb) * 0.5);
                    let snap_tol = a.width.max(b.width) * 0.5;
                    // Nearest interior vertex of either road within the snap
                    // tolerance, at compatible elevation. `other` is the road
                    // the node must capture at its closest approach.
                    let mut snapped: Option<(f64, DVec3, usize, [DVec3; 2])> = None;
                    for (road_index, road, other, other_seg) in
                        [(ai, a, bi, seg_b), (bi, b, ai, seg_a)]
                    {
                        spatial.interior_vertices[road_index].query(
                            (xy, xy),
                            snap_tol,
                            &mut vertex_matches,
                            &mut query_stack,
                        );
                        vertex_matches.sort_unstable();
                        for &interior_index in &vertex_matches {
                            let vertex = &road.centerline[interior_index + 1];
                            let d = (vertex.truncate() - xy).length();
                            if d < snap_tol
                                && (vertex.z - cross.z).abs() <= NODE_Z_TOL
                                && snapped.is_none_or(|(best, ..)| d < best)
                            {
                                snapped = Some((d, *vertex, other, [other_seg[0], other_seg[1]]));
                            }
                        }
                    }
                    match snapped {
                        Some((_, vertex, other, seg)) => {
                            let (proj, _) = crate::model::kernel::project_onto_segment(
                                vertex.truncate(),
                                seg[0].truncate(),
                                seg[1].truncate(),
                            );
                            let capture = (proj - vertex.truncate()).length() + NODE_XY_EPS;
                            crossing_positions.push((vertex, Some((other, capture))));
                        }
                        None => crossing_positions.push((cross, None)),
                    }
                }
            }
        }
    }
    for (pos, capture) in crossing_positions {
        let index = node_grid.find_or_create(&mut nodes, pos);
        if let Some(capture) = capture {
            nodes[index].capture_roads.push(capture);
        }
    }

    // Interior vertices resting on another road also form junctions: the
    // draw tool lets a stroke snap onto a road and continue past it, which
    // leaves the contact as a mid-stroke vertex rather than an endpoint or a
    // strict segment crossing.
    let mut interior_hits: Vec<DVec3> = Vec::new();
    for (ai, a) in sources.iter().enumerate() {
        for (vi, v) in a.centerline.iter().enumerate() {
            if vi == 0 || vi + 1 == a.centerline.len() {
                continue;
            }
            let xy = v.truncate();
            spatial.all_vertex_index.query(
                (xy, xy),
                NODE_XY_EPS,
                &mut vertex_matches,
                &mut query_stack,
            );
            let touches_vertex = vertex_matches.iter().any(|&candidate| {
                let candidate = spatial.all_vertices[candidate];
                if candidate.road == ai {
                    return false;
                }
                let other = sources[candidate.road].centerline[candidate.vertex];
                (other.truncate() - xy).length() < NODE_XY_EPS
                    && (other.z - v.z).abs() <= NODE_Z_TOL
            });
            let touches_other_road = if touches_vertex {
                true
            } else {
                spatial.all_segment_index.query(
                    (xy, xy),
                    NODE_XY_EPS,
                    &mut segment_matches,
                    &mut query_stack,
                );
                segment_matches.iter().any(|&candidate| {
                    let candidate = spatial.all_segments[candidate];
                    candidate.road != ai
                        && point_on_segment_xy(
                            *v,
                            sources[candidate.road].centerline[candidate.segment],
                            sources[candidate.road].centerline[candidate.segment + 1],
                        )
                })
            };
            if touches_other_road {
                interior_hits.push(*v);
            }
        }
    }
    for pos in interior_hits {
        node_grid.find_or_create(&mut nodes, pos);
    }

    // Polyline attachments: query only segments whose bounds touch the node,
    // then add the same immediate neighbours as the legacy full scan.
    let attachment_index = PolylineAttachmentIndex::build(document);
    for node in &mut nodes {
        node.attachment_segments =
            attachment_index.attached_segments(node.pos, &mut segment_matches, &mut query_stack);
    }

    nodes
}

/// Closest point on another road's stored centerline for a road endpoint
/// that rests inside that road's body but off its centerline. `None` when
/// the endpoint is a plain dead end or already sits exactly on a centerline
/// (which the strict node/split tolerances handle as before).
fn attach_endpoint_target(
    sources: &[SourceRoad],
    spatial: &RoadSpatialIndex,
    road_index: usize,
    endpoint: DVec3,
    matches: &mut Vec<usize>,
    query_stack: &mut Vec<usize>,
) -> Option<(DVec3, f64, usize)> {
    let own_hw = sources[road_index].width * 0.5;
    let mut best: Option<(DVec3, f64, usize)> = None;
    let xy = endpoint.truncate();
    spatial.all_segment_index.query(
        (xy, xy),
        own_hw.max(spatial.max_half_width),
        matches,
        query_stack,
    );
    // Flat segment indices were built in road/segment order, matching the old
    // nested scan so equal-distance ties keep the same target.
    matches.sort_unstable();
    for &candidate in matches.iter() {
        let candidate = spatial.all_segments[candidate];
        let other_index = candidate.road;
        if other_index == road_index {
            continue;
        }
        let other = &sources[other_index];
        let tol = own_hw.max(other.width * 0.5);
        let seg = &other.centerline[candidate.segment..=candidate.segment + 1];
        let (proj, t) = crate::model::kernel::project_onto_segment(
            endpoint.truncate(),
            seg[0].truncate(),
            seg[1].truncate(),
        );
        let dist = (proj - endpoint.truncate()).length();
        if dist >= tol {
            continue;
        }
        // The endpoint may carry a junction-flattened z from snapping to
        // the drawn line, hence the doubled window (mirrors the capture
        // cuts in `split_roads_at_nodes`).
        let z = seg[0].z + (seg[1].z - seg[0].z) * t;
        if (z - endpoint.z).abs() > 2.0 * NODE_Z_TOL {
            continue;
        }
        if best.is_none_or(|(_, d, _)| dist < d) {
            best = Some((DVec3::new(proj.x, proj.y, z), dist, other_index));
        }
    }
    // Exactly on a line already: keep the endpoint itself as the node.
    best.filter(|&(_, dist, _)| dist > NODE_XY_EPS)
}

// ---------------------------------------------------------------------------
// Stage 2 — split roads into edges at nodes
// ---------------------------------------------------------------------------

fn split_roads_at_nodes(sources: &[SourceRoad], nodes: &mut [Node]) -> Vec<WorkEdge> {
    let mut edges: Vec<WorkEdge> = Vec::new();

    // A captured cut sits off the stored centerline (endpoint attach, corner
    // snap). The bend back onto the stored line is confined to this run from
    // the node so the rest of the road stays exactly where drawn, instead of
    // the whole edge pivoting. It must clear every junction approach zone:
    // clearance + hw / tan(min-branch-half-angle) bounds the zone.
    let max_hw = sources
        .iter()
        .map(|source| source.width * 0.5)
        .fold(0.0_f64, f64::max);
    let reroute_len = ROAD_INTERSECTION_FLAT_CLEARANCE_M
        + max_hw / (MIN_ROAD_TURN_ANGLE_DEGREES * 0.5).to_radians().tan()
        + 1.0;

    for (road_index, source) in sources.iter().enumerate() {
        let pts = &source.centerline;
        let stations = cumulative_stations(pts);
        let total = *stations.last().expect("len >= 2");

        // Cut list: (station, node index, plan deviation from the line).
        let mut cuts: Vec<(f64, usize, f64)> = Vec::new();
        for (node_index, node) in nodes.iter().enumerate() {
            let node_xy = node.pos.truncate();
            // Vertex coincidences.
            for (i, p) in pts.iter().enumerate() {
                if (p.truncate() - node_xy).length() < NODE_XY_EPS
                    && (p.z - node.pos.z).abs() <= NODE_Z_TOL
                {
                    push_cut(&mut cuts, stations[i], node_index, 0.0);
                }
            }
            // Interior segment coincidences. A node that snapped onto another
            // road's vertex captures this road within its recorded radius
            // (cut at the closest approach); everywhere else the strict
            // point-on-line tolerance applies. The z window widens with a
            // loose capture because the crossing and the vertex may each sit
            // up to NODE_Z_TOL from the interpolated z.
            let capture = node
                .capture_roads
                .iter()
                .filter(|&&(road, _)| road == road_index)
                .map(|&(_, radius)| radius)
                .fold(NODE_XY_EPS, f64::max);
            let z_tol = if capture > NODE_XY_EPS {
                2.0 * NODE_Z_TOL
            } else {
                NODE_Z_TOL
            };
            for i in 0..pts.len() - 1 {
                let a = pts[i];
                let b = pts[i + 1];
                let ab = b.truncate() - a.truncate();
                let len_sq = ab.length_squared();
                if len_sq < 1e-12 {
                    continue;
                }
                let t = (node_xy - a.truncate()).dot(ab) / len_sq;
                if !(1e-6..=1.0 - 1e-6).contains(&t) {
                    continue;
                }
                let deviation = (a.truncate() + ab * t - node_xy).length();
                if deviation >= capture {
                    continue;
                }
                let z = a.z + (b.z - a.z) * t;
                if (z - node.pos.z).abs() > z_tol {
                    continue;
                }
                push_cut(
                    &mut cuts,
                    stations[i] + len_sq.sqrt() * t,
                    node_index,
                    deviation,
                );
            }
        }
        cuts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        cuts.dedup_by(|a, b| (a.0 - b.0).abs() < NODE_XY_EPS && a.1 == b.1);

        let anchor_deviation = |nodes: &[Node], node: usize, p: DVec3| {
            (p.truncate() - nodes[node].pos.truncate()).length()
        };

        // Endpoints resting off-line (attached into another road's body) have
        // no exact cut; anchor the start on its node so the guards below can
        // complete the ends. Everything else always has endpoint cuts.
        if cuts.is_empty() {
            let node = nearest_node(nodes, pts[0]);
            cuts.push((0.0, node, anchor_deviation(nodes, node, pts[0])));
        }

        // Merge cuts that landed at nearly the same station (duplicate nodes
        // from crossings right at endpoints): keep the first.
        cuts.dedup_by(|a, b| (a.0 - b.0).abs() < NODE_XY_EPS);
        if cuts.first().map(|c| c.0) != Some(0.0) {
            // Guard: ensure the start station cut exists.
            if let Some(first) = cuts.first().copied()
                && first.0 > NODE_XY_EPS
            {
                // Off-line attach (or fall back to the nearest node at start).
                let node = nearest_node(nodes, pts[0]);
                cuts.insert(0, (0.0, node, anchor_deviation(nodes, node, pts[0])));
            }
        }
        if cuts.last().map(|c| (c.0 - total).abs() < NODE_XY_EPS) != Some(true) {
            let last = *pts.last().expect("len >= 2");
            let node = nearest_node(nodes, last);
            cuts.push((total, node, anchor_deviation(nodes, node, last)));
        }

        // Build one edge per consecutive cut pair.
        for pair in cuts.windows(2) {
            let (s0, n0, dev0) = pair[0];
            let (s1, n1, dev1) = pair[1];
            if s1 - s0 < NODE_XY_EPS {
                continue;
            }
            // Rejoin points for off-line cuts: the derived center runs
            // node → stored line at `reroute_len` → exact stored geometry,
            // instead of pivoting the whole edge onto the node.
            let mut kinks: Vec<f64> = Vec::new();
            if dev0 > NODE_XY_EPS {
                kinks.push(s0 + reroute_len);
            }
            if dev1 > NODE_XY_EPS {
                kinks.push(s1 - reroute_len);
            }
            kinks.retain(|&s| s > s0 + NODE_XY_EPS && s < s1 - NODE_XY_EPS);
            if kinks.len() == 2 && kinks[0] >= kinks[1] {
                kinks.clear(); // reroutes overlap: fall back to the pivot
            }
            let mut waypoints: Vec<(f64, DVec3)> = kinks
                .into_iter()
                .map(|s| (s, point_at_station(pts, &stations, s)))
                .collect();
            for (i, p) in pts.iter().enumerate() {
                if stations[i] > s0 + NODE_XY_EPS && stations[i] < s1 - NODE_XY_EPS {
                    waypoints.push((stations[i], *p));
                } else if stations[i] >= s1 - NODE_XY_EPS {
                    break;
                }
            }
            waypoints.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut center = vec![nodes[n0].pos];
            center.extend(waypoints.into_iter().map(|(_, p)| p));
            center.push(nodes[n1].pos);
            let center = dedup_centerline(center.into_iter());
            if center.len() < 2 {
                continue;
            }
            let edge_index = edges.len();
            let heading_start = polyline_heading(&center, true);
            let heading_end = polyline_heading(&center, false);
            let hw = sources[road_index].width * 0.5;
            nodes[n0].ports.push(Port {
                edge: edge_index,
                at_start: true,
                heading: heading_start,
                hw,
                clearance: ROAD_INTERSECTION_FLAT_CLEARANCE_M,
                // start port: ccw of outward heading = edge's left side.
                slope_ccw: source.slope_left,
                slope_cw: source.slope_right,
            });
            nodes[n1].ports.push(Port {
                edge: edge_index,
                at_start: false,
                heading: heading_end,
                hw,
                clearance: ROAD_INTERSECTION_FLAT_CLEARANCE_M,
                // end port: ccw of outward heading = edge's right side.
                slope_ccw: source.slope_right,
                slope_cw: source.slope_left,
            });
            edges.push(WorkEdge {
                road: road_index,
                start_node: n0,
                end_node: n1,
                center,
                left: Vec::new(),
                right: Vec::new(),
                start_blend: NEUTRAL_BLEND,
                end_blend: NEUTRAL_BLEND,
                compromised: None,
                bend_compromised: None,
            });
        }
    }

    edges
}

const NEUTRAL_BLEND: EndBlend = EndBlend {
    zone: 0.0,
    target_w: 0.0,
    target_slope_left: 0.0,
    target_slope_right: 0.0,
    flatten_z: None,
};

fn push_cut(cuts: &mut Vec<(f64, usize, f64)>, station: f64, node: usize, deviation: f64) {
    if !cuts
        .iter()
        .any(|&(s, n, _)| n == node && (s - station).abs() < NODE_XY_EPS)
    {
        cuts.push((station, node, deviation));
    }
}

/// Point on the stored polyline at `station` (linear in 3D).
fn point_at_station(pts: &[DVec3], stations: &[f64], station: f64) -> DVec3 {
    for i in 0..pts.len() - 1 {
        if station <= stations[i + 1] {
            let seg = stations[i + 1] - stations[i];
            if seg < 1e-9 {
                continue;
            }
            let t = ((station - stations[i]) / seg).clamp(0.0, 1.0);
            return pts[i] + (pts[i + 1] - pts[i]) * t;
        }
    }
    *pts.last().expect("non-empty")
}

fn nearest_node(nodes: &[Node], pos: DVec3) -> usize {
    nodes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.pos - pos)
                .length_squared()
                .partial_cmp(&(b.pos - pos).length_squared())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn polyline_heading(pts: &[DVec3], from_start: bool) -> DVec2 {
    if from_start {
        for p in pts.iter().skip(1) {
            let d = p.truncate() - pts[0].truncate();
            if d.length() > 1e-9 {
                return d.normalize();
            }
        }
    } else {
        let last = pts.last().expect("non-empty").truncate();
        for p in pts.iter().rev().skip(1) {
            let d = p.truncate() - last;
            if d.length() > 1e-9 {
                return d.normalize();
            }
        }
    }
    DVec2::X
}

// ---------------------------------------------------------------------------
// Stage 3 — classify nodes, compute clearances and end blends
// ---------------------------------------------------------------------------

fn classify_nodes(nodes: &mut [Node]) {
    let seam_dot = -(SEAM_STRAIGHT_TOL_DEGREES.to_radians().cos());

    for node in nodes.iter_mut() {
        node.kind = match node.ports.len() {
            0 => continue,
            1 => {
                if node.attachment_segments.is_empty() {
                    NodeKind::DeadEnd
                } else {
                    NodeKind::Attachment
                }
            }
            2 => {
                let dot = node.ports[0].heading.dot(node.ports[1].heading);
                if dot <= seam_dot && node.attachment_segments.is_empty() {
                    NodeKind::Seam
                } else {
                    NodeKind::Junction
                }
            }
            _ => NodeKind::Junction,
        };

        match node.kind {
            NodeKind::Junction => {
                // Flat clearance per port grows so edge miters have room:
                // base + half-width / tan(θ/2) against every other branch.
                let headings: Vec<(DVec2, f64)> = node
                    .ports
                    .iter()
                    .map(|port| (port.heading, port.hw))
                    .collect();
                for port in &mut node.ports {
                    let mut clearance = ROAD_INTERSECTION_FLAT_CLEARANCE_M;
                    for &(other_heading, other_hw) in &headings {
                        let dot = port.heading.dot(other_heading).clamp(-1.0, 1.0);
                        if dot > 1.0 - 1e-9 {
                            continue; // itself / parallel duplicate
                        }
                        let angle = dot.acos();
                        // Minimum turn rule keeps θ/2 ≥ 15°; floor the tangent
                        // accordingly so bad inputs can't explode the zone.
                        let tangent = (angle * 0.5).tan().max(15f64.to_radians().tan());
                        clearance = clearance.max(
                            ROAD_INTERSECTION_FLAT_CLEARANCE_M + port.hw.max(other_hw) / tangent,
                        );
                    }
                    port.clearance = clearance;
                }
            }
            NodeKind::Seam => {
                // X2/X3: both roads meet the seam at averaged width and
                // camber, blending back to their own settings over the runoff.
                let hw = (node.ports[0].hw + node.ports[1].hw) * 0.5;
                let shared_a = (node.ports[0].slope_ccw + node.ports[1].slope_cw) * 0.5;
                let shared_b = (node.ports[0].slope_cw + node.ports[1].slope_ccw) * 0.5;
                for (index, port) in node.ports.iter_mut().enumerate() {
                    port.hw = hw;
                    port.clearance = SEAM_BLEND_M;
                    if index == 0 {
                        port.slope_ccw = shared_a;
                        port.slope_cw = shared_b;
                    } else {
                        port.slope_ccw = shared_b;
                        port.slope_cw = shared_a;
                    }
                }
            }
            NodeKind::Attachment => {
                for port in &mut node.ports {
                    port.clearance = ROAD_INTERSECTION_FLAT_CLEARANCE_M;
                    port.slope_ccw = 0.0;
                    port.slope_cw = 0.0;
                }
            }
            NodeKind::DeadEnd => {}
        }
    }
}

fn end_blend_for(node: &Node, port: &Port, source: &SourceRoad) -> EndBlend {
    // Port slopes are expressed ccw/cw of the outward heading; convert back
    // to the edge's left/right of travel.
    let (slope_left, slope_right) = if port.at_start {
        (port.slope_ccw, port.slope_cw)
    } else {
        (port.slope_cw, port.slope_ccw)
    };
    match node.kind {
        NodeKind::DeadEnd => NEUTRAL_BLEND,
        NodeKind::Seam => EndBlend {
            zone: port.clearance,
            target_w: port.hw * 2.0,
            target_slope_left: slope_left,
            target_slope_right: slope_right,
            flatten_z: None,
        },
        NodeKind::Junction | NodeKind::Attachment => EndBlend {
            zone: port.clearance,
            target_w: source.width,
            target_slope_left: 0.0,
            target_slope_right: 0.0,
            flatten_z: Some(node.pos.z),
        },
    }
}

// ---------------------------------------------------------------------------
// Stage 4 — center profile (junction flattening + grade-break flats)
// ---------------------------------------------------------------------------

fn build_center_profiles(sources: &[SourceRoad], edges: &mut [WorkEdge], nodes: &[Node]) {
    for (edge_index, edge) in edges.iter_mut().enumerate() {
        let source = &sources[edge.road];
        let start_node = &nodes[edge.start_node];
        let end_node = &nodes[edge.end_node];
        edge.start_blend = port_of(start_node, edge_index, true)
            .map(|port| end_blend_for(start_node, port, source))
            .unwrap_or(NEUTRAL_BLEND);
        edge.end_blend = port_of(end_node, edge_index, false)
            .map(|port| end_blend_for(end_node, port, source))
            .unwrap_or(NEUTRAL_BLEND);

        let total = polyline_length_xy(&edge.center);
        // If both flat zones can't fit, shrink them proportionally so they
        // meet in the middle instead of fighting (placement validation should
        // normally prevent this).
        let mut zone_start = edge.start_blend.zone;
        let mut zone_end = edge.end_blend.zone;
        let need = zone_start + zone_end;
        if need > total && need > 1e-9 {
            let scale = total / need;
            zone_start *= scale;
            zone_end *= scale;
            edge.start_blend.zone = zone_start;
            edge.end_blend.zone = zone_end;
            // Seam blends shrinking is cosmetic; junction/attachment flat
            // zones shrinking means the flat approach cannot fit.
            if edge.start_blend.flatten_z.is_some() || edge.end_blend.flatten_z.is_some() {
                mark_compromised(&mut edge.compromised, need, total);
            }
        }

        if let Some(z) = edge.start_blend.flatten_z {
            flatten_from_start(&mut edge.center, z, zone_start);
        }
        if let Some(z) = edge.end_blend.flatten_z {
            edge.center.reverse();
            flatten_from_start(&mut edge.center, z, zone_end);
            edge.center.reverse();
        }

        if let Some((required, actual)) = insert_grade_break_flats(&mut edge.center) {
            mark_compromised(&mut edge.compromised, required, actual);
        }

        // Extra stations through blend zones so camber/width transitions are
        // sampled smoothly rather than jumping between sparse vertices.
        for fraction in [1.0 / 3.0, 2.0 / 3.0, 1.0] {
            if zone_start > 1e-9 {
                insert_station_at(&mut edge.center, zone_start * fraction);
            }
            if zone_end > 1e-9 {
                let total = polyline_length_xy(&edge.center);
                insert_station_at(&mut edge.center, total - zone_end * fraction);
            }
        }

        // Junction/attachment approaches must be straight in plan: the pad
        // solve trusts the port heading for the whole zone, so a bend inside
        // it puts pad corners off the road (crossing side lines). Crossings
        // that land near a vertex snap onto it in `build_nodes`; anything
        // still bending here is refused via `validate_ghost`.
        for (at_start, node, blend) in [
            (true, start_node, edge.start_blend),
            (false, end_node, edge.end_blend),
        ] {
            if blend.flatten_z.is_none() {
                continue;
            }
            let Some(port) = port_of(node, edge_index, at_start) else {
                continue;
            };
            let deviation =
                straight_approach_deviation(&edge.center, port.heading, blend.zone, at_start);
            if deviation > port.hw * JUNCTION_APPROACH_DEVIATION_FRAC
                && edge
                    .bend_compromised
                    .is_none_or(|(_, worst)| deviation > worst)
            {
                edge.bend_compromised = Some((blend.zone, deviation));
            }
        }
    }
}

/// Worst lateral deviation of the centerline from a straight run along
/// `heading` within `zone` metres of the edge's node end. Doubling back past
/// the node counts as deviation too.
fn straight_approach_deviation(
    center: &[DVec3],
    heading: DVec2,
    zone: f64,
    from_start: bool,
) -> f64 {
    if center.len() < 2 || zone <= 1e-9 {
        return 0.0;
    }
    let pts: Vec<DVec2> = if from_start {
        center.iter().map(|p| p.truncate()).collect()
    } else {
        center.iter().rev().map(|p| p.truncate()).collect()
    };
    let node = pts[0];
    let mut worst = 0.0_f64;
    let mut travelled = 0.0;
    for pair in pts.windows(2) {
        travelled += (pair[1] - pair[0]).length();
        let rel = pair[1] - node;
        let along = rel.dot(heading);
        let perp = (rel - heading * along).length();
        worst = worst.max(perp).max(-along);
        if travelled >= zone - 1e-9 {
            break;
        }
    }
    worst
}

/// Force z = `node_z` over the first `clearance` metres, inserting a boundary
/// vertex where the flat ends so the grade change has a station.
fn flatten_from_start(pts: &mut Vec<DVec3>, node_z: f64, clearance: f64) {
    if pts.len() < 2 || clearance < 1e-9 {
        if let Some(first) = pts.first_mut() {
            first.z = node_z;
        }
        return;
    }
    pts[0].z = node_z;
    let mut travelled = 0.0;
    let mut i = 0;
    while i + 1 < pts.len() {
        let a = pts[i];
        let b = pts[i + 1];
        let seg = (b.truncate() - a.truncate()).length();
        if seg < 1e-9 {
            pts[i + 1].z = node_z;
            i += 1;
            continue;
        }
        let remaining = clearance - travelled;
        if seg <= remaining + 1e-9 {
            pts[i + 1].z = node_z;
            travelled += seg;
            i += 1;
            continue;
        }
        let t = remaining / seg;
        if t > 1e-6 {
            let xy = a.truncate() + (b.truncate() - a.truncate()) * t;
            pts.insert(i + 1, DVec3::new(xy.x, xy.y, node_z));
        }
        return;
    }
}

/// Every inclined segment reserves flat clearance at both ends, carved out of
/// the segment itself — unless an adjacent flat run already provides it
/// (which makes this idempotent over already-flattened legacy centerlines).
///
/// Returns the worst `(required, available)` shortfall when a needed pocket
/// did not fit, so the edge can be flagged compromised.
fn insert_grade_break_flats(pts: &mut Vec<DVec3>) -> Option<(f64, f64)> {
    let clearance = ROAD_INTERSECTION_FLAT_CLEARANCE_M;
    let original = pts.clone();
    let mut shortfall: Option<(f64, f64)> = None;
    let mut out: Vec<DVec3> = vec![original[0]];
    for i in 0..original.len() - 1 {
        let a = original[i];
        let b = original[i + 1];
        let seg = (b.truncate() - a.truncate()).length();
        if (b.z - a.z).abs() > 1e-6 && seg > 1e-9 {
            let dir = (b.truncate() - a.truncate()) / seg;
            // A turn vertex must sit in a flat pocket unconditionally: a
            // mitered corner on a grade skews the side lines, because the
            // miter points are displaced in plan while carrying the vertex z.
            // Straight grade breaks only need the flat if an adjacent run
            // doesn't already provide it (keeps re-resolution of legacy
            // centerlines that stored their flats stable).
            let need_before =
                is_turn_vertex(&original, i) || flat_run_before(&original, i) < clearance - 1e-6;
            let need_after = is_turn_vertex(&original, i + 1)
                || flat_run_after(&original, i + 1) < clearance - 1e-6;
            let da = if need_before { clearance } else { 0.0 };
            let db = if need_after { clearance } else { 0.0 };
            if da + db < seg - 1e-6 {
                if need_before {
                    let xy = a.truncate() + dir * da;
                    out.push(DVec3::new(xy.x, xy.y, a.z));
                }
                if need_after {
                    let xy = b.truncate() - dir * db;
                    out.push(DVec3::new(xy.x, xy.y, b.z));
                }
            } else if da + db > 0.0 {
                mark_compromised(&mut shortfall, da + db, seg);
            }
        }
        out.push(b);
    }
    *pts = out;
    shortfall
}

/// Record a flat-zone shortfall, keeping the worst deficit seen so far.
fn mark_compromised(slot: &mut Option<(f64, f64)>, required: f64, actual: f64) {
    let worse = slot.is_none_or(|(r, a)| required - actual > r - a);
    if worse {
        *slot = Some((required, actual));
    }
}

/// True when the centerline changes heading at `vertex` by more than a
/// negligible amount (0.5°), i.e. the vertex is a corner rather than a
/// collinear grade break.
fn is_turn_vertex(pts: &[DVec3], vertex: usize) -> bool {
    if vertex == 0 || vertex + 1 >= pts.len() {
        return false;
    }
    let (Some(dir_in), Some(dir_out)) = (
        seg_dir(pts[vertex - 1], pts[vertex]),
        seg_dir(pts[vertex], pts[vertex + 1]),
    ) else {
        return false;
    };
    dir_in.dot(dir_out) < 0.5_f64.to_radians().cos()
}

fn flat_run_before(pts: &[DVec3], vertex: usize) -> f64 {
    let mut run = 0.0;
    let mut i = vertex;
    while i > 0 {
        let a = pts[i - 1];
        let b = pts[i];
        if (b.z - a.z).abs() > 1e-6 {
            break;
        }
        run += (b.truncate() - a.truncate()).length();
        i -= 1;
    }
    run
}

fn flat_run_after(pts: &[DVec3], vertex: usize) -> f64 {
    let mut run = 0.0;
    let mut i = vertex;
    while i + 1 < pts.len() {
        let a = pts[i];
        let b = pts[i + 1];
        if (b.z - a.z).abs() > 1e-6 {
            break;
        }
        run += (b.truncate() - a.truncate()).length();
        i += 1;
    }
    run
}

fn insert_station_at(pts: &mut Vec<DVec3>, station: f64) {
    if station <= 1e-6 {
        return;
    }
    let mut travelled = 0.0;
    for i in 0..pts.len() - 1 {
        let a = pts[i];
        let b = pts[i + 1];
        let seg = (b.truncate() - a.truncate()).length();
        if seg < 1e-9 {
            continue;
        }
        if travelled + seg >= station - 1e-6 {
            let t = ((station - travelled) / seg).clamp(0.0, 1.0);
            if t > 1e-4 && t < 1.0 - 1e-4 {
                pts.insert(i + 1, a + (b - a) * t);
            }
            return;
        }
        travelled += seg;
    }
}

fn cumulative_stations(pts: &[DVec3]) -> Vec<f64> {
    let mut stations = Vec::with_capacity(pts.len());
    let mut s = 0.0;
    stations.push(0.0);
    for pair in pts.windows(2) {
        s += (pair[1].truncate() - pair[0].truncate()).length();
        stations.push(s);
    }
    stations
}

fn polyline_length_xy(pts: &[DVec3]) -> f64 {
    pts.windows(2)
        .map(|pair| (pair[1].truncate() - pair[0].truncate()).length())
        .sum()
}

// ---------------------------------------------------------------------------
// Stage 5 — cross-section sampling
// ---------------------------------------------------------------------------

fn sample_cross_sections(sources: &[SourceRoad], edges: &mut [WorkEdge]) {
    for edge in edges.iter_mut() {
        let source = &sources[edge.road];
        let pts = &edge.center;
        let n = pts.len();
        let stations = cumulative_stations(pts);
        let total = *stations.last().expect("len >= 2");

        let base_w = source.width;
        let base_l = source.slope_left;
        let base_r = source.slope_right;
        let sb = edge.start_blend;
        let eb = edge.end_blend;
        // Neutral blends target the edge's own settings.
        let (sb_w, sb_l, sb_r) = if sb.zone > 1e-9 {
            (sb.target_w, sb.target_slope_left, sb.target_slope_right)
        } else {
            (base_w, base_l, base_r)
        };
        let (eb_w, eb_l, eb_r) = if eb.zone > 1e-9 {
            (eb.target_w, eb.target_slope_left, eb.target_slope_right)
        } else {
            (base_w, base_l, base_r)
        };

        let mut left = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);
        for i in 0..n {
            let s = stations[i];
            let fs = blend_factor(s, sb.zone);
            let fe = blend_factor(total - s, eb.zone);
            let w = base_w + (1.0 - fs) * (sb_w - base_w) + (1.0 - fe) * (eb_w - base_w);
            let slope_l = base_l + (1.0 - fs) * (sb_l - base_l) + (1.0 - fe) * (eb_l - base_l);
            let slope_r = base_r + (1.0 - fs) * (sb_r - base_r) + (1.0 - fe) * (eb_r - base_r);
            let normal = miter_normal(pts, i);
            let hw = w * 0.5;
            let c = pts[i];
            left.push(DVec3::new(
                c.x + normal.x * hw,
                c.y + normal.y * hw,
                c.z + slope_l * hw,
            ));
            right.push(DVec3::new(
                c.x - normal.x * hw,
                c.y - normal.y * hw,
                c.z + slope_r * hw,
            ));
        }
        edge.left = left;
        edge.right = right;
    }
}

fn blend_factor(distance_from_end: f64, zone: f64) -> f64 {
    if zone <= 1e-9 {
        return 1.0;
    }
    let x = (distance_from_end / zone).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Left-perpendicular offset direction at vertex `i`, miter-scaled so that
/// offsetting by `hw` along it reproduces the intersection of the two
/// adjacent offset edges (capped at MITER_LIMIT).
fn miter_normal(pts: &[DVec3], i: usize) -> DVec2 {
    let n = pts.len();
    let dir_in = if i > 0 {
        seg_dir(pts[i - 1], pts[i])
    } else {
        None
    };
    let dir_out = if i + 1 < n {
        seg_dir(pts[i], pts[i + 1])
    } else {
        None
    };
    match (dir_in, dir_out) {
        (Some(din), Some(dout)) => {
            let perp_in = DVec2::new(-din.y, din.x);
            let perp_out = DVec2::new(-dout.y, dout.x);
            let bis = perp_in + perp_out;
            if bis.length() < 1e-9 {
                return perp_out;
            }
            let bis = bis.normalize();
            // cos of the half-angle between the adjacent edges.
            let cos_half = bis.dot(perp_out).clamp(1.0 / MITER_LIMIT, 1.0);
            bis / cos_half
        }
        (Some(d), None) | (None, Some(d)) => DVec2::new(-d.y, d.x),
        (None, None) => DVec2::Y,
    }
}

fn seg_dir(a: DVec3, b: DVec3) -> Option<DVec2> {
    let d = b.truncate() - a.truncate();
    let len = d.length();
    (len > 1e-9).then(|| d / len)
}

// ---------------------------------------------------------------------------
// Stage 6 — junction solver
// ---------------------------------------------------------------------------

fn solve_junctions(edges: &mut [WorkEdge], nodes: &[Node]) {
    for node in nodes.iter() {
        match node.kind {
            NodeKind::DeadEnd => {}
            NodeKind::Attachment => solve_attachment(node, edges),
            NodeKind::Seam | NodeKind::Junction => solve_pad(node, edges),
        }
    }
}

/// Compute the shared corner between adjacent ports and trim each side line
/// to it. Corners exist once; both neighbouring edges receive the same value.
fn solve_pad(node: &Node, edges: &mut [WorkEdge]) {
    let mut order: Vec<usize> = (0..node.ports.len()).collect();
    order.sort_by(|&a, &b| {
        let ha = node.ports[a].heading;
        let hb = node.ports[b].heading;
        ha.y.atan2(ha.x)
            .partial_cmp(&hb.y.atan2(hb.x))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let node_xy = node.pos.truncate();
    let count = order.len();
    for k in 0..count {
        let pi = &node.ports[order[k]];
        let pj = &node.ports[order[(k + 1) % count]];
        // Skip the self-pair on single-port pads (shouldn't occur: pads have
        // ≥ 2 ports), and the duplicate second pair on 2-port nodes is fine —
        // it computes the other side's corner.
        if count == 1 {
            break;
        }
        let hi = pi.heading;
        let hj = pj.heading;
        let base_i = node_xy + DVec2::new(-hi.y, hi.x) * pi.hw; // ccw side of i
        let base_j = node_xy + DVec2::new(hj.y, -hj.x) * pj.hw; // cw side of j
        let max_reach = MITER_LIMIT * 2.0 * pi.hw.max(pj.hw);
        let corner_xy = intersect_lines(base_i, hi, base_j, hj)
            .filter(|xy| (*xy - node_xy).length() <= max_reach)
            .unwrap_or((base_i + base_j) * 0.5);

        let corner_z = match node.kind {
            // Flat pad plane.
            NodeKind::Junction => node.pos.z,
            // Seam: shared cross-slope value on this geometric side.
            NodeKind::Seam => {
                let shared = (pi.slope_ccw + pj.slope_cw) * 0.5;
                let hw = (pi.hw + pj.hw) * 0.5;
                node.pos.z + shared * hw
            }
            _ => node.pos.z,
        };
        let corner = DVec3::new(corner_xy.x, corner_xy.y, corner_z);

        let pin_i = matches!(node.kind, NodeKind::Junction);
        trim_port_side(edges, pi, true, node_xy, corner, pin_i);
        trim_port_side(edges, pj, false, node_xy, corner, pin_i);
    }
}

/// Terminate each side line exactly on the attached polyline: shoot the side
/// line's own direction at the attachment segments, take the contact with
/// the attachment's interpolated z, then blend camber/z away from it.
fn solve_attachment(node: &Node, edges: &mut [WorkEdge]) {
    let node_xy = node.pos.truncate();
    for port in &node.ports {
        for ccw in [true, false] {
            let hw = port.hw;
            let h = port.heading;
            let perp = if ccw {
                DVec2::new(-h.y, h.x)
            } else {
                DVec2::new(h.y, -h.x)
            };
            let base = node_xy + perp * hw;
            let mut best: Option<(f64, DVec3)> = None;
            for segment in &node.attachment_segments {
                let a = segment[0].truncate();
                let b = segment[1].truncate();
                let Some((xy, t)) = intersect_line_with_segment(base, h, a, b) else {
                    continue;
                };
                let z = segment[0].z + (segment[1].z - segment[0].z) * t;
                let ray_t = (xy - base).dot(h);
                if best
                    .as_ref()
                    .is_none_or(|(best_t, _)| ray_t.abs() < best_t.abs())
                {
                    best = Some((ray_t, DVec3::new(xy.x, xy.y, z)));
                }
            }
            let Some((_, corner)) = best else {
                continue;
            };
            trim_port_side(edges, port, ccw, node_xy, corner, true);
        }
    }
}

/// Replace the node-side terminus of one side line with `corner`, dropping
/// samples the corner overtakes (inside the pad) and keeping samples beyond
/// it. Outer corners (negative along) extend the line instead. When `pin_z`
/// is set, z re-blends from the corner over the port's clearance zone.
fn trim_port_side(
    edges: &mut [WorkEdge],
    port: &Port,
    ccw: bool,
    node_xy: DVec2,
    corner: DVec3,
    pin_z: bool,
) {
    let edge = &mut edges[port.edge];
    // ccw of outward heading ↔ left of travel at start ports, right at end.
    let side = match (port.at_start, ccw) {
        (true, true) | (false, false) => &mut edge.left,
        (true, false) | (false, true) => &mut edge.right,
    };
    if side.len() < 2 {
        return;
    }
    let reversed = !port.at_start;
    if reversed {
        side.reverse();
    }

    let h = port.heading;
    let along = |p: &DVec3| (p.truncate() - node_xy).dot(h);
    let corner_along = along(&corner);
    // First sample the corner does not overtake. Inner corners (positive
    // along) drop pad-interior samples; outer corners (negative along) keep
    // everything and the corner extends the line.
    let keep_from = side
        .iter()
        .position(|p| along(p) > corner_along + 1e-6)
        .unwrap_or(side.len() - 1)
        .min(side.len() - 1);
    let mut new_side = Vec::with_capacity(side.len() - keep_from + 1);
    new_side.push(corner);
    new_side.extend_from_slice(&side[keep_from..]);

    if pin_z && port.clearance > corner_along + 1e-6 {
        let zone = port.clearance;
        for p in new_side.iter_mut().skip(1) {
            let a = along(p);
            if a >= zone {
                break;
            }
            let f = blend_factor(a - corner_along, zone - corner_along);
            p.z = corner.z + (p.z - corner.z) * f;
        }
    }

    if reversed {
        new_side.reverse();
    }
    *side = new_side;
}

/// Remove self-intersection loops from a side line. On the inside of a tight
/// bend the mitered offset folds back over itself; splitting at the crossing
/// and dropping the loop leaves a clean pinched corner. Endpoints (shared
/// junction corners) are never touched.
fn remove_side_line_folds(pts: &mut Vec<DVec3>) {
    let max_passes = pts.len();
    for _ in 0..max_passes {
        let n = pts.len();
        if n < 4 {
            return;
        }
        let mut fold: Option<(usize, usize, DVec3)> = None;
        'search: for i in 0..n - 1 {
            for j in (i + 2)..(n - 1) {
                if let Some((xy, t, _)) = segment_intersection_xy(
                    pts[i].truncate(),
                    pts[i + 1].truncate(),
                    pts[j].truncate(),
                    pts[j + 1].truncate(),
                ) {
                    let z = pts[i].z + (pts[i + 1].z - pts[i].z) * t;
                    fold = Some((i, j, DVec3::new(xy.x, xy.y, z)));
                    break 'search;
                }
            }
        }
        let Some((i, j, crossing)) = fold else {
            return;
        };
        pts.splice(i + 1..=j, [crossing]);
    }
}

// ---------------------------------------------------------------------------
// Small geometry helpers
// ---------------------------------------------------------------------------

fn intersect_lines(a: DVec2, da: DVec2, b: DVec2, db: DVec2) -> Option<DVec2> {
    crate::model::kernel::line_line(a, da, b, db)
}

/// Intersect an unbounded line with a bounded segment; returns the point and
/// the segment parameter. Accepts touch points up to `XY_TOL` metres past
/// either segment endpoint.
fn intersect_line_with_segment(
    line_point: DVec2,
    line_dir: DVec2,
    seg_a: DVec2,
    seg_b: DVec2,
) -> Option<(DVec2, f64)> {
    crate::model::kernel::line_segment(
        line_point,
        line_dir,
        seg_a,
        seg_b,
        crate::model::kernel::XY_TOL,
    )
}

/// Strict-interior XY intersection of two segments; returns point and both
/// parameters.
fn segment_intersection_xy(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> Option<(DVec2, f64, f64)> {
    match crate::model::kernel::segment_segment(a, b, c, d) {
        crate::model::kernel::SegSeg::Crossing { point, t, u } => Some((point, t, u)),
        _ => None,
    }
}

fn points_bounds_xy(points: impl Iterator<Item = DVec3>) -> (DVec2, DVec2) {
    let mut min = DVec2::splat(f64::INFINITY);
    let mut max = DVec2::splat(f64::NEG_INFINITY);
    for point in points {
        min = min.min(point.truncate());
        max = max.max(point.truncate());
    }
    (min, max)
}

fn bounds_overlap_xy(a: (DVec2, DVec2), b: (DVec2, DVec2), margin: f64) -> bool {
    a.0.x <= b.1.x + margin
        && a.1.x >= b.0.x - margin
        && a.0.y <= b.1.y + margin
        && a.1.y >= b.0.y - margin
}

fn point_on_segment_xy(point: DVec3, a: DVec3, b: DVec3) -> bool {
    crate::model::kernel::point_on_segment_interior_3d(point, a, b)
}

fn points_coincident_3d(a: DVec3, b: DVec3) -> bool {
    crate::model::kernel::points_coincident_3d(a, b)
}
