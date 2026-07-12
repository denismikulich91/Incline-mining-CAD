use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt, fs, io,
    path::Path,
};

use glam::DVec3;

use super::tri00t;

const SNAPPY_STREAM_IDENTIFIER: &[u8] = b"\xff\x06\x00\x00sNaPpY";
/// A DUF is parsed from one expanded design stream. Bound that stream before
/// the heuristic scanners can retain geometry derived from it.
const MAX_DUF_EXPANDED_BYTES: usize = 512 << 20;
const DUF_DECODE_CHUNK_BYTES: usize = 64 << 10;

const PROP_LAYER_NAME: u32 = 0x594;
const PROP_COORD_X0: u32 = 0x52c;
const PROP_COORD_Y0: u32 = 0x52d;
const PROP_COORD_Z0: u32 = 0x52e;
const PROP_COORD_X1: u32 = 0x52f;
const PROP_COORD_Y1: u32 = 0x530;
const PROP_COORD_Z1: u32 = 0x531;
const PROP_POINT_POSITION: u32 = 0x1dd;
const PROP_OBJECT_GUID: u32 = 0x535;
const PROP_ENTITY_LAYER_REF: u32 = 0x2a3;

const TABLE_ENTRY_CLASS_ID: u32 = 0x529;
const POLYLINE_CLASS_ID: u32 = 0x47;
const POINT_CLASS_ID: u32 = 0xc8;
const POLYLINE_VERTEX_COUNT_MARKER: &[u8] = b"\x06\xf9\x00\x00\x00";
const POLYLINE_VERTEX_ARRAY_MARKER: &[u8] = b"\x1c\x00";
const POLYFACE_FACE_COUNT_MARKER: &[u8] = b"\x06\xdf\x01\x00\x00";
const POLYFACE_VERTEX_COUNT_MARKER: &[u8] = b"\x06\xe4\x01\x00\x00";
const POLYFACE_VERTEX_ARRAY_MARKER: &[u8] = b"\x1b\x00\x00\x00\x00";
const POLYFACE_COMPACT_VERTEX_MARKER: u8 = 0x1b;
const FACE_RECORD_INTS: usize = 5;
const FACE_RECORD_BYTES: usize = FACE_RECORD_INTS * 4;
const POLYFACE_CLASS_ID: u32 = 0xce;
/// Entity records carry their layer reference (a GUID property) within the
/// first few property slots, well before any bulk geometry arrays. Limiting
/// the scan avoids false tag matches inside vertex/face data.
const ENTITY_HEAD_SCAN_LIMIT: usize = 200;
const MAX_REASONABLE_COORDINATE_ABS: f64 = 1.0e9;
const MAX_REASONABLE_GEOMETRY_SPAN: f64 = 10_000_000.0;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct DufData {
    pub(crate) layers: Vec<DufLayer>,
    pub(crate) points: Vec<DufPoint>,
    pub(crate) polylines: Vec<DufPolyline>,
    pub(crate) polyfaces: Vec<DufPolyface>,
    pub(crate) skipped: DufSkipped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DufLayer {
    pub(crate) name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DufPolyline {
    pub(crate) layer_name: String,
    pub(crate) vertices: Vec<DVec3>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DufPoint {
    pub(crate) layer_name: String,
    pub(crate) position: DVec3,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DufPolyface {
    pub(crate) layer_name: String,
    pub(crate) vertices: Vec<tri00t::Vertex>,
    pub(crate) faces: Vec<[u32; 3]>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DufSkipped {
    pub(crate) unsupported_design_entities: usize,
    pub(crate) unsupported_mesh_entities: usize,
}

#[derive(Debug)]
pub(crate) enum DufError {
    Io(io::Error),
    Decompress(String),
    Invalid(String),
}

impl fmt::Display for DufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Decompress(message) => write!(f, "{message}"),
            Self::Invalid(message) => write!(f, "invalid DUF: {message}"),
        }
    }
}

impl Error for DufError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Decompress(_) => None,
            Self::Invalid(_) => None,
        }
    }
}

pub(crate) fn read_duf(path: impl AsRef<Path>) -> Result<DufData, DufError> {
    let bytes = fs::read(path).map_err(DufError::Io)?;
    read_duf_bytes(&bytes)
}

pub(crate) fn read_duf_bytes(bytes: &[u8]) -> Result<DufData, DufError> {
    let data = decode_snappy_frame(bytes)?;
    // `scan_layers` deliberately supplies a synthetic "DUF Import" layer
    // when a real file has geometry but no readable layer table. Do not use
    // that fallback as evidence that arbitrary Snappy payloads are DUFs.
    let has_named_layer = scan_string_property(&data, PROP_LAYER_NAME)
        .into_iter()
        .map(|name| clean_layer_name(&name))
        .any(|name| is_useful_layer_name(&name));
    let parsed = parse_duf_stream(&data);
    let recognized = has_named_layer
        || !parsed.points.is_empty()
        || !parsed.polylines.is_empty()
        || !parsed.polyfaces.is_empty()
        || parsed.skipped.unsupported_design_entities > 0
        || parsed.skipped.unsupported_mesh_entities > 0;
    if !recognized {
        return Err(DufError::Invalid(
            "expanded stream contains no recognizable design structure".to_owned(),
        ));
    }
    Ok(parsed)
}

pub(crate) fn decode_snappy_frame(bytes: &[u8]) -> Result<Vec<u8>, DufError> {
    decode_snappy_frame_with_limit(bytes, MAX_DUF_EXPANDED_BYTES)
}

fn decode_snappy_frame_with_limit(
    bytes: &[u8],
    max_expanded_bytes: usize,
) -> Result<Vec<u8>, DufError> {
    if !bytes.starts_with(SNAPPY_STREAM_IDENTIFIER) {
        return Err(DufError::Decompress(
            "DUF file is not a Snappy framed stream".to_owned(),
        ));
    }
    let mut decoder = snap::read::FrameDecoder::new(bytes);
    let mut out = Vec::new();
    let mut chunk = [0_u8; DUF_DECODE_CHUNK_BYTES];
    loop {
        let read = io::Read::read(&mut decoder, &mut chunk).map_err(|error| {
            DufError::Decompress(format!("Snappy decompression failed: {error}"))
        })?;
        if read == 0 {
            break;
        }
        let expanded_len = out.len().checked_add(read).ok_or_else(|| {
            DufError::Decompress("Snappy expanded size overflows addressable memory".to_owned())
        })?;
        if expanded_len > max_expanded_bytes {
            return Err(DufError::Decompress(format!(
                "Snappy-expanded DUF exceeds the import limit of {max_expanded_bytes} bytes"
            )));
        }

        if expanded_len > out.capacity() {
            // Grow geometrically but never request capacity beyond the
            // configured logical ceiling. `try_reserve_exact` turns address
            // space exhaustion into an import error instead of a panic.
            let target_capacity = out
                .capacity()
                .saturating_mul(2)
                .max(expanded_len)
                .min(max_expanded_bytes);
            out.try_reserve_exact(target_capacity - out.len())
                .map_err(|error| {
                    DufError::Decompress(format!(
                        "Could not allocate the Snappy-expanded DUF stream: {error}"
                    ))
                })?;
        }
        out.extend_from_slice(&chunk[..read]);
    }
    Ok(out)
}

fn parse_duf_stream(data: &[u8]) -> DufData {
    let layers = scan_layers(data);
    let layer_names = layer_names_by_guid(data);
    let points = scan_point_records(data, &layer_names);
    let mut polylines = scan_polyline_records(data, &layer_names);
    let polylines_from_fallback = polylines.is_empty();
    if polylines_from_fallback {
        polylines = scan_coordinate_pair_polylines(data);
    }
    let polyfaces = scan_polyfaces(data, &layer_names);
    let probable_mesh_entities = count_probable_mesh_entities(data);
    // The coordinate-pair fallback can misread a mesh's vertex stream as one
    // long polyline; a lone fallback polyline alongside meshes is suspect.
    // Polylines from genuine class records are always kept.
    if probable_mesh_entities > 0 && polylines_from_fallback && polylines.len() == 1 {
        polylines.clear();
    }
    // Fallback polylines don't correspond to design-class records, so they
    // can't be subtracted from the class-record count.
    let class_record_polylines = if polylines_from_fallback {
        0
    } else {
        polylines.len()
    };
    DufData {
        skipped: DufSkipped {
            unsupported_design_entities: count_probable_design_entities(data)
                .saturating_sub(points.len() + class_record_polylines),
            unsupported_mesh_entities: probable_mesh_entities.saturating_sub(polyfaces.len()),
        },
        layers,
        points,
        polylines,
        polyfaces,
    }
}

fn scan_layers(data: &[u8]) -> Vec<DufLayer> {
    let mut names = Vec::new();
    let mut seen = HashSet::new();
    for string in scan_string_property(data, PROP_LAYER_NAME) {
        let name = clean_layer_name(&string);
        if is_useful_layer_name(&name) && seen.insert(name.clone()) {
            names.push(DufLayer { name });
        }
    }
    if names.is_empty() {
        names.push(DufLayer {
            name: "DUF Import".to_owned(),
        });
    }
    names
}

/// Heuristic scan for `property` string records. The length field is read as
/// a single byte, so strings longer than 255 bytes are never matched (their
/// encoding, if the format supports them at all, is unknown).
fn scan_string_property(data: &[u8], property: u32) -> Vec<String> {
    let mut out = Vec::new();
    let pattern = property.to_le_bytes();
    let mut i = 0usize;
    while i + 6 <= data.len() {
        if data[i] == 0x15 && data[i + 1..i + 5] == pattern {
            let len = data[i + 5] as usize;
            let start = i + 6;
            let end = start + len;
            if end <= data.len()
                && let Ok(value) = std::str::from_utf8(&data[start..end])
            {
                out.push(value.to_owned());
                i = end;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn clean_layer_name(name: &str) -> String {
    let name = name.trim();
    name.rsplit('\\')
        .next()
        .unwrap_or(name)
        .trim()
        .trim_matches('\0')
        .to_owned()
}

fn is_useful_layer_name(name: &str) -> bool {
    !name.is_empty()
        && name != "_DW_SETTINGS"
        && name != "_DW_HIDDEN_GRAPHICS"
        && !name.starts_with("_dw_")
}

/// Map the GUID of every named table entry (class 0x529: layers, linetypes,
/// text styles, ...) to its 0x594 name. Entities reference their layer by one
/// of these GUIDs, so resolving through this map recovers the Deswik layer
/// path for each entity.
fn layer_names_by_guid(data: &[u8]) -> HashMap<[u8; 16], String> {
    let mut map = HashMap::new();
    for record in object_records(data) {
        if record.class_id != TABLE_ENTRY_CLASS_ID {
            continue;
        }
        let Some(guid) = guid_property(record.bytes, 0x19, PROP_OBJECT_GUID) else {
            continue;
        };
        let Some(name) = scan_string_property(record.bytes, PROP_LAYER_NAME)
            .into_iter()
            .next()
        else {
            continue;
        };
        let name = name.trim().trim_matches('\0').to_owned();
        if !name.is_empty() {
            map.insert(guid, name);
        }
    }
    map
}

/// Find property `prop` encoded as `type_byte` + property id + 16 GUID bytes.
fn guid_property(record: &[u8], type_byte: u8, prop: u32) -> Option<[u8; 16]> {
    let mut tag = [0u8; 5];
    tag[0] = type_byte;
    tag[1..5].copy_from_slice(&prop.to_le_bytes());
    let offset = find_bytes(record, &tag)?;
    let start = offset + tag.len();
    record.get(start..start + 16)?.try_into().ok()
}

/// Resolve an entity record's layer reference (property 0x2a3, a GUID) to the
/// layer's name. Only the head of the record is scanned so geometry bytes
/// cannot be misread as the property tag.
fn entity_layer_name(record: &[u8], layer_names: &HashMap<[u8; 16], String>) -> Option<String> {
    let head = &record[..record.len().min(ENTITY_HEAD_SCAN_LIMIT)];
    let guid = guid_property(head, 0x03, PROP_ENTITY_LAYER_REF)?;
    layer_names.get(&guid).cloned()
}

fn scan_coordinate_pair_polylines(data: &[u8]) -> Vec<DufPolyline> {
    let records = object_records(data);
    let mut polylines = Vec::new();
    for record in records {
        let mut doubles = HashMap::new();
        for (prop, value) in scan_double_properties(record.bytes) {
            doubles.insert(prop, value);
        }
        let Some(x0) = doubles.get(&PROP_COORD_X0).copied() else {
            continue;
        };
        let Some(y0) = doubles.get(&PROP_COORD_Y0).copied() else {
            continue;
        };
        let Some(z0) = doubles.get(&PROP_COORD_Z0).copied() else {
            continue;
        };
        let Some(x1) = doubles.get(&PROP_COORD_X1).copied() else {
            continue;
        };
        let Some(y1) = doubles.get(&PROP_COORD_Y1).copied() else {
            continue;
        };
        let Some(z1) = doubles.get(&PROP_COORD_Z1).copied() else {
            continue;
        };
        let vertices = vec![DVec3::new(x0, y0, z0), DVec3::new(x1, y1, z1)];
        if !is_reasonable_design_vertices(&vertices) {
            continue;
        }
        polylines.push(DufPolyline {
            layer_name: "DUF Design".to_owned(),
            vertices,
        });
    }
    polylines
}

fn scan_polyline_records(data: &[u8], layer_names: &HashMap<[u8; 16], String>) -> Vec<DufPolyline> {
    object_records(data)
        .into_iter()
        .filter(|record| record.class_id == POLYLINE_CLASS_ID)
        .filter_map(|record| {
            let mut polyline = parse_polyline_record(record.bytes)?;
            if let Some(name) = entity_layer_name(record.bytes, layer_names) {
                polyline.layer_name = name;
            }
            Some(polyline)
        })
        .collect()
}

fn scan_point_records(data: &[u8], layer_names: &HashMap<[u8; 16], String>) -> Vec<DufPoint> {
    object_records(data)
        .into_iter()
        .filter(|record| record.class_id == POINT_CLASS_ID)
        .filter_map(|record| {
            let mut point = parse_point_record(record.bytes)?;
            if let Some(name) = entity_layer_name(record.bytes, layer_names) {
                point.layer_name = name;
            }
            Some(point)
        })
        .collect()
}

fn parse_point_record(bytes: &[u8]) -> Option<DufPoint> {
    scan_vector3_properties(bytes)
        .into_iter()
        .find_map(|(prop, position)| {
            (prop == PROP_POINT_POSITION && is_reasonable_design_vertices(&[position])).then_some(
                DufPoint {
                    layer_name: "DUF Design".to_owned(),
                    position,
                },
            )
        })
}

fn parse_polyline_record(bytes: &[u8]) -> Option<DufPolyline> {
    let marker = find_bytes(bytes, POLYLINE_VERTEX_COUNT_MARKER)?;
    let count_offset = marker + POLYLINE_VERTEX_COUNT_MARKER.len();
    let array_marker = count_offset + 4;
    if array_marker + POLYLINE_VERTEX_ARRAY_MARKER.len() > bytes.len()
        || &bytes[array_marker..array_marker + POLYLINE_VERTEX_ARRAY_MARKER.len()]
            != POLYLINE_VERTEX_ARRAY_MARKER
    {
        return None;
    }
    let vertex_count =
        u32::from_le_bytes(bytes[count_offset..count_offset + 4].try_into().ok()?) as usize;
    let vertex_start = array_marker + POLYLINE_VERTEX_ARRAY_MARKER.len();
    let vertex_bytes = vertex_count.checked_mul(24)?;
    let vertex_end = vertex_start.checked_add(vertex_bytes)?;
    if vertex_count < 2 || vertex_end > bytes.len() {
        return None;
    }

    let vertices: Vec<DVec3> = bytes[vertex_start..vertex_end]
        .chunks_exact(24)
        .map(|chunk| {
            let x = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let y = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
            let z = f64::from_le_bytes(chunk[16..24].try_into().unwrap());
            DVec3::new(x, y, z)
        })
        .collect();
    is_reasonable_design_vertices(&vertices).then_some(DufPolyline {
        layer_name: "DUF Design".to_owned(),
        vertices,
    })
}

fn is_reasonable_design_vertices(vertices: &[DVec3]) -> bool {
    vertices.iter().all(|vertex| {
        [vertex.x, vertex.y, vertex.z]
            .into_iter()
            .all(is_reasonable_coordinate)
    }) && design_span(vertices) <= MAX_REASONABLE_GEOMETRY_SPAN
}

fn design_span(vertices: &[DVec3]) -> f64 {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for vertex in vertices {
        min[0] = min[0].min(vertex.x);
        min[1] = min[1].min(vertex.y);
        min[2] = min[2].min(vertex.z);
        max[0] = max[0].max(vertex.x);
        max[1] = max[1].max(vertex.y);
        max[2] = max[2].max(vertex.z);
    }
    (0..3).map(|axis| max[axis] - min[axis]).fold(0.0, f64::max)
}

fn scan_double_properties(data: &[u8]) -> Vec<(u32, f64)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 13 <= data.len() {
        if data[i] == 0x0c {
            let prop = u32::from_le_bytes(data[i + 1..i + 5].try_into().unwrap());
            let value = f64::from_le_bytes(data[i + 5..i + 13].try_into().unwrap());
            if value.is_finite() {
                out.push((prop, value));
            }
            i += 13;
        } else {
            i += 1;
        }
    }
    out
}

fn scan_vector3_properties(data: &[u8]) -> Vec<(u32, DVec3)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 29 <= data.len() {
        if data[i] == 0x1b {
            let prop = u32::from_le_bytes(data[i + 1..i + 5].try_into().unwrap());
            let x = f64::from_le_bytes(data[i + 5..i + 13].try_into().unwrap());
            let y = f64::from_le_bytes(data[i + 13..i + 21].try_into().unwrap());
            let z = f64::from_le_bytes(data[i + 21..i + 29].try_into().unwrap());
            let position = DVec3::new(x, y, z);
            if [x, y, z].into_iter().all(is_reasonable_coordinate) {
                out.push((prop, position));
            }
            i += 29;
        } else {
            i += 1;
        }
    }
    out
}

#[derive(Clone, Copy)]
struct ObjectRecord<'a> {
    class_id: u32,
    bytes: &'a [u8],
}

fn object_records(data: &[u8]) -> Vec<ObjectRecord<'_>> {
    let mut starts = Vec::new();
    let pattern = b"\x02\x12\x00\x00\x00\x00";
    let mut i = 0usize;
    while let Some(relative) = find_bytes(&data[i..], pattern) {
        let start = i + relative;
        if start + 10 <= data.len() {
            let class_id = u32::from_le_bytes(data[start + 6..start + 10].try_into().unwrap());
            starts.push((start, class_id));
        }
        i = start + 1;
    }
    let mut records = Vec::new();
    for (index, (start, class_id)) in starts.iter().copied().enumerate() {
        let end = starts
            .get(index + 1)
            .map(|(next, _)| *next)
            .unwrap_or(data.len());
        if start < end {
            records.push(ObjectRecord {
                class_id,
                bytes: &data[start..end],
            });
        }
    }
    records
}

fn scan_polyfaces(data: &[u8], layer_names: &HashMap<[u8; 16], String>) -> Vec<DufPolyface> {
    let mut out = Vec::new();
    for record in object_records(data)
        .into_iter()
        .filter(|record| record.class_id == POLYFACE_CLASS_ID)
    {
        let Some(face_marker) = find_bytes(record.bytes, POLYFACE_FACE_COUNT_MARKER) else {
            continue;
        };
        let face_count_offset = face_marker + POLYFACE_FACE_COUNT_MARKER.len();
        if face_count_offset + 4 > record.bytes.len() {
            continue;
        }
        let face_int_count = u32::from_le_bytes(
            record.bytes[face_count_offset..face_count_offset + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        if face_int_count == 0 || !face_int_count.is_multiple_of(FACE_RECORD_INTS) {
            continue;
        }
        let face_start = face_count_offset + 5;
        let Some(face_end) = face_start.checked_add(face_int_count * 4) else {
            continue;
        };

        let Some(vertex_marker) = find_bytes(record.bytes, POLYFACE_VERTEX_COUNT_MARKER) else {
            continue;
        };
        if face_end > vertex_marker {
            continue;
        }

        let count_offset = vertex_marker + POLYFACE_VERTEX_COUNT_MARKER.len();
        if count_offset + 4 > record.bytes.len() {
            continue;
        }
        let vertex_count = u32::from_le_bytes(
            record.bytes[count_offset..count_offset + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let raw_vertices = parse_polyface_vertices(record.bytes, count_offset + 4, vertex_count);
        let face_records = scan_face_records_at(record.bytes, face_start, face_end, vertex_count);
        if face_records.is_empty() {
            continue;
        }
        let raw_faces = triangulate_face_records(&face_records);
        let Some((vertices, faces)) = compact_valid_mesh(raw_vertices, raw_faces) else {
            continue;
        };
        if !faces.is_empty() {
            out.push(DufPolyface {
                layer_name: entity_layer_name(record.bytes, layer_names)
                    .unwrap_or_else(|| "DUF Mesh".to_owned()),
                vertices,
                faces,
            });
        }
    }
    out
}

fn parse_polyface_vertices(
    record: &[u8],
    marker_offset: usize,
    vertex_count: usize,
) -> Vec<Option<tri00t::Vertex>> {
    if marker_offset + POLYFACE_VERTEX_ARRAY_MARKER.len() <= record.len()
        && &record[marker_offset..marker_offset + POLYFACE_VERTEX_ARRAY_MARKER.len()]
            == POLYFACE_VERTEX_ARRAY_MARKER
    {
        let vertex_start = marker_offset + POLYFACE_VERTEX_ARRAY_MARKER.len();
        let Some(vertex_bytes) = vertex_count
            .checked_mul(24)
            .and_then(|bytes| bytes.checked_add(4))
        else {
            return Vec::new();
        };
        let vertex_end = vertex_start + vertex_bytes;
        if vertex_count == 0 || vertex_end > record.len() {
            return Vec::new();
        }
        return parse_split_polyface_vertices(&record[vertex_start..vertex_end]);
    }

    if record.get(marker_offset).copied() == Some(POLYFACE_COMPACT_VERTEX_MARKER) {
        let vertex_start = marker_offset + 1;
        let Some(vertex_bytes) = vertex_count.checked_mul(24) else {
            return Vec::new();
        };
        let vertex_end = vertex_start + vertex_bytes;
        if vertex_count == 0 || vertex_end > record.len() {
            return Vec::new();
        }
        return parse_standard_vertices(&record[vertex_start..vertex_end]);
    }

    Vec::new()
}

fn parse_split_polyface_vertices(bytes: &[u8]) -> Vec<Option<tri00t::Vertex>> {
    if bytes.len() < 4 {
        return Vec::new();
    }
    let mut previous_x_high: Option<[u8; 4]> = bytes[0..4].try_into().ok();
    bytes[4..]
        .chunks_exact(24)
        .map(|chunk| {
            let x_high: [u8; 4] = previous_x_high?;
            let y = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let z = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
            let x_low: [u8; 4] = chunk[16..20].try_into().unwrap();
            let x = f64::from_le_bytes([
                x_low[0], x_low[1], x_low[2], x_low[3], x_high[0], x_high[1], x_high[2], x_high[3],
            ]);
            previous_x_high = Some(chunk[20..24].try_into().unwrap());
            ([x, y, z].into_iter().all(is_reasonable_coordinate)).then_some(tri00t::Vertex {
                x,
                y,
                z,
            })
        })
        .collect()
}

fn parse_standard_vertices(bytes: &[u8]) -> Vec<Option<tri00t::Vertex>> {
    bytes
        .chunks_exact(24)
        .map(|chunk| {
            let x = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let y = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
            let z = f64::from_le_bytes(chunk[16..24].try_into().unwrap());
            ([x, y, z].into_iter().all(is_reasonable_coordinate)).then_some(tri00t::Vertex {
                x,
                y,
                z,
            })
        })
        .collect()
}

fn is_reasonable_coordinate(value: f64) -> bool {
    value.is_finite() && value.abs() <= MAX_REASONABLE_COORDINATE_ABS
}

fn compact_valid_mesh(
    raw_vertices: Vec<Option<tri00t::Vertex>>,
    raw_faces: Vec<[u32; 3]>,
) -> Option<(Vec<tri00t::Vertex>, Vec<[u32; 3]>)> {
    let mut remap = vec![None; raw_vertices.len()];
    let mut vertices = Vec::new();
    for (old_index, vertex) in raw_vertices.into_iter().enumerate() {
        if let Some(vertex) = vertex {
            remap[old_index] = Some(vertices.len() as u32);
            vertices.push(vertex);
        }
    }
    let faces: Vec<[u32; 3]> = raw_faces
        .into_iter()
        .filter_map(|face| {
            let a = remap.get(face[0] as usize).copied().flatten()?;
            let b = remap.get(face[1] as usize).copied().flatten()?;
            let c = remap.get(face[2] as usize).copied().flatten()?;
            (a != b && b != c && c != a).then_some([a, b, c])
        })
        .collect();
    if vertices.is_empty()
        || faces.is_empty()
        || mesh_span(&vertices) > MAX_REASONABLE_GEOMETRY_SPAN
    {
        return None;
    }
    Some((vertices, faces))
}

fn mesh_span(vertices: &[tri00t::Vertex]) -> f64 {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for vertex in vertices {
        min[0] = min[0].min(vertex.x);
        min[1] = min[1].min(vertex.y);
        min[2] = min[2].min(vertex.z);
        max[0] = max[0].max(vertex.x);
        max[1] = max[1].max(vertex.y);
        max[2] = max[2].max(vertex.z);
    }
    (0..3).map(|axis| max[axis] - min[axis]).fold(0.0, f64::max)
}

fn scan_face_records_at(
    data: &[u8],
    start: usize,
    end: usize,
    vertex_count: usize,
) -> Vec<[i32; 5]> {
    let mut records = Vec::new();
    let mut cursor = start;
    while cursor + FACE_RECORD_BYTES <= end {
        let group = [
            read_i32(data, cursor),
            read_i32(data, cursor + 4),
            read_i32(data, cursor + 8),
            read_i32(data, cursor + 12),
            read_i32(data, cursor + 16),
        ];
        if !is_plausible_face_group(group, vertex_count) {
            break;
        }
        records.push(group);
        cursor += FACE_RECORD_BYTES;
    }
    records
}

fn read_i32(data: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn is_plausible_face_group(group: [i32; 5], vertex_count: usize) -> bool {
    let color_or_flag_ok = group[4] == -1 || group[4] >= 0;
    color_or_flag_ok
        && group[..4].iter().all(|value| {
            let abs = value.unsigned_abs() as usize;
            abs >= 1 && abs <= vertex_count
        })
}

fn triangulate_face_records(records: &[[i32; 5]]) -> Vec<[u32; 3]> {
    let mut out = Vec::new();
    for record in records {
        let idx = [
            record[0].unsigned_abs() - 1,
            record[1].unsigned_abs() - 1,
            record[2].unsigned_abs() - 1,
            record[3].unsigned_abs() - 1,
        ];
        if idx[0] == idx[1] || idx[1] == idx[2] || idx[2] == idx[0] {
            continue;
        }
        out.push([idx[0], idx[1], idx[2]]);
        if idx[3] != idx[0] && idx[3] != idx[1] && idx[3] != idx[2] {
            out.push([idx[2], idx[3], idx[0]]);
        }
    }
    out
}

fn count_probable_design_entities(data: &[u8]) -> usize {
    object_records(data)
        .into_iter()
        .filter(|record| record.class_id == POLYLINE_CLASS_ID || record.class_id == POINT_CLASS_ID)
        .count()
}

fn count_probable_mesh_entities(data: &[u8]) -> usize {
    object_records(data)
        .into_iter()
        .filter(|record| record.class_id == POLYFACE_CLASS_ID)
        .count()
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
