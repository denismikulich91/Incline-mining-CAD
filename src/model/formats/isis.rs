use std::{
    collections::HashMap,
    error::Error,
    fmt, fs, io,
    path::{Path, PathBuf},
};

use super::tri00t::{VULZ_MAGIC, decode_vulz_archive};

pub const DGD_COORD_RECORD_LEN: usize = 117;

// ── Public types ──────────────────────────────────────────────────────────────

/// A single coordinate record from a Vulcan `.dgd.isis` design database.
///
/// Records with `seg_type == 0` are segment-header points (start of a new
/// polyline/polygon). Records with `seg_type == 1` are continuation points.
/// Reconstruct individual polylines in file order, breaking on `seg_type == 0`;
/// the name field can be a generated point label and is not always a layer.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignPoint {
    /// Byte offset of this record in the decompressed ISIS stream.
    pub offset: usize,
    /// Layer / segment name (trimmed of trailing whitespace).
    pub name: String,
    /// Secondary record name/attribute field. Some DGD variants store the
    /// useful layer-like name here while the first field is a generated point
    /// label.
    pub secondary_name: String,
    /// Layer this point belongs to, resolved from the database's structural
    /// layer records.
    pub layer_name: Option<String>,
    /// 0 = segment header (first point of a new feature), 1 = continuation.
    pub seg_type: u8,
    pub geometry_kind: DesignGeometryKind,
    /// Whether the owning object (type-03 header) is a closed polygon. Vulcan
    /// does not repeat the first point, so closure can only come from this flag.
    pub closed: bool,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// A text object (type-04 TEXT or type-0a 3DTEXT) from a `.dgd.isis` database.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignText {
    /// Byte offset of the object header record in the decompressed stream.
    pub offset: usize,
    /// Layer resolved from the database's structural layer records.
    pub layer_name: Option<String>,
    /// Text content; multi-line records are joined with `\n`.
    pub content: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Character height in world units.
    pub height: f64,
    pub rotation_degrees: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DesignGeometryKind {
    Unknown,
    Line,
    Point,
}

/// A named entry from a Vulcan `.dgd.isix` design index sidecar.
///
/// The offset points into the decompressed `.dgd.isis` stream near the first
/// object associated with the named design layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DesignIndexEntry {
    pub offset: usize,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DgdDesignData {
    pub points: Vec<DesignPoint>,
    pub texts: Vec<DesignText>,
    pub layer_names: Vec<String>,
}

#[derive(Debug)]
pub enum IsisError {
    Io(io::Error),
    Decompress(String),
}

impl fmt::Display for IsisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IsisError::Io(e) => write!(f, "{e}"),
            IsisError::Decompress(msg) => write!(f, "vulZ decompression failed: {msg}"),
        }
    }
}

impl Error for IsisError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            IsisError::Io(e) => Some(e),
            _ => None,
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Read all design coordinate points from a `.dgd.isis` file (compressed or raw).
///
/// Use `seg_type == 0` records as segment boundaries in file order: a new
/// polyline starts whenever `seg_type` transitions back to 0.
#[allow(dead_code)]
pub fn read_dgd_points(path: impl AsRef<Path>) -> Result<Vec<DesignPoint>, IsisError> {
    Ok(read_dgd_design(path)?.points)
}

pub fn read_dgd_design(path: impl AsRef<Path>) -> Result<DgdDesignData, IsisError> {
    let bytes = fs::read(path).map_err(IsisError::Io)?;
    read_dgd_design_bytes(&bytes)
}

#[allow(dead_code)]
pub fn read_dgd_points_bytes(bytes: &[u8]) -> Result<Vec<DesignPoint>, IsisError> {
    Ok(read_dgd_design_bytes(bytes)?.points)
}

pub fn read_dgd_design_bytes(bytes: &[u8]) -> Result<DgdDesignData, IsisError> {
    let (data, aux) = decompress_if_vulz(bytes)?;
    let gallery = if aux.is_empty() { &data } else { &aux };
    let mut layer_names = scan_dgd_embedded_layer_names(gallery);
    let layer_headers = scan_dgd_layer_headers(&data);
    let saves = scan_dgd_layer_saves(&data);
    let objects = scan_dgd_objects(&data);
    let mut text_coord_offsets = std::collections::HashSet::new();
    let mut texts = extract_dgd_texts(&data, &objects, &mut text_coord_offsets);
    let mut points = scan_dgd_points(&data);
    points.retain(|point| !text_coord_offsets.contains(&point.offset));
    attribute_dgd_closed(&mut points, &objects);
    attribute_dgd_layers(&mut points, &mut texts, &layer_headers, &saves);
    for name in points
        .iter()
        .filter_map(|point| point.layer_name.as_deref())
        .chain(texts.iter().filter_map(|text| text.layer_name.as_deref()))
    {
        push_unique_layer_name(&mut layer_names, name);
    }
    Ok(DgdDesignData {
        points,
        texts,
        layer_names,
    })
}

pub fn read_dgd_index(path: impl AsRef<Path>) -> Result<Vec<DesignIndexEntry>, IsisError> {
    let bytes = fs::read(path).map_err(IsisError::Io)?;
    Ok(read_dgd_index_bytes(&bytes))
}

pub fn read_dgd_index_bytes(bytes: &[u8]) -> Vec<DesignIndexEntry> {
    scan_dgd_index(bytes)
}

pub(crate) fn same_stem_isix_path(isis_path: &Path) -> Option<PathBuf> {
    let mut path = isis_path.to_path_buf();
    path.set_extension("isix");
    if path.is_file() {
        return Some(path);
    }
    let expected = path.file_name()?.to_str()?;
    let parent = path.parent()?;
    fs::read_dir(parent)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|candidate| {
            candidate
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case(expected))
                && candidate.is_file()
        })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Returns `(logical database image, auxiliary preview stream)`. Journaled
/// databases legitimately have freed pages, so holes are accepted here; they
/// read back as zero bytes and simply contain no records.
fn decompress_if_vulz(bytes: &[u8]) -> Result<(Vec<u8>, Vec<u8>), IsisError> {
    if bytes.starts_with(VULZ_MAGIC) {
        let archive =
            decode_vulz_archive(bytes).map_err(|e| IsisError::Decompress(e.to_string()))?;
        Ok((archive.data, archive.aux))
    } else {
        Ok((bytes.to_vec(), Vec::new()))
    }
}

/// Scan a raw (decompressed) DGD ISIS stream for SEGCRD coordinate records.
///
/// Record layout (117 bytes each):
/// ```text
/// [0x05]  [0x20]  [seg: 3 ASCII chars, spaces and digits]
/// [X:f64be]  [Y:f64be]  [Z:f64be]
/// [8 bytes extra]  [name0: 40 bytes space-padded]  [name1: 40 bytes space-padded]
/// ```
/// The seg field's alignment varies between writer versions (`  0` and
/// `1  ` both occur); 0 marks a segment-header, anything else a continuation.
fn scan_dgd_points(data: &[u8]) -> Vec<DesignPoint> {
    const MIN_SCAN_OFFSET: usize = 0x1000;
    const COORD_OFFSET: usize = 5;
    const NAME_OFFSET: usize = 37;
    const SECONDARY_NAME_OFFSET: usize = 77;
    const NAME_LEN: usize = 40;

    let mut out = Vec::new();

    let limit = data.len().saturating_sub(DGD_COORD_RECORD_LEN);
    let mut i = MIN_SCAN_OFFSET;

    while i <= limit {
        if data[i] == 0x05
            && data[i + 1] == 0x20
            && let Some(seg_type) = parse_dgd_seg_field(&data[i + 2..i + 5])
            && let Some((x, y, z)) = try_read_xyz(data, i + COORD_OFFSET)
            && is_plausible_coord(x, y, z)
        {
            let name = decode_name(&data[i + NAME_OFFSET..i + NAME_OFFSET + NAME_LEN]);
            let secondary_name =
                decode_name(&data[i + SECONDARY_NAME_OFFSET..i + SECONDARY_NAME_OFFSET + NAME_LEN]);
            out.push(DesignPoint {
                offset: i,
                name,
                secondary_name,
                layer_name: None,
                seg_type,
                geometry_kind: infer_dgd_geometry_kind(data, i),
                closed: false,
                x,
                y,
                z,
            });
            i += DGD_COORD_RECORD_LEN;
            continue;
        }
        i += 1;
    }

    out
}

fn parse_dgd_seg_field(field: &[u8]) -> Option<u8> {
    if !field
        .iter()
        .all(|byte| *byte == b' ' || byte.is_ascii_digit())
        || !field.iter().any(u8::is_ascii_digit)
    {
        return None;
    }
    let digits: String = field
        .iter()
        .filter(|byte| byte.is_ascii_digit())
        .map(|&byte| byte as char)
        .collect();
    Some(digits.parse::<u32>().ok()?.min(u8::MAX as u32) as u8)
}

/// A layer header/open record (type 01): written before a layer's content and
/// carrying the active layer name for subsequent records.
struct DgdLayerHeader {
    offset: usize,
    flag: u8,
    name: String,
}

/// Scan for type-01 layer header records. These are the ownership records used
/// by DGD streams that include explicit layer-open metadata: byte 0 is `01`,
/// byte 1 is a status flag (space = live, `D` = deleted, `$` = temporary), then
/// a 40-byte name, a 40-byte timestamp/user field, and a mostly blank tail.
fn scan_dgd_layer_headers(data: &[u8]) -> Vec<DgdLayerHeader> {
    const NAME_LEN: usize = 40;
    const STAMP_LEN: usize = 40;
    const NAME_OFFSET: usize = 2;
    const STAMP_OFFSET: usize = NAME_OFFSET + NAME_LEN;
    const TAIL_OFFSET: usize = STAMP_OFFSET + STAMP_LEN;

    let mut headers = Vec::new();
    let mut i = 0;
    while i + DGD_COORD_RECORD_LEN <= data.len() {
        if data[i] != 0x01 || !matches!(data[i + 1], b' ' | b'D' | b'$') {
            i += 1;
            continue;
        }
        let Some(name) = decode_ascii_name(&data[i + NAME_OFFSET..i + NAME_OFFSET + NAME_LEN])
        else {
            i += 1;
            continue;
        };
        let Some(stamp) = decode_ascii_name(&data[i + STAMP_OFFSET..i + STAMP_OFFSET + STAMP_LEN])
        else {
            i += 1;
            continue;
        };
        if !is_dgd_layer_header_stamp(&stamp)
            || !data[i + TAIL_OFFSET..i + DGD_COORD_RECORD_LEN]
                .iter()
                .all(|byte| *byte == 0 || *byte == b' ')
        {
            i += 1;
            continue;
        }
        headers.push(DgdLayerHeader {
            offset: i,
            flag: data[i + 1],
            name,
        });
        i += DGD_COORD_RECORD_LEN;
    }
    headers
}

fn is_dgd_layer_header_stamp(stamp: &str) -> bool {
    let upper = stamp.to_ascii_uppercase();
    let has_month = [
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    ]
    .iter()
    .any(|month| upper.contains(month));
    stamp.contains(':') && (upper.contains("DGEDIT") || has_month)
}

fn is_live_dgd_layer_header(header: &DgdLayerHeader) -> bool {
    header.flag == b' '
        && is_dgd_meaningful_layer_name(&header.name)
        && !header.name.starts_with("DIG$")
}

fn is_dgd_layer_header_attribution_candidate(header: &DgdLayerHeader) -> bool {
    is_live_dgd_layer_header(header)
}

/// A layer save record (type 09): written after a layer's content with the
/// name the layer was saved under. This is used as a fallback for streams that
/// do not contain explicit type-01 layer headers.
struct DgdLayerSave {
    offset: usize,
    name: String,
    deleted: bool,
}

/// Scan for type-09 layer save records: `09`, a flag byte (space = live,
/// `D` = deleted, `$` = unnamed temporary), a 40-byte space-padded name, then
/// spaces and digit fields for the rest of the 117-byte record.
fn scan_dgd_layer_saves(data: &[u8]) -> Vec<DgdLayerSave> {
    const NAME_LEN: usize = 40;

    let mut saves = Vec::new();
    let mut i = 0;
    while i + DGD_COORD_RECORD_LEN <= data.len() {
        if data[i] != 0x09 || !matches!(data[i + 1], b' ' | b'D' | b'$') {
            i += 1;
            continue;
        }
        let Some(name) = decode_ascii_name(&data[i + 2..i + 2 + NAME_LEN]) else {
            i += 1;
            continue;
        };
        let rest = &data[i + 2 + NAME_LEN..i + DGD_COORD_RECORD_LEN];
        if !rest
            .iter()
            .all(|byte| *byte == b' ' || byte.is_ascii_digit())
        {
            i += 1;
            continue;
        }
        saves.push(DgdLayerSave {
            offset: i,
            name,
            deleted: data[i + 1] == b'D',
        });
        i += DGD_COORD_RECORD_LEN;
    }
    saves
}

/// How a record offset relates to DGD layer ownership metadata.
enum LayerResolution<'a> {
    /// No authoritative layer was found; keep the record and let the caller's
    /// fallbacks name it.
    Unattributed,
    /// The record sits in a stale/deleted/system block Vulcan does not show.
    Dropped,
    /// The record belongs to this live layer.
    Live(&'a str),
}

/// Resolves record offsets against preceding type-01 layer headers. A live
/// layer header owns the following object records until the next layer header;
/// deleted, temporary, and system headers describe content Vulcan should not
/// display as user design layers.
struct DgdLayerHeaderResolver<'a> {
    headers: &'a [DgdLayerHeader],
}

impl<'a> DgdLayerHeaderResolver<'a> {
    fn new(headers: &'a [DgdLayerHeader]) -> Self {
        Self { headers }
    }

    fn resolve(&self, offset: usize) -> LayerResolution<'a> {
        let index = self
            .headers
            .partition_point(|header| header.offset < offset);
        let Some(header) = index
            .checked_sub(1)
            .and_then(|index| self.headers.get(index))
        else {
            return LayerResolution::Dropped;
        };
        if is_live_dgd_layer_header(header) {
            LayerResolution::Live(&header.name)
        } else {
            LayerResolution::Dropped
        }
    }
}

/// Resolves record offsets against the type-09 save records. A layer edited
/// multiple times leaves one content block per save; only the block ending at
/// the layer's *last* save reflects what Vulcan shows, so earlier blocks are
/// dropped, as are deleted layers and `DIG$*` system tables (colour maps and
/// the like, whose payload can look like coordinate records).
struct DgdSaveResolver<'a> {
    saves: &'a [DgdLayerSave],
    last_save: HashMap<&'a str, usize>,
}

impl<'a> DgdSaveResolver<'a> {
    fn new(saves: &'a [DgdLayerSave]) -> Self {
        let mut last_save = HashMap::new();
        for save in saves {
            last_save.insert(save.name.as_str(), save.offset);
        }
        Self { saves, last_save }
    }

    fn resolve(&self, offset: usize) -> LayerResolution<'a> {
        let index = self.saves.partition_point(|save| save.offset < offset);
        let Some(save) = self.saves.get(index) else {
            return LayerResolution::Unattributed;
        };
        if save.deleted
            || save.name.starts_with("DIG$")
            || self.last_save[save.name.as_str()] != save.offset
        {
            LayerResolution::Dropped
        } else {
            LayerResolution::Live(&save.name)
        }
    }
}

/// Attribute points and texts to DGD layer records, dropping records from
/// deleted, temporary, stale, or system blocks.
fn attribute_dgd_layers(
    points: &mut Vec<DesignPoint>,
    texts: &mut Vec<DesignText>,
    headers: &[DgdLayerHeader],
    saves: &[DgdLayerSave],
) {
    if headers
        .iter()
        .any(is_dgd_layer_header_attribution_candidate)
    {
        let resolver = DgdLayerHeaderResolver::new(headers);
        points.retain_mut(|point| match resolver.resolve(point.offset) {
            LayerResolution::Unattributed => true,
            LayerResolution::Dropped => false,
            LayerResolution::Live(name) => {
                point.layer_name = Some(name.to_owned());
                true
            }
        });
        texts.retain_mut(|text| match resolver.resolve(text.offset) {
            LayerResolution::Unattributed => true,
            LayerResolution::Dropped => false,
            LayerResolution::Live(name) => {
                text.layer_name = Some(name.to_owned());
                true
            }
        });
        return;
    }

    let resolver = DgdSaveResolver::new(saves);
    points.retain_mut(|point| match resolver.resolve(point.offset) {
        LayerResolution::Unattributed => true,
        LayerResolution::Dropped => false,
        LayerResolution::Live(name) => {
            point.layer_name = Some(name.to_owned());
            true
        }
    });
    texts.retain_mut(|text| match resolver.resolve(text.offset) {
        LayerResolution::Unattributed => true,
        LayerResolution::Dropped => false,
        LayerResolution::Live(name) => {
            text.layer_name = Some(name.to_owned());
            true
        }
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DgdObjectKind {
    Poly,
    Text,
    Text3d,
}

/// An object header record: type 03 (line/polygon), 04 (TEXT) or 0a (3DTEXT).
struct DgdObjectHeader {
    offset: usize,
    kind: DgdObjectKind,
    /// Digit field at byte 76 of type-03 headers: `1` marks a closed polygon.
    closed: bool,
}

/// Scan for object header records: a type byte, a flag byte, a 40-byte
/// space-padded name ("POLY", "3DTEXT", a reactive-text label, ...), eight
/// zero bytes, then space/digit attribute fields for the rest of the record.
fn scan_dgd_objects(data: &[u8]) -> Vec<DgdObjectHeader> {
    const NAME_LEN: usize = 40;
    const ZEROS_LEN: usize = 8;
    const CLOSED_FLAG_OFFSET: usize = 76;

    let mut out = Vec::new();
    let mut i = 0;
    while i + DGD_COORD_RECORD_LEN <= data.len() {
        let kind = match data[i] {
            0x03 => DgdObjectKind::Poly,
            0x04 => DgdObjectKind::Text,
            0x0a => DgdObjectKind::Text3d,
            _ => {
                i += 1;
                continue;
            }
        };
        let zeros = &data[i + 2 + NAME_LEN..i + 2 + NAME_LEN + ZEROS_LEN];
        let rest = &data[i + 2 + NAME_LEN + ZEROS_LEN..i + DGD_COORD_RECORD_LEN];
        if !matches!(data[i + 1], b' ' | b'D' | b'$')
            || decode_ascii_name(&data[i + 2..i + 2 + NAME_LEN]).is_none()
            || !zeros.iter().all(|byte| *byte == 0)
            || !rest
                .iter()
                .all(|byte| *byte == b' ' || byte.is_ascii_digit())
        {
            i += 1;
            continue;
        }
        out.push(DgdObjectHeader {
            offset: i,
            kind,
            closed: data[i + CLOSED_FLAG_OFFSET] == b'1',
        });
        i += DGD_COORD_RECORD_LEN;
    }
    out
}

/// Mark points of closed polygon objects. Vulcan never repeats the first
/// point of a closed shape, so this flag is the only closure signal.
fn attribute_dgd_closed(points: &mut [DesignPoint], objects: &[DgdObjectHeader]) {
    for point in points {
        let index = objects.partition_point(|object| object.offset < point.offset);
        if let Some(object) = index.checked_sub(1).and_then(|index| objects.get(index))
            && object.kind == DgdObjectKind::Poly
        {
            point.closed = object.closed;
        }
    }
}

/// Extract text objects. A TEXT object header is followed by a description
/// record, an origin coordinate, a parameter coordinate `(height, size
/// factor, rotation degrees)` and one type-06 record per text line. A 3DTEXT
/// object instead carries five coordinates (origin, x direction, y direction,
/// character size, spacing) and its first type-06 record is the font name.
///
/// The offsets of the coordinate records consumed here are recorded in
/// `text_coord_offsets` so the point scanner's output can be pruned — a text
/// origin is not a design point.
fn extract_dgd_texts(
    data: &[u8],
    objects: &[DgdObjectHeader],
    text_coord_offsets: &mut std::collections::HashSet<usize>,
) -> Vec<DesignText> {
    const CONTENT_LEN: usize = 80;
    const MAX_OBJECT_RECORDS: usize = 512;

    let mut out = Vec::new();
    for object in objects {
        if object.kind == DgdObjectKind::Poly {
            continue;
        }
        let mut coords: Vec<(usize, f64, f64, f64)> = Vec::new();
        let mut lines: Vec<String> = Vec::new();
        let mut i = object.offset + DGD_COORD_RECORD_LEN;
        for _ in 0..MAX_OBJECT_RECORDS {
            if i + DGD_COORD_RECORD_LEN > data.len() {
                break;
            }
            match data[i] {
                0x02 | 0x07 => {}
                0x05 => {
                    let Some((x, y, z)) = try_read_xyz(data, i + 5) else {
                        break;
                    };
                    coords.push((i, x, y, z));
                }
                0x06 => {
                    lines.push(decode_name(&data[i + 2..i + 2 + CONTENT_LEN]));
                }
                _ => break,
            }
            i += DGD_COORD_RECORD_LEN;
        }
        let parsed = match object.kind {
            DgdObjectKind::Text => parse_dgd_text(&coords, &lines),
            DgdObjectKind::Text3d => parse_dgd_text3d(&coords, &lines),
            DgdObjectKind::Poly => unreachable!(),
        };
        let Some((origin, height, rotation_degrees, content)) = parsed else {
            continue;
        };
        text_coord_offsets.extend(coords.iter().map(|(offset, ..)| *offset));
        out.push(DesignText {
            offset: object.offset,
            layer_name: None,
            content,
            x: origin.0,
            y: origin.1,
            z: origin.2,
            height,
            rotation_degrees,
        });
    }
    out
}

type DgdParsedText = ((f64, f64, f64), f64, f64, String);

fn parse_dgd_text(coords: &[(usize, f64, f64, f64)], lines: &[String]) -> Option<DgdParsedText> {
    let &(_, x, y, z) = coords.first()?;
    let &(_, height, _, rotation_degrees) = coords.get(1)?;
    let content = join_dgd_text_lines(lines)?;
    Some(((x, y, z), height, rotation_degrees, content))
}

fn parse_dgd_text3d(coords: &[(usize, f64, f64, f64)], lines: &[String]) -> Option<DgdParsedText> {
    let &(_, x, y, z) = coords.first()?;
    let &(_, dir_x, dir_y, _) = coords.get(1)?;
    let &(_, _, height, _) = coords.get(3)?;
    // The first type-06 record of a 3DTEXT is the font name, not content.
    let content = join_dgd_text_lines(lines.get(1..)?)?;
    Some(((x, y, z), height, dir_y.atan2(dir_x).to_degrees(), content))
}

fn join_dgd_text_lines(lines: &[String]) -> Option<String> {
    let content = lines.join("\n").trim_end().to_owned();
    (!content.is_empty()).then_some(content)
}

fn infer_dgd_geometry_kind(data: &[u8], coord_offset: usize) -> DesignGeometryKind {
    const LOOKBACK: usize = 2048;
    let start = coord_offset.saturating_sub(LOOKBACK);
    let window = &data[start..coord_offset];
    [
        (b"POLYPOINT".as_slice(), DesignGeometryKind::Point),
        (b"POLYLINE".as_slice(), DesignGeometryKind::Line),
        (b"LINE".as_slice(), DesignGeometryKind::Line),
    ]
    .into_iter()
    .filter_map(|(token, kind)| {
        window
            .windows(token.len())
            .rposition(|bytes| bytes == token)
            .map(|pos| (pos, kind))
    })
    .max_by_key(|(pos, _)| *pos)
    .map(|(_, kind)| kind)
    .unwrap_or(DesignGeometryKind::Unknown)
}

fn scan_dgd_embedded_layer_names(data: &[u8]) -> Vec<String> {
    const PNG_SIGNATURE: &[u8] = &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    const PNG_IEND: &[u8] = &[b'I', b'E', b'N', b'D', 0xae, b'B', b'`', 0x82];
    const NAME_LEN: usize = 40;
    const MAX_GAP_TO_NEXT_PNG: usize = 160;

    let mut names = Vec::new();

    // The first gallery entry has no preceding PNG: a 16-byte entry header,
    // then the same name-then-PNG layout as every following entry.
    const FIRST_ENTRY_NAME_OFFSET: usize = 16;
    if data.len() > FIRST_ENTRY_NAME_OFFSET + NAME_LEN
        && let Some(png_at) = find_bytes(
            &data[FIRST_ENTRY_NAME_OFFSET
                ..(FIRST_ENTRY_NAME_OFFSET + MAX_GAP_TO_NEXT_PNG).min(data.len())],
            PNG_SIGNATURE,
        )
        && png_at >= NAME_LEN
        && let Some(name) =
            decode_ascii_name(&data[FIRST_ENTRY_NAME_OFFSET..FIRST_ENTRY_NAME_OFFSET + NAME_LEN])
        && is_dgd_meaningful_layer_name(&name)
    {
        push_unique_layer_name(&mut names, &name);
    }

    let mut offset = 0;
    while let Some(relative) = find_bytes(&data[offset..], PNG_IEND) {
        let name_offset = offset + relative + PNG_IEND.len();
        let next_png_limit = (name_offset + MAX_GAP_TO_NEXT_PNG).min(data.len());
        let Some(next_png_relative) = find_bytes(&data[name_offset..next_png_limit], PNG_SIGNATURE)
        else {
            offset = name_offset;
            continue;
        };
        if next_png_relative >= NAME_LEN
            && let Some(name) = decode_ascii_name(&data[name_offset..name_offset + NAME_LEN])
            && is_dgd_meaningful_layer_name(&name)
        {
            push_unique_layer_name(&mut names, &name);
        }
        offset = name_offset;
    }
    names
}

pub(crate) fn is_dgd_meaningful_layer_name(name: &str) -> bool {
    let name = name.trim();
    is_dgd_index_layer_name(name)
        && !is_generated_dgd_point_name(name)
        && !is_scale_label(name)
        && !is_deleted_dgd_layer_name(name)
        && name.bytes().any(|byte| byte.is_ascii_alphabetic())
}

pub(crate) fn is_dgd_index_layer_name(name: &str) -> bool {
    let name = name.trim();
    !name.is_empty()
        && !is_dgd_system_layer_name(name)
        && !is_dgd_object_descriptor(name)
        && !name.bytes().any(|byte| byte == b'?')
        && name.bytes().all(|byte| (0x20..0x7f).contains(&byte))
}

fn is_dgd_system_layer_name(name: &str) -> bool {
    name.starts_with('$') || name.to_ascii_uppercase().starts_with("DIG$")
}

fn is_generated_dgd_point_name(name: &str) -> bool {
    name.strip_prefix("POINT_").is_some_and(|suffix| {
        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
    })
}

fn is_scale_label(name: &str) -> bool {
    let Some((left, right)) = name.split_once(':') else {
        return false;
    };
    !left.is_empty()
        && !right.is_empty()
        && left.bytes().all(|byte| byte.is_ascii_digit())
        && right.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_deleted_dgd_layer_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('D') else {
        return false;
    };
    rest.bytes()
        .next()
        .is_some_and(|byte| byte.is_ascii_digit())
        && rest.contains("_20")
}

fn is_dgd_object_descriptor(name: &str) -> bool {
    matches!(
        name.to_ascii_uppercase().as_str(),
        "LINE" | "POLY" | "POLYLINE" | "POLYPOINT" | "TEXT" | "TXT_3D" | "TXT_NEW"
    ) || name.eq_ignore_ascii_case("Imported from AutoCAD")
}

fn push_unique_layer_name(names: &mut Vec<String>, name: &str) {
    if !names.iter().any(|existing| existing == name) {
        names.push(name.to_owned());
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

const DGD_INDEX_START: usize = 0x400;
const DGD_INDEX_PAGE_LEN: usize = 0x400;
const DGD_INDEX_ENTRY_LEN: usize = 48;
const DGD_INDEX_NAME_OFFSET: usize = 8;
const DGD_INDEX_NAME_LEN: usize = 40;
const DGD_INDEX_MARKER: [u8; 4] = [0xff; 4];

/// Scan a DGD `.isix` sidecar for fixed-size layer index entries.
///
/// Vulcan stores a padded header first, then page-sized index tables of
/// 48-byte entries: `[offset:u32be] [0xffffffff] [name:40 bytes space-padded]`.
/// The current design-layer table starts at a page boundary and begins with a
/// user layer. Older/history and object tables can appear later in the same
/// sidecar, so they are only used by the fallback bytewise scan.
fn scan_dgd_index(data: &[u8]) -> Vec<DesignIndexEntry> {
    let current_page = scan_dgd_index_current_page(data);
    if !current_page.is_empty() {
        return current_page;
    }
    scan_dgd_index_unaligned(data)
}

fn scan_dgd_index_current_page(data: &[u8]) -> Vec<DesignIndexEntry> {
    let mut page_start = DGD_INDEX_START;
    while page_start + DGD_INDEX_ENTRY_LEN <= data.len() {
        if decode_dgd_index_entry(data, page_start).is_some() {
            let page_end = (page_start + DGD_INDEX_PAGE_LEN).min(data.len());
            let mut entries = Vec::new();
            let mut offset = page_start;
            while offset + DGD_INDEX_ENTRY_LEN <= page_end {
                if data[offset + 4..offset + 8] != DGD_INDEX_MARKER {
                    break;
                }
                if let Some(entry) = decode_dgd_index_entry(data, offset) {
                    entries.push(entry);
                }
                offset += DGD_INDEX_ENTRY_LEN;
            }
            return dedupe_dgd_index_entries(entries);
        }
        page_start += DGD_INDEX_PAGE_LEN;
    }
    Vec::new()
}

fn scan_dgd_index_unaligned(data: &[u8]) -> Vec<DesignIndexEntry> {
    let mut entries = Vec::new();
    let mut offset = DGD_INDEX_START.min(data.len());
    while offset + DGD_INDEX_ENTRY_LEN <= data.len() {
        if let Some(entry) = decode_dgd_index_entry(data, offset) {
            entries.push(entry);
            offset += DGD_INDEX_ENTRY_LEN;
            continue;
        }
        offset += 1;
    }
    entries.sort_by(|a, b| a.offset.cmp(&b.offset).then_with(|| a.name.cmp(&b.name)));
    entries.dedup();
    entries
}

fn is_meaningful_index_name(name: &str) -> bool {
    is_dgd_index_layer_name(name)
}

fn decode_dgd_index_entry(data: &[u8], offset: usize) -> Option<DesignIndexEntry> {
    if offset + DGD_INDEX_ENTRY_LEN > data.len() || data[offset + 4..offset + 8] != DGD_INDEX_MARKER
    {
        return None;
    }
    let name = decode_name(
        &data[offset + DGD_INDEX_NAME_OFFSET..offset + DGD_INDEX_NAME_OFFSET + DGD_INDEX_NAME_LEN],
    );
    if !is_meaningful_index_name(&name) {
        return None;
    }
    let pointer = u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    Some(DesignIndexEntry {
        offset: pointer,
        name,
    })
}

fn dedupe_dgd_index_entries(entries: Vec<DesignIndexEntry>) -> Vec<DesignIndexEntry> {
    let mut deduped = Vec::new();
    for entry in entries {
        if !deduped.iter().any(|existing| existing == &entry) {
            deduped.push(entry);
        }
    }
    deduped
}

fn decode_ascii_name(bytes: &[u8]) -> Option<String> {
    let mut out = String::new();
    for &byte in bytes.iter().take_while(|&&byte| byte != 0) {
        if !(0x20..0x7f).contains(&byte) {
            return None;
        }
        out.push(byte as char);
    }
    let name = out.trim().to_string();
    (!name.is_empty()).then_some(name)
}

fn try_read_xyz(data: &[u8], offset: usize) -> Option<(f64, f64, f64)> {
    if offset + 24 > data.len() {
        return None;
    }
    let x = f64::from_be_bytes(data[offset..offset + 8].try_into().ok()?);
    let y = f64::from_be_bytes(data[offset + 8..offset + 16].try_into().ok()?);
    let z = f64::from_be_bytes(data[offset + 16..offset + 24].try_into().ok()?);
    if x.is_finite() && y.is_finite() && z.is_finite() {
        Some((x, y, z))
    } else {
        None
    }
}

fn is_plausible_coord(x: f64, y: f64, z: f64) -> bool {
    x.abs() > 100.0 && x.abs() < 1e8 && y.abs() > 100.0 && y.abs() < 1e8 && z.abs() < 50_000.0
}

fn decode_name(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| {
            if (0x20..0x7f).contains(&b) {
                b as char
            } else {
                '?'
            }
        })
        .collect::<String>()
        .trim_end()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gated on `INCLINE_TEST_ISIS=<path>` — run against real Vulcan files that
    /// can't live in the repo. Prints record/segment stats for manual review.
    #[test]
    fn reads_dgd_isis_from_env_path() {
        let Ok(path) = std::env::var("INCLINE_TEST_ISIS") else {
            return;
        };
        if let Ok(dump_path) = std::env::var("INCLINE_TEST_ISIS_DUMP") {
            let bytes = fs::read(&path).unwrap();
            let (data, aux) = decompress_if_vulz(&bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
            fs::write(&dump_path, &data).unwrap();
            fs::write(format!("{dump_path}.aux"), &aux).unwrap();
            println!(
                "dumped {} logical + {} aux bytes to {dump_path}",
                data.len(),
                aux.len()
            );
        }
        let design = read_dgd_design(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let segments = design.points.iter().filter(|p| p.seg_type == 0).count();
        let mut names: Vec<&str> = design.points.iter().map(|p| p.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        println!(
            "{path}: {} points, {segments} segments, {} texts, {} distinct names, {} embedded layer names",
            design.points.len(),
            design.texts.len(),
            names.len(),
            design.layer_names.len()
        );
        for text in design.texts.iter().take(10) {
            println!(
                "  text {:?} at ({:.1},{:.1},{:.1}) h={} rot={:.1} layer={:?}",
                text.content,
                text.x,
                text.y,
                text.z,
                text.height,
                text.rotation_degrees,
                text.layer_name
            );
        }
        let closed = design.points.iter().filter(|p| p.closed).count();
        println!(
            "  {closed} of {} points belong to closed polygons",
            design.points.len()
        );
        for name in names.iter().take(20) {
            println!("  name: {name:?}");
        }
        for name in design.layer_names.iter().take(20) {
            println!("  layer: {name:?}");
        }
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for point in &design.points {
            *counts
                .entry(point.layer_name.as_deref().unwrap_or("<unattributed>"))
                .or_default() += 1;
        }
        let mut counts: Vec<_> = counts.into_iter().collect();
        counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (name, count) in counts {
            println!("  layer {name:?}: {count} points");
        }
        if let Ok(expected) = std::env::var("INCLINE_TEST_ISIS_LAYER_NAME") {
            assert!(
                design.layer_names.iter().any(|name| name == &expected)
                    || design
                        .points
                        .iter()
                        .any(|point| point.layer_name.as_deref() == Some(expected.as_str())),
                "{expected:?} was not found in {path}"
            );
        }
        assert!(
            !design.points.is_empty(),
            "no design points found in {path}"
        );
    }

    #[test]
    fn reads_embedded_dgd_layer_names_between_png_previews() {
        const PNG_SIGNATURE: &[u8] = &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        const PNG_IEND: &[u8] = &[b'I', b'E', b'N', b'D', 0xae, b'B', b'`', 0x82];

        let mut bytes = Vec::new();
        bytes.extend_from_slice(PNG_IEND);
        let mut slot = [0_u8; 56];
        let name = b"30P202_0514_20250319";
        slot[..name.len()].copy_from_slice(name);
        bytes.extend_from_slice(&slot);
        bytes.extend_from_slice(PNG_SIGNATURE);

        let design = read_dgd_design_bytes(&bytes).unwrap();

        assert_eq!(design.layer_names, vec!["30P202_0514_20250319"]);
    }

    fn write_coord_record(bytes: &mut [u8], offset: usize, seg: &[u8; 3], name: &str) {
        write_coord_record_xyz(bytes, offset, seg, name, (1234.0, 5678.0, 1000.0));
    }

    fn write_coord_record_xyz(
        bytes: &mut [u8],
        offset: usize,
        seg: &[u8; 3],
        name: &str,
        (x, y, z): (f64, f64, f64),
    ) {
        bytes[offset] = 0x05;
        bytes[offset + 1] = b' ';
        bytes[offset + 2..offset + 5].copy_from_slice(seg);
        bytes[offset + 5..offset + 13].copy_from_slice(&x.to_be_bytes());
        bytes[offset + 13..offset + 21].copy_from_slice(&y.to_be_bytes());
        bytes[offset + 21..offset + 29].copy_from_slice(&z.to_be_bytes());
        write_fixed_name(&mut bytes[offset + 37..offset + 77], name);
        bytes[offset + 77..offset + 117].fill(b' ');
    }

    fn write_object_record(bytes: &mut [u8], offset: usize, kind: u8, name: &str, closed: bool) {
        bytes[offset] = kind;
        bytes[offset + 1] = b' ';
        write_fixed_name(&mut bytes[offset + 2..offset + 42], name);
        // eight zero bytes at 42..50 are already present in a fresh buffer
        bytes[offset + 50..offset + DGD_COORD_RECORD_LEN].fill(b' ');
        bytes[offset + 76] = if closed { b'1' } else { b'0' };
    }

    fn write_text_line_record(bytes: &mut [u8], offset: usize, content: &str) {
        bytes[offset] = 0x06;
        bytes[offset + 1] = b' ';
        bytes[offset + 2..offset + DGD_COORD_RECORD_LEN].fill(b' ');
        bytes[offset + 2..offset + 2 + content.len()].copy_from_slice(content.as_bytes());
    }

    fn write_save_record(bytes: &mut [u8], offset: usize, flag: u8, name: &str) {
        bytes[offset] = 0x09;
        bytes[offset + 1] = flag;
        write_fixed_name(&mut bytes[offset + 2..offset + 42], name);
        bytes[offset + 42..offset + DGD_COORD_RECORD_LEN].fill(b' ');
    }

    fn write_layer_header_record(bytes: &mut [u8], offset: usize, flag: u8, name: &str) {
        bytes[offset] = 0x01;
        bytes[offset + 1] = flag;
        write_fixed_name(&mut bytes[offset + 2..offset + 42], name);
        write_fixed_name(
            &mut bytes[offset + 42..offset + 82],
            "1119Jul202317:41:24DGEDIT",
        );
        bytes[offset + 82..offset + DGD_COORD_RECORD_LEN].fill(b' ');
    }

    #[test]
    fn attributes_points_to_the_following_layer_save_record() {
        let coord_offset = 0x1000;
        let save_offset = coord_offset + DGD_COORD_RECORD_LEN;
        let mut bytes = vec![0_u8; save_offset + DGD_COORD_RECORD_LEN];

        write_coord_record(&mut bytes, coord_offset, b"  0", "POINT_1");
        write_save_record(&mut bytes, save_offset, b' ', "30P202_0526_20250414");

        let points = read_dgd_points_bytes(&bytes).unwrap();

        assert_eq!(points.len(), 1);
        assert_eq!(
            points[0].layer_name.as_deref(),
            Some("30P202_0526_20250414")
        );
    }

    #[test]
    fn layer_header_owns_points_even_when_following_save_has_stale_name() {
        let rec = DGD_COORD_RECORD_LEN;
        let base = 0x1000;
        let mut bytes = vec![0_u8; base + 3 * rec];
        write_layer_header_record(&mut bytes, base, b' ', "ACTIVE_LAYER");
        write_coord_record(&mut bytes, base + rec, b"  0", "POINT_1");
        write_save_record(&mut bytes, base + 2 * rec, b' ', "STALE_SAVE_NAME");

        let points = read_dgd_points_bytes(&bytes).unwrap();

        assert_eq!(points.len(), 1);
        assert_eq!(points[0].layer_name.as_deref(), Some("ACTIVE_LAYER"));
    }

    #[test]
    fn layer_headers_drop_temporary_deleted_and_system_blocks() {
        let rec = DGD_COORD_RECORD_LEN;
        let base = 0x1000;
        let mut bytes = vec![0_u8; base + 8 * rec];
        write_layer_header_record(&mut bytes, base, b' ', "ACTIVE_LAYER");
        write_coord_record(&mut bytes, base + rec, b"  0", "POINT_1");
        write_layer_header_record(&mut bytes, base + 2 * rec, b'$', "$12345");
        write_coord_record(&mut bytes, base + 3 * rec, b"  0", "POINT_2");
        write_layer_header_record(&mut bytes, base + 4 * rec, b'D', "DELETED_LAYER");
        write_coord_record(&mut bytes, base + 5 * rec, b"  0", "POINT_3");
        write_layer_header_record(&mut bytes, base + 6 * rec, b' ', "DIG$COLOUR");
        write_coord_record(&mut bytes, base + 7 * rec, b"  0", "POINT_4");

        let points = read_dgd_points_bytes(&bytes).unwrap();

        assert_eq!(points.len(), 1);
        assert_eq!(points[0].name, "POINT_1");
        assert_eq!(points[0].layer_name.as_deref(), Some("ACTIVE_LAYER"));
    }

    #[test]
    fn keeps_only_the_latest_save_of_a_layer_and_drops_deleted_layers() {
        let rec = DGD_COORD_RECORD_LEN;
        let base = 0x1000;
        // stale save block, deleted block, live block — one point each
        let mut bytes = vec![0_u8; base + 6 * rec];
        write_coord_record(&mut bytes, base, b"  0", "POINT_1");
        write_save_record(&mut bytes, base + rec, b' ', "BENCH");
        write_coord_record(&mut bytes, base + 2 * rec, b"  0", "POINT_2");
        write_save_record(&mut bytes, base + 3 * rec, b'D', "OLD_LAYER");
        write_coord_record(&mut bytes, base + 4 * rec, b"1  ", "POINT_3");
        write_save_record(&mut bytes, base + 5 * rec, b' ', "BENCH");

        let points = read_dgd_points_bytes(&bytes).unwrap();

        assert_eq!(points.len(), 1, "stale and deleted blocks are dropped");
        assert_eq!(points[0].name, "POINT_3");
        assert_eq!(points[0].seg_type, 1, "left-aligned seg field parses");
        assert_eq!(points[0].layer_name.as_deref(), Some("BENCH"));
    }

    #[test]
    fn extracts_text_objects_and_prunes_their_coordinates_from_points() {
        let rec = DGD_COORD_RECORD_LEN;
        let base = 0x1000;
        let mut bytes = vec![0_u8; base + 6 * rec];
        write_object_record(&mut bytes, base, 0x04, "REACTIVETEXT$1", false);
        write_coord_record_xyz(
            &mut bytes,
            base + rec,
            b"2  ",
            "1:1250",
            (1234.0, 5678.0, 500.0),
        );
        write_coord_record_xyz(
            &mut bytes,
            base + 2 * rec,
            b"44 ",
            "POINT_1",
            (12.5, 1.0, -12.5),
        );
        write_text_line_record(&mut bytes, base + 3 * rec, "300m Non-Vib");
        write_text_line_record(&mut bytes, base + 4 * rec, "Offset");
        write_save_record(&mut bytes, base + 5 * rec, b' ', "BENCH");

        let design = read_dgd_design_bytes(&bytes).unwrap();

        assert_eq!(design.texts.len(), 1);
        let text = &design.texts[0];
        assert_eq!(text.content, "300m Non-Vib\nOffset");
        assert_eq!((text.x, text.y, text.z), (1234.0, 5678.0, 500.0));
        assert_eq!(text.height, 12.5);
        assert_eq!(text.rotation_degrees, -12.5);
        assert_eq!(text.layer_name.as_deref(), Some("BENCH"));
        assert!(
            design.points.is_empty(),
            "text sub-coordinates must not surface as design points"
        );
    }

    #[test]
    fn extracts_3dtext_objects_skipping_the_font_record() {
        let rec = DGD_COORD_RECORD_LEN;
        let base = 0x1000;
        let mut bytes = vec![0_u8; base + 9 * rec];
        write_object_record(&mut bytes, base, 0x0a, "3DTEXT", false);
        write_coord_record_xyz(
            &mut bytes,
            base + rec,
            b"0  ",
            "1:1250",
            (1234.0, 5678.0, 0.0),
        );
        write_coord_record_xyz(
            &mut bytes,
            base + 2 * rec,
            b"0  ",
            "POINT_2",
            (1.0, 0.0, 0.0),
        );
        write_coord_record_xyz(
            &mut bytes,
            base + 3 * rec,
            b"0  ",
            "POINT_3",
            (0.0, 1.0, 0.0),
        );
        write_coord_record_xyz(
            &mut bytes,
            base + 4 * rec,
            b"0  ",
            "POINT_4",
            (0.08, 0.25, 0.0),
        );
        write_coord_record_xyz(&mut bytes, base + 5 * rec, b"0  ", "", (2.0, 2.0, 0.0));
        write_text_line_record(&mut bytes, base + 6 * rec, "TIMES+");
        write_text_line_record(&mut bytes, base + 7 * rec, "M_HEL0103");
        write_save_record(&mut bytes, base + 8 * rec, b' ', "BENCH");

        let design = read_dgd_design_bytes(&bytes).unwrap();

        assert_eq!(design.texts.len(), 1);
        let text = &design.texts[0];
        assert_eq!(
            text.content, "M_HEL0103",
            "the leading font record is not content"
        );
        assert_eq!(text.height, 0.25);
        assert_eq!(text.rotation_degrees, 0.0);
        assert!(design.points.is_empty());
    }

    #[test]
    fn reads_the_closed_flag_from_poly_object_headers() {
        let rec = DGD_COORD_RECORD_LEN;
        let base = 0x1000;
        let mut bytes = vec![0_u8; base + 8 * rec];
        write_object_record(&mut bytes, base, 0x03, "POLY", true);
        write_coord_record(&mut bytes, base + rec, b"0  ", "POINT_1");
        write_coord_record(&mut bytes, base + 2 * rec, b"1  ", "POINT_2");
        write_coord_record(&mut bytes, base + 3 * rec, b"1  ", "POINT_3");
        write_object_record(&mut bytes, base + 4 * rec, 0x03, "POLY", false);
        write_coord_record(&mut bytes, base + 5 * rec, b"0  ", "POINT_1");
        write_coord_record(&mut bytes, base + 6 * rec, b"1  ", "POINT_2");
        write_save_record(&mut bytes, base + 7 * rec, b' ', "BENCH");

        let points = read_dgd_points_bytes(&bytes).unwrap();

        assert_eq!(points.len(), 5);
        assert!(points[..3].iter().all(|point| point.closed));
        assert!(points[3..].iter().all(|point| !point.closed));
    }

    #[test]
    fn reads_dgd_isix_entries() {
        let mut bytes = vec![b' '; 0x400 + 48];
        write_dgd_index_entry(&mut bytes, 0x400, 0x1234, "LAYER");

        assert_eq!(
            read_dgd_index_bytes(&bytes),
            vec![DesignIndexEntry {
                offset: 0x1234,
                name: "LAYER".to_owned(),
            }]
        );
    }

    #[test]
    fn reads_dgd_isix_entries_without_fixed_table_alignment() {
        let entry_offset = 0x411;
        let mut bytes = vec![b' '; entry_offset + 48];
        write_dgd_index_entry(&mut bytes, entry_offset, 0x4321, "BENCH1");

        assert_eq!(
            read_dgd_index_bytes(&bytes),
            vec![DesignIndexEntry {
                offset: 0x4321,
                name: "BENCH1".to_owned(),
            }]
        );
    }

    #[test]
    fn reads_current_dgd_isix_layer_page_before_other_index_pages() {
        let mut bytes = vec![b' '; 0x1400];
        write_dgd_index_entry(&mut bytes, 0x400, 0x10, "$12345");
        write_dgd_index_entry(&mut bytes, 0x430, 0x20, "$23456");
        write_dgd_index_entry(&mut bytes, 0x800, 0x11, "Dog");
        write_dgd_index_entry(&mut bytes, 0x830, 0x22, "1234");
        write_dgd_index_entry(&mut bytes, 0x860, 0x33, "Cat");
        write_dgd_index_entry(&mut bytes, 0x890, 0x44, "DIG$COLOUR");
        write_dgd_index_entry(&mut bytes, 0x8c0, 0x55, "POLYLINE");
        write_dgd_index_entry(&mut bytes, 0x1000, 0x66, "Wolf");

        assert_eq!(
            read_dgd_index_bytes(&bytes),
            vec![
                DesignIndexEntry {
                    offset: 0x11,
                    name: "Dog".to_owned(),
                },
                DesignIndexEntry {
                    offset: 0x22,
                    name: "1234".to_owned(),
                },
                DesignIndexEntry {
                    offset: 0x33,
                    name: "Cat".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn reads_dgd_isix_from_env_path() {
        let Ok(path) = std::env::var("INCLINE_TEST_ISIX") else {
            return;
        };
        let entries = read_dgd_index(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        println!("{path}: {} current layer entries", entries.len());
        for entry in &entries {
            println!("  {:#x}: {}", entry.offset, entry.name);
        }
        assert!(
            !entries.is_empty(),
            "no current layer entries found in {path}"
        );
    }

    #[test]
    fn same_stem_isix_path_finds_mixed_case_sidecar() {
        let dir =
            std::env::temp_dir().join(format!("proinspector_isix_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir(&dir).unwrap();
        let sidecar = dir.join("mine.DGD.ISIX");
        fs::write(&sidecar, []).unwrap();

        let source = dir.join("mine.dgd.isis");
        assert_eq!(same_stem_isix_path(&source), Some(sidecar));

        fs::remove_dir_all(&dir).unwrap();
    }

    fn write_fixed_name(slot: &mut [u8], name: &str) {
        slot.fill(b' ');
        slot[..name.len()].copy_from_slice(name.as_bytes());
    }

    fn write_dgd_index_entry(bytes: &mut [u8], offset: usize, pointer: u32, name: &str) {
        bytes[offset..offset + 4].copy_from_slice(&pointer.to_be_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&[0xff; 4]);
        write_fixed_name(&mut bytes[offset + 8..offset + 48], name);
    }
}
