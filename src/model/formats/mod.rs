//! Translation between Vulcan triangulations and common triangle-mesh formats.
//!
//! OBJ and ASCII PLY polygons are triangulated using a triangle fan. STL
//! normals, OBJ materials/texture coordinates, and PLY properties other than
//! positions and vertex indices are intentionally ignored.

pub(crate) mod bmf;
pub(crate) mod duf;
pub(crate) mod dxf;
pub(crate) mod isis;
pub(crate) mod point_cloud;
pub(crate) mod tri00t;

use std::{
    collections::HashMap,
    error::Error,
    fmt,
    fs::{self, File},
    io::{self, Write},
    path::Path,
};

use tri00t::{ReadError, Triangulation, Vertex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshFormat {
    Obj,
    Stl,
    Ply,
    T00,
}

impl MeshFormat {
    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension
            .trim_start_matches('.')
            .to_ascii_lowercase()
            .as_str()
        {
            "obj" => Some(Self::Obj),
            "stl" => Some(Self::Stl),
            "ply" => Some(Self::Ply),
            "00t" => Some(Self::T00),
            _ => None,
        }
    }

    pub fn from_path(path: impl AsRef<Path>) -> Option<Self> {
        path.as_ref()
            .extension()
            .and_then(|value| value.to_str())
            .and_then(Self::from_extension)
    }
}

#[derive(Debug)]
pub enum TranslationError {
    Io(io::Error),
    UnsupportedExtension(String),
    UnsupportedFeature(String),
    InvalidData {
        line: Option<usize>,
        message: String,
    },
    InvalidMesh(String),
    TooLarge(&'static str),
    Read00t(ReadError),
}

impl fmt::Display for TranslationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::UnsupportedExtension(ext) => write!(f, "unsupported mesh extension: {ext}"),
            Self::UnsupportedFeature(message) => write!(f, "unsupported mesh feature: {message}"),
            Self::InvalidData {
                line: Some(line),
                message,
            } => write!(f, "line {line}: {message}"),
            Self::InvalidData {
                line: None,
                message,
            } => write!(f, "{message}"),
            Self::InvalidMesh(message) => write!(f, "invalid mesh: {message}"),
            Self::TooLarge(section) => write!(f, "{section} count exceeds the format limit"),
            Self::Read00t(error) => write!(f, "{error}"),
        }
    }
}

impl Error for TranslationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Read00t(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for TranslationError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ReadError> for TranslationError {
    fn from(value: ReadError) -> Self {
        Self::Read00t(value)
    }
}

pub fn read_mesh(path: impl AsRef<Path>) -> Result<Triangulation, TranslationError> {
    let path = path.as_ref();
    match MeshFormat::from_path(path).ok_or_else(|| unsupported_path(path))? {
        MeshFormat::Obj => read_obj(path),
        MeshFormat::Stl => read_stl(path),
        MeshFormat::Ply => read_ply(path),
        MeshFormat::T00 => Ok(Triangulation::from_path(path)?),
    }
}

/// Write `mesh` to `path` in the format implied by the extension, invoking
/// `progress` with a completion fraction in `0.0..=1.0` as the file is written
/// (throttled to every few thousand items). Pass `&mut |_| {}` when progress
/// is not needed.
pub fn write_mesh_with_progress(
    mesh: &Triangulation,
    path: impl AsRef<Path>,
    progress: &mut dyn FnMut(f32),
) -> Result<(), TranslationError> {
    let path = path.as_ref();
    let format = MeshFormat::from_path(path).ok_or_else(|| unsupported_path(path))?;
    let file = File::create(path)?;
    let mut writer = io::BufWriter::new(file);
    match format {
        MeshFormat::Obj => write_obj_with_progress(mesh, &mut writer, progress)?,
        MeshFormat::Stl => write_stl_with_progress(mesh, &mut writer, progress)?,
        MeshFormat::Ply => write_ply_with_progress(mesh, &mut writer, progress)?,
        MeshFormat::T00 => mesh.write_00t_with_progress(&mut writer, progress)?,
    }
    writer.flush()?;
    Ok(())
}

/// How many vertices/faces to write between progress callbacks.
pub(crate) const WRITE_PROGRESS_STRIDE: usize = 4096;

fn unsupported_path(path: &Path) -> TranslationError {
    TranslationError::UnsupportedExtension(
        path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string(),
    )
}

pub fn read_obj(path: impl AsRef<Path>) -> Result<Triangulation, TranslationError> {
    read_obj_bytes(&fs::read(path)?)
}

pub fn read_obj_bytes(bytes: &[u8]) -> Result<Triangulation, TranslationError> {
    use rayon::prelude::*;

    let text = std::str::from_utf8(bytes).map_err(|_| invalid(None, "OBJ is not valid UTF-8"))?;

    // Pass 1 (sequential): classify lines. For face lines we record the vertex count at that
    // point so that negative OBJ indices (relative to current position) resolve correctly.
    let mut vertex_lines: Vec<(usize, &str)> = Vec::new();
    let mut face_lines: Vec<(usize, &str, usize)> = Vec::new();
    let mut vertex_count = 0usize;

    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.split('#').next().unwrap_or("").trim();
        match line.split_ascii_whitespace().next() {
            Some("v") => {
                vertex_lines.push((line_number, line));
                vertex_count += 1;
            }
            Some("f") => {
                face_lines.push((line_number, line, vertex_count));
            }
            _ => {}
        }
    }

    // Pass 2: parse vertices in parallel (order preserved by rayon collect).
    let vertices = vertex_lines
        .par_iter()
        .map(|(line_number, line)| {
            let mut fields = line.split_ascii_whitespace();
            fields.next(); // skip "v"
            let x = parse_f64(fields.next(), *line_number, "vertex x")?;
            let y = parse_f64(fields.next(), *line_number, "vertex y")?;
            let z = parse_f64(fields.next(), *line_number, "vertex z")?;
            Ok(Vertex::new(x, y, z))
        })
        .collect::<Result<Vec<Vertex>, TranslationError>>()?;

    // Pass 3: parse and triangulate face lines in parallel.
    let face_groups = face_lines
        .par_iter()
        .map(|(line_number, line, vcount_at_point)| {
            let mut fields = line.split_ascii_whitespace();
            fields.next(); // skip "f"
            let polygon = fields
                .map(|field| parse_obj_index(field, *vcount_at_point, *line_number))
                .collect::<Result<Vec<_>, _>>()?;
            let mut tris = Vec::new();
            triangulate(&polygon, &mut tris, *line_number)?;
            Ok(tris)
        })
        .collect::<Result<Vec<Vec<[u32; 3]>>, TranslationError>>()?;

    let triangles: Vec<[u32; 3]> = face_groups.into_iter().flatten().collect();

    build_mesh(vertices, triangles)
}

fn parse_obj_index(field: &str, vertex_count: usize, line: usize) -> Result<u32, TranslationError> {
    let raw = field.split('/').next().unwrap_or("");
    let index = raw
        .parse::<i64>()
        .map_err(|_| invalid(Some(line), "invalid OBJ vertex index"))?;
    if index == 0 {
        return Err(invalid(Some(line), "OBJ indices cannot be zero"));
    }
    let zero_based = if index > 0 {
        index - 1
    } else {
        vertex_count as i64 + index
    };
    if zero_based < 0 || zero_based >= vertex_count as i64 {
        return Err(invalid(Some(line), "OBJ vertex index is out of range"));
    }
    u32::try_from(zero_based).map_err(|_| TranslationError::TooLarge("vertex"))
}

pub fn read_stl(path: impl AsRef<Path>) -> Result<Triangulation, TranslationError> {
    read_stl_bytes(&fs::read(path)?)
}

pub fn read_stl_bytes(bytes: &[u8]) -> Result<Triangulation, TranslationError> {
    if bytes.len() >= 84 {
        let count = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
        if 84usize.checked_add(count.saturating_mul(50)) == Some(bytes.len()) {
            return read_binary_stl(bytes, count);
        }
    }
    read_ascii_stl(bytes)
}

fn read_binary_stl(bytes: &[u8], count: usize) -> Result<Triangulation, TranslationError> {
    let mut vertices = Vec::new();
    let mut triangles = Vec::with_capacity(count);
    let mut indices = HashMap::<[u32; 3], u32>::new();
    for face in 0..count {
        let base = 84 + face * 50 + 12;
        let mut triangle = [0; 3];
        for (corner, target) in triangle.iter_mut().enumerate() {
            let offset = base + corner * 12;
            let coords = [
                f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()),
                f32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()),
                f32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap()),
            ];
            if !coords.iter().all(|v| v.is_finite()) {
                return Err(invalid(None, "STL contains non-finite coordinates"));
            }
            *target = intern_stl_vertex(coords, &mut vertices, &mut indices)?;
        }
        triangles.push(triangle);
    }
    build_mesh(vertices, triangles)
}

fn read_ascii_stl(bytes: &[u8]) -> Result<Triangulation, TranslationError> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| invalid(None, "invalid binary or ASCII STL"))?;
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut indices = HashMap::<[u32; 3], u32>::new();
    let mut current = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let mut fields = line.split_whitespace();
        if fields.next() != Some("vertex") {
            continue;
        }
        let xyz = [
            parse_f32(fields.next(), line_index + 1, "vertex x")?,
            parse_f32(fields.next(), line_index + 1, "vertex y")?,
            parse_f32(fields.next(), line_index + 1, "vertex z")?,
        ];
        current.push(intern_stl_vertex(xyz, &mut vertices, &mut indices)?);
        if current.len() == 3 {
            triangles.push([current[0], current[1], current[2]]);
            current.clear();
        }
    }
    if !current.is_empty() {
        return Err(invalid(None, "incomplete ASCII STL facet"));
    }
    build_mesh(vertices, triangles)
}

fn intern_stl_vertex(
    xyz: [f32; 3],
    vertices: &mut Vec<Vertex>,
    indices: &mut HashMap<[u32; 3], u32>,
) -> Result<u32, TranslationError> {
    let key = xyz.map(f32::to_bits);
    if let Some(index) = indices.get(&key) {
        return Ok(*index);
    }
    let index = u32::try_from(vertices.len()).map_err(|_| TranslationError::TooLarge("vertex"))?;
    vertices.push(Vertex::new(xyz[0] as f64, xyz[1] as f64, xyz[2] as f64));
    indices.insert(key, index);
    Ok(index)
}

pub fn read_ply(path: impl AsRef<Path>) -> Result<Triangulation, TranslationError> {
    read_ply_bytes(&fs::read(path)?)
}

pub fn read_ply_bytes(bytes: &[u8]) -> Result<Triangulation, TranslationError> {
    let data_start = ply_data_start(bytes)?;
    let header_text = std::str::from_utf8(&bytes[..data_start])
        .map_err(|_| invalid(None, "PLY header is not valid UTF-8"))?;
    let header = parse_ply_header(header_text)?;
    match header.format {
        PlyFormat::Ascii => read_ascii_ply(&bytes[data_start..], &header),
        PlyFormat::BinaryLittleEndian => read_binary_ply(&bytes[data_start..], &header),
    }
}

fn ply_data_start(bytes: &[u8]) -> Result<usize, TranslationError> {
    let mut line_start = 0;
    while line_start < bytes.len() {
        let Some(relative_line_end) = bytes[line_start..].iter().position(|byte| *byte == b'\n')
        else {
            return Err(invalid(None, "missing PLY end_header"));
        };
        let line_end = line_start + relative_line_end;
        let mut line = &bytes[line_start..line_end];
        if line.ends_with(b"\r") {
            line = &line[..line.len() - 1];
        }
        if line == b"end_header" {
            return Ok(line_end + 1);
        }
        line_start = line_end + 1;
    }
    Err(invalid(None, "missing PLY end_header"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
}

#[derive(Clone, Copy, Debug)]
enum PlyScalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

#[derive(Clone, Debug)]
enum PlyProperty {
    Scalar {
        data_type: PlyScalar,
        name: String,
    },
    List {
        count_type: PlyScalar,
        item_type: PlyScalar,
        name: String,
    },
}

struct PlyHeader {
    format: PlyFormat,
    vertex_count: usize,
    face_count: usize,
    vertex_properties: Vec<PlyProperty>,
    face_properties: Vec<PlyProperty>,
}

fn parse_ply_header(text: &str) -> Result<PlyHeader, TranslationError> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.first().map(|line| line.trim()) != Some("ply") {
        return Err(invalid(Some(1), "missing PLY signature"));
    }
    let mut format = None;
    let mut vertex_count = None;
    let mut face_count = None;
    let mut element = "";
    let mut vertex_properties = Vec::new();
    let mut face_properties = Vec::new();
    for (line_index, line) in lines.iter().enumerate() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.as_slice() {
            ["format", "ascii", "1.0"] => format = Some(PlyFormat::Ascii),
            ["format", "binary_little_endian", "1.0"] => {
                format = Some(PlyFormat::BinaryLittleEndian)
            }
            ["format", "binary_big_endian", _] => {
                return Err(TranslationError::UnsupportedFeature(
                    "big-endian binary PLY is not supported".into(),
                ));
            }
            ["element", "vertex", count] => {
                vertex_count = count.parse().ok();
                element = "vertex";
            }
            ["element", "face", count] => {
                face_count = count.parse().ok();
                element = "face";
            }
            ["element", name, _] => element = name,
            ["property", data_type, name] if element == "vertex" || element == "face" => {
                let property = PlyProperty::Scalar {
                    data_type: parse_ply_scalar(data_type, line_index + 1)?,
                    name: (*name).to_string(),
                };
                if element == "vertex" {
                    vertex_properties.push(property);
                } else {
                    face_properties.push(property);
                }
            }
            ["property", "list", count_type, item_type, name]
                if element == "vertex" || element == "face" =>
            {
                if element == "vertex" {
                    return Err(invalid(
                        Some(line_index + 1),
                        "list property is invalid for PLY vertices",
                    ));
                }
                face_properties.push(PlyProperty::List {
                    count_type: parse_ply_scalar(count_type, line_index + 1)?,
                    item_type: parse_ply_scalar(item_type, line_index + 1)?,
                    name: (*name).to_string(),
                });
            }
            _ => {}
        }
    }
    let format = format.ok_or_else(|| invalid(None, "missing or unsupported PLY format"))?;
    let vertex_count = vertex_count.ok_or_else(|| invalid(None, "missing PLY vertex element"))?;
    let face_count = face_count.ok_or_else(|| invalid(None, "missing PLY face element"))?;
    Ok(PlyHeader {
        format,
        vertex_count,
        face_count,
        vertex_properties,
        face_properties,
    })
}

fn parse_ply_scalar(name: &str, line: usize) -> Result<PlyScalar, TranslationError> {
    match name {
        "char" | "int8" => Ok(PlyScalar::I8),
        "uchar" | "uint8" => Ok(PlyScalar::U8),
        "short" | "int16" => Ok(PlyScalar::I16),
        "ushort" | "uint16" => Ok(PlyScalar::U16),
        "int" | "int32" => Ok(PlyScalar::I32),
        "uint" | "uint32" => Ok(PlyScalar::U32),
        "float" | "float32" => Ok(PlyScalar::F32),
        "double" | "float64" => Ok(PlyScalar::F64),
        _ => Err(invalid(
            Some(line),
            format!("unsupported PLY scalar type {name}"),
        )),
    }
}

fn ply_position_properties(header: &PlyHeader) -> Result<[usize; 3], TranslationError> {
    let positions = ["x", "y", "z"].map(|name| {
        header
            .vertex_properties
            .iter()
            .position(|property| matches!(property, PlyProperty::Scalar { name: property_name, .. } if property_name == name))
    });
    if positions.iter().any(Option::is_none) {
        return Err(invalid(None, "PLY vertices require x, y, and z properties"));
    }
    Ok(positions.map(Option::unwrap))
}

fn read_ascii_ply(bytes: &[u8], header: &PlyHeader) -> Result<Triangulation, TranslationError> {
    let text = std::str::from_utf8(bytes).map_err(|_| invalid(None, "invalid ASCII PLY data"))?;
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < header.vertex_count + header.face_count {
        return Err(invalid(None, "PLY data is truncated"));
    }
    let positions = ply_position_properties(header)?;
    let mut vertices = Vec::with_capacity(header.vertex_count);
    for (row, line) in lines.iter().take(header.vertex_count).enumerate() {
        let line_number = row + 1;
        let fields: Vec<&str> = line.split_whitespace().collect();
        let value = |axis: usize| -> Result<f64, TranslationError> {
            fields
                .get(positions[axis])
                .and_then(|v| v.parse().ok())
                .filter(|v: &f64| v.is_finite())
                .ok_or_else(|| invalid(Some(line_number), "invalid PLY vertex coordinate"))
        };
        vertices.push(Vertex::new(value(0)?, value(1)?, value(2)?));
    }
    let mut triangles = Vec::new();
    for row in 0..header.face_count {
        let line_number = header.vertex_count + row + 1;
        let fields: Vec<&str> = lines[header.vertex_count + row]
            .split_whitespace()
            .collect();
        let mut field_index = 0;
        let mut polygon = None;
        for property in &header.face_properties {
            match property {
                PlyProperty::Scalar { .. } => {
                    fields
                        .get(field_index)
                        .ok_or_else(|| invalid(Some(line_number), "truncated PLY face"))?;
                    field_index += 1;
                }
                PlyProperty::List { name, .. } => {
                    let count = fields
                        .get(field_index)
                        .and_then(|value| value.parse::<usize>().ok())
                        .ok_or_else(|| invalid(Some(line_number), "invalid PLY face"))?;
                    field_index += 1;
                    if fields.len() < field_index + count {
                        return Err(invalid(Some(line_number), "truncated PLY face"));
                    }
                    let values = fields[field_index..field_index + count]
                        .iter()
                        .map(|value| {
                            value
                                .parse::<u32>()
                                .map_err(|_| invalid(Some(line_number), "invalid PLY face index"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    field_index += count;
                    if name == "vertex_indices" || name == "vertex_index" {
                        polygon = Some(values);
                    }
                }
            }
        }
        let polygon = polygon
            .ok_or_else(|| invalid(Some(line_number), "PLY face has no vertex_indices property"))?;
        if polygon.iter().any(|i| *i as usize >= vertices.len()) {
            return Err(invalid(Some(line_number), "PLY face index is out of range"));
        }
        triangulate(&polygon, &mut triangles, line_number)?;
    }
    build_mesh(vertices, triangles)
}

fn read_binary_ply(bytes: &[u8], header: &PlyHeader) -> Result<Triangulation, TranslationError> {
    let positions = ply_position_properties(header)?;
    let mut offset = 0;
    let mut vertices = Vec::with_capacity(header.vertex_count);
    for _ in 0..header.vertex_count {
        let mut xyz = [0.0; 3];
        for (property_index, property) in header.vertex_properties.iter().enumerate() {
            let PlyProperty::Scalar { data_type, .. } = property else {
                unreachable!()
            };
            let value = read_ply_number(bytes, &mut offset, *data_type)?;
            if let Some(axis) = positions
                .iter()
                .position(|position| *position == property_index)
            {
                xyz[axis] = value;
            }
        }
        if !xyz.iter().all(|value| value.is_finite()) {
            return Err(invalid(None, "PLY contains non-finite coordinates"));
        }
        vertices.push(Vertex::new(xyz[0], xyz[1], xyz[2]));
    }
    let mut triangles = Vec::new();
    for face_index in 0..header.face_count {
        let mut polygon = None;
        for property in &header.face_properties {
            match property {
                PlyProperty::Scalar { data_type, .. } => {
                    read_ply_number(bytes, &mut offset, *data_type)?;
                }
                PlyProperty::List {
                    count_type,
                    item_type,
                    name,
                } => {
                    let count = read_ply_integer(bytes, &mut offset, *count_type)?;
                    let mut values = Vec::with_capacity(count);
                    for _ in 0..count {
                        let value = read_ply_integer(bytes, &mut offset, *item_type)?;
                        values.push(
                            u32::try_from(value)
                                .map_err(|_| TranslationError::TooLarge("face index"))?,
                        );
                    }
                    if name == "vertex_indices" || name == "vertex_index" {
                        polygon = Some(values);
                    }
                }
            }
        }
        let polygon =
            polygon.ok_or_else(|| invalid(None, "PLY face has no vertex_indices property"))?;
        if polygon
            .iter()
            .any(|index| *index as usize >= vertices.len())
        {
            return Err(invalid(None, "PLY face index is out of range"));
        }
        triangulate(&polygon, &mut triangles, face_index + 1)?;
    }
    build_mesh(vertices, triangles)
}

fn read_ply_number(
    bytes: &[u8],
    offset: &mut usize,
    data_type: PlyScalar,
) -> Result<f64, TranslationError> {
    let take = |offset: &mut usize, size: usize| -> Result<&[u8], TranslationError> {
        let value = bytes
            .get(*offset..*offset + size)
            .ok_or_else(|| invalid(None, "binary PLY data is truncated"))?;
        *offset += size;
        Ok(value)
    };
    Ok(match data_type {
        PlyScalar::I8 => take(offset, 1)?[0] as i8 as f64,
        PlyScalar::U8 => take(offset, 1)?[0] as f64,
        PlyScalar::I16 => i16::from_le_bytes(take(offset, 2)?.try_into().unwrap()) as f64,
        PlyScalar::U16 => u16::from_le_bytes(take(offset, 2)?.try_into().unwrap()) as f64,
        PlyScalar::I32 => i32::from_le_bytes(take(offset, 4)?.try_into().unwrap()) as f64,
        PlyScalar::U32 => u32::from_le_bytes(take(offset, 4)?.try_into().unwrap()) as f64,
        PlyScalar::F32 => f32::from_le_bytes(take(offset, 4)?.try_into().unwrap()) as f64,
        PlyScalar::F64 => f64::from_le_bytes(take(offset, 8)?.try_into().unwrap()),
    })
}

fn read_ply_integer(
    bytes: &[u8],
    offset: &mut usize,
    data_type: PlyScalar,
) -> Result<usize, TranslationError> {
    match data_type {
        PlyScalar::F32 | PlyScalar::F64 => Err(invalid(
            None,
            "PLY list sizes and indices must use integer types",
        )),
        _ => {
            let value = read_ply_number(bytes, offset, data_type)?;
            if value < 0.0 || value > usize::MAX as f64 {
                return Err(invalid(None, "invalid PLY list integer"));
            }
            Ok(value as usize)
        }
    }
}

pub fn write_obj_with_progress(
    mesh: &Triangulation,
    writer: &mut impl Write,
    progress: &mut dyn FnMut(f32),
) -> io::Result<()> {
    let vertex_count = mesh.vertex_count().max(1);
    let face_count = mesh.face_count().max(1);
    writeln!(writer, "# exported by vulcan-formats")?;
    for (i, v) in mesh.vertices().iter().enumerate() {
        // Convert Z-up coordinates back to OBJ's Y-up convention.
        writeln!(writer, "v {:.12} {:.12} {:.12}", v.x, v.z, -v.y)?;
        if i % WRITE_PROGRESS_STRIDE == 0 {
            progress(0.5 * i as f32 / vertex_count as f32);
        }
    }
    for (i, f) in mesh.face_vertex_indices_iter().enumerate() {
        writeln!(writer, "f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1)?;
        if i % WRITE_PROGRESS_STRIDE == 0 {
            progress(0.5 + 0.5 * i as f32 / face_count as f32);
        }
    }
    progress(1.0);
    Ok(())
}

pub fn write_stl_with_progress(
    mesh: &Triangulation,
    writer: &mut impl Write,
    progress: &mut dyn FnMut(f32),
) -> io::Result<()> {
    let mut header = [0u8; 80];
    let label = b"binary STL from vulcan-formats";
    header[..label.len()].copy_from_slice(label);
    writer.write_all(&header)?;
    writer.write_all(&(mesh.face_count() as u32).to_le_bytes())?;
    let face_count = mesh.face_count().max(1);
    for (face_index, triangle) in mesh.triangles().enumerate() {
        if face_index % WRITE_PROGRESS_STRIDE == 0 {
            progress(face_index as f32 / face_count as f32);
        }
        let a = triangle.vertices[0];
        let b = triangle.vertices[1];
        let c = triangle.vertices[2];
        let ux = b.x - a.x;
        let uy = b.y - a.y;
        let uz = b.z - a.z;
        let vx = c.x - a.x;
        let vy = c.y - a.y;
        let vz = c.z - a.z;
        let mut normal = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > 0.0 {
            normal.iter_mut().for_each(|v| *v /= length);
        }
        for value in normal
            .into_iter()
            .chain(triangle.vertices.into_iter().flat_map(Vertex::as_array))
        {
            writer.write_all(&(value as f32).to_le_bytes())?;
        }
        writer.write_all(&0u16.to_le_bytes())?;
    }
    progress(1.0);
    Ok(())
}

pub fn write_ply_with_progress(
    mesh: &Triangulation,
    writer: &mut impl Write,
    progress: &mut dyn FnMut(f32),
) -> io::Result<()> {
    writeln!(
        writer,
        "ply\nformat ascii 1.0\ncomment exported by vulcan-formats"
    )?;
    writeln!(
        writer,
        "element vertex {}\nproperty double x\nproperty double y\nproperty double z",
        mesh.vertex_count()
    )?;
    writeln!(
        writer,
        "element face {}\nproperty list uchar uint vertex_indices\nend_header",
        mesh.face_count()
    )?;
    let vertex_count = mesh.vertex_count().max(1);
    let face_count = mesh.face_count().max(1);
    for (i, v) in mesh.vertices().iter().enumerate() {
        writeln!(writer, "{:.12} {:.12} {:.12}", v.x, v.y, v.z)?;
        if i % WRITE_PROGRESS_STRIDE == 0 {
            progress(0.5 * i as f32 / vertex_count as f32);
        }
    }
    for (i, f) in mesh.face_vertex_indices_iter().enumerate() {
        writeln!(writer, "3 {} {} {}", f[0], f[1], f[2])?;
        if i % WRITE_PROGRESS_STRIDE == 0 {
            progress(0.5 + 0.5 * i as f32 / face_count as f32);
        }
    }
    progress(1.0);
    Ok(())
}

fn triangulate(
    polygon: &[u32],
    output: &mut Vec<[u32; 3]>,
    line: usize,
) -> Result<(), TranslationError> {
    if polygon.len() < 3 {
        return Err(invalid(Some(line), "face has fewer than three vertices"));
    }
    for i in 1..polygon.len() - 1 {
        output.push([polygon[0], polygon[i], polygon[i + 1]]);
    }
    Ok(())
}

fn build_mesh(
    vertices: Vec<Vertex>,
    triangles: Vec<[u32; 3]>,
) -> Result<Triangulation, TranslationError> {
    if vertices.is_empty() {
        return Err(TranslationError::InvalidMesh("mesh has no vertices".into()));
    }
    if triangles.is_empty() {
        return Err(TranslationError::InvalidMesh("mesh has no faces".into()));
    }
    if !vertices
        .iter()
        .all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite())
    {
        return Err(TranslationError::InvalidMesh(
            "mesh contains non-finite coordinates".into(),
        ));
    }
    Ok(Triangulation::from_vertices_and_faces(vertices, triangles))
}

fn parse_f64(value: Option<&str>, line: usize, name: &str) -> Result<f64, TranslationError> {
    value
        .and_then(|v| v.parse().ok())
        .filter(|v: &f64| v.is_finite())
        .ok_or_else(|| invalid(Some(line), format!("invalid {name}")))
}

fn parse_f32(value: Option<&str>, line: usize, name: &str) -> Result<f32, TranslationError> {
    value
        .and_then(|v| v.parse().ok())
        .filter(|v: &f32| v.is_finite())
        .ok_or_else(|| invalid(Some(line), format!("invalid {name}")))
}

fn invalid(line: Option<usize>, message: impl Into<String>) -> TranslationError {
    TranslationError::InvalidData {
        line,
        message: message.into(),
    }
}
