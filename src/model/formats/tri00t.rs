use std::{
    error::Error,
    fmt, fs,
    io::{self, Write},
    path::Path,
};

const RAW_HEADER_LEN: usize = 120;
const RAW_VERTEX_COUNT_OFFSET: usize = 0x48;
const RAW_FACE_COUNT_OFFSET: usize = 0x60;
const RAW_VERTEX_SIZE: usize = 24;
const RAW_FACE_SIZE: usize = 24;
const VULZ_TOTAL_EXPANDED_OFFSET: usize = 0x20;
const VULZ_PAGE_EXPANDED_LEN: usize = 25_600;
const VULZ_NEXT_PAGE_SCAN_LIMIT: usize = 0x1000;
const VULZ_TOC_BLOCK_SIZE: usize = 0x800;
const VULZ_TOC_NEXT_PTR_OFFSET: usize = 0x3c;

pub const VULZ_MAGIC: &[u8; 8] = b"\xea\xfb\xa7\x8avulZ";

#[derive(Clone, Debug, PartialEq)]
pub struct Triangulation {
    vertices: Vec<Vertex>,
    faces: Vec<Face>,
    bounds: Bounds,
    index_base: IndexBase,
    raw_header: Vec<u8>,
    trailing_attributes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Face {
    indices: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle {
    pub face_index: usize,
    pub vertex_indices: [usize; 3],
    pub vertices: [Vertex; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min: Vertex,
    pub max: Vertex,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexBase {
    Zero,
    One,
}

#[derive(Debug)]
pub enum ReadError {
    Io(io::Error),
    ShortFile {
        needed: usize,
        actual: usize,
    },
    UnexpectedEof {
        offset: usize,
        needed: usize,
    },
    Overflow(&'static str),
    InvalidRawCounts {
        vertices: usize,
        faces: usize,
    },
    InvalidCoordinates,
    InvalidFaceIndices {
        min: u32,
        max: u32,
        vertex_count: usize,
    },
    InvalidVulzPage {
        offset: usize,
        message: String,
    },
    MissingVulzPage {
        search_start: usize,
        decoded_bytes: usize,
        total_bytes: usize,
    },
}

impl Triangulation {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ReadError> {
        let bytes = fs::read(path).map_err(ReadError::Io)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReadError> {
        if bytes.starts_with(VULZ_MAGIC) {
            let raw = decode_vulz_to_raw(bytes)?;
            return read_raw_triangulation(&raw);
        }

        read_raw_triangulation(bytes)
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    pub fn face(&self, index: usize) -> Option<Face> {
        self.faces.get(index).copied()
    }

    pub fn bounds(&self) -> Bounds {
        self.bounds
    }

    pub fn face_vertex_indices(&self, face_index: usize) -> Option<[usize; 3]> {
        self.face(face_index)
            .map(|face| face.indices_zero_based(self.index_base))
    }

    pub fn triangle(&self, face_index: usize) -> Option<Triangle> {
        let face = self.face(face_index)?;
        let vertex_indices = face.indices_zero_based(self.index_base);
        let vertices = [
            self.vertices.get(vertex_indices[0]).copied()?,
            self.vertices.get(vertex_indices[1]).copied()?,
            self.vertices.get(vertex_indices[2]).copied()?,
        ];

        Some(Triangle {
            face_index,
            vertex_indices,
            vertices,
        })
    }

    pub fn triangles(&self) -> impl Iterator<Item = Triangle> + '_ {
        (0..self.faces.len()).filter_map(|index| self.triangle(index))
    }

    pub fn face_vertex_indices_iter(&self) -> impl Iterator<Item = [usize; 3]> + '_ {
        self.faces
            .iter()
            .map(|face| face.indices_zero_based(self.index_base))
    }

    /// Construct a triangulation from a computed vertex list and face index triples.
    /// Face indices must be zero-based and within bounds. The raw header is populated
    /// with the counts so the result can be round-tripped through `write_mesh`.
    pub fn from_vertices_and_faces(vertices: Vec<Vertex>, faces: Vec<[u32; 3]>) -> Self {
        let bounds = Bounds::from_vertices(&vertices);
        let mut raw_header = vec![0u8; RAW_HEADER_LEN];
        raw_header[RAW_VERTEX_COUNT_OFFSET..RAW_VERTEX_COUNT_OFFSET + 4]
            .copy_from_slice(&(vertices.len() as u32).to_be_bytes());
        raw_header[RAW_FACE_COUNT_OFFSET..RAW_FACE_COUNT_OFFSET + 4]
            .copy_from_slice(&(faces.len() as u32).to_be_bytes());
        let faces = faces.into_iter().map(|indices| Face { indices }).collect();
        Self {
            vertices,
            faces,
            bounds,
            index_base: IndexBase::Zero,
            raw_header,
            trailing_attributes: Vec::new(),
        }
    }

    pub fn write_00t(&self, writer: &mut impl Write) -> io::Result<()> {
        let mut header = if self.raw_header.len() == RAW_HEADER_LEN {
            self.raw_header.clone()
        } else {
            vec![0; RAW_HEADER_LEN]
        };
        header[RAW_VERTEX_COUNT_OFFSET..RAW_VERTEX_COUNT_OFFSET + 4]
            .copy_from_slice(&(self.vertex_count() as u32).to_be_bytes());
        header[RAW_FACE_COUNT_OFFSET..RAW_FACE_COUNT_OFFSET + 4]
            .copy_from_slice(&(self.face_count() as u32).to_be_bytes());

        writer.write_all(&header)?;
        for vertex in &self.vertices {
            for value in vertex.as_array() {
                writer.write_all(&value.to_be_bytes())?;
            }
        }
        for (_face, indices) in self.faces.iter().zip(self.face_vertex_indices_iter()) {
            for index in indices {
                writer.write_all(&(index as u32).to_be_bytes())?;
            }
            writer.write_all(&[0u8; 12])?;
        }
        writer.write_all(&self.trailing_attributes)
    }
}

impl Vertex {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub const fn as_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

impl Face {
    pub fn indices_zero_based(self, index_base: IndexBase) -> [usize; 3] {
        match index_base {
            IndexBase::Zero => [
                self.indices[0] as usize,
                self.indices[1] as usize,
                self.indices[2] as usize,
            ],
            IndexBase::One => [
                (self.indices[0] - 1) as usize,
                (self.indices[1] - 1) as usize,
                (self.indices[2] - 1) as usize,
            ],
        }
    }
}

impl Bounds {
    pub fn from_vertices(vertices: &[Vertex]) -> Self {
        let mut min = Vertex::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Vertex::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

        for vertex in vertices {
            min.x = min.x.min(vertex.x);
            min.y = min.y.min(vertex.y);
            min.z = min.z.min(vertex.z);
            max.x = max.x.max(vertex.x);
            max.y = max.y.max(vertex.y);
            max.z = max.z.max(vertex.z);
        }

        Self { min, max }
    }

    pub fn center(self) -> Vertex {
        Vertex::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Io(err) => write!(f, "{err}"),
            ReadError::ShortFile { needed, actual } => {
                write!(
                    f,
                    "file is too short: needed {needed} bytes, found {actual}"
                )
            }
            ReadError::UnexpectedEof { offset, needed } => write!(
                f,
                "unexpected end of file at 0x{offset:x}; needed {needed} bytes"
            ),
            ReadError::Overflow(section) => write!(f, "{section} section overflow"),
            ReadError::InvalidRawCounts { vertices, faces } => {
                write!(f, "invalid raw counts: vertices={vertices} faces={faces}")
            }
            ReadError::InvalidCoordinates => {
                write!(f, "vertex section contains non-finite coordinates")
            }
            ReadError::InvalidFaceIndices {
                min,
                max,
                vertex_count,
            } => {
                write!(
                    f,
                    "face indices are outside the vertex range: min={min} max={max} vertices={vertex_count}"
                )
            }
            ReadError::InvalidVulzPage { offset, message } => {
                write!(f, "invalid vulZ page at 0x{offset:x}: {message}")
            }
            ReadError::MissingVulzPage {
                search_start,
                decoded_bytes,
                total_bytes,
            } => {
                write!(
                    f,
                    "could not find vulZ page after 0x{search_start:x}; decoded {decoded_bytes} of {total_bytes} bytes"
                )
            }
        }
    }
}

impl Error for ReadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReadError::Io(err) => Some(err),
            _ => None,
        }
    }
}

pub fn decode_vulz_bytes(bytes: &[u8]) -> Result<Vec<u8>, ReadError> {
    decode_vulz_to_raw(bytes)
}

fn read_raw_triangulation(bytes: &[u8]) -> Result<Triangulation, ReadError> {
    if bytes.len() < RAW_HEADER_LEN {
        return Err(ReadError::ShortFile {
            needed: RAW_HEADER_LEN,
            actual: bytes.len(),
        });
    }

    let vertex_count = read_be_u32(bytes, RAW_VERTEX_COUNT_OFFSET)? as usize;
    let face_count = read_be_u32(bytes, RAW_FACE_COUNT_OFFSET)? as usize;

    if vertex_count == 0 || face_count == 0 {
        return Err(ReadError::InvalidRawCounts {
            vertices: vertex_count,
            faces: face_count,
        });
    }

    let vertex_bytes = vertex_count
        .checked_mul(RAW_VERTEX_SIZE)
        .ok_or(ReadError::Overflow("vertex"))?;
    let face_bytes = face_count
        .checked_mul(RAW_FACE_SIZE)
        .ok_or(ReadError::Overflow("face"))?;
    let face_start = RAW_HEADER_LEN
        .checked_add(vertex_bytes)
        .ok_or(ReadError::Overflow("vertex"))?;
    let trailer_start = face_start
        .checked_add(face_bytes)
        .ok_or(ReadError::Overflow("face"))?;

    if trailer_start > bytes.len() {
        return Err(ReadError::InvalidRawCounts {
            vertices: vertex_count,
            faces: face_count,
        });
    }

    let mut vertices = Vec::with_capacity(vertex_count);
    for n in 0..vertex_count {
        let offset = RAW_HEADER_LEN + n * RAW_VERTEX_SIZE;
        vertices.push(Vertex::new(
            read_be_f64(bytes, offset)?,
            read_be_f64(bytes, offset + 8)?,
            read_be_f64(bytes, offset + 16)?,
        ));
    }

    if !vertices
        .iter()
        .all(|vertex| vertex.x.is_finite() && vertex.y.is_finite() && vertex.z.is_finite())
    {
        return Err(ReadError::InvalidCoordinates);
    }

    let mut faces = Vec::with_capacity(face_count);
    for n in 0..face_count {
        let offset = face_start + n * RAW_FACE_SIZE;
        faces.push(Face {
            indices: [
                read_be_u32(bytes, offset)?,
                read_be_u32(bytes, offset + 4)?,
                read_be_u32(bytes, offset + 8)?,
            ],
        });
    }

    let index_base = detect_index_base(&faces, vertex_count)?;

    Ok(Triangulation {
        bounds: Bounds::from_vertices(&vertices),
        vertices,
        faces,
        index_base,
        raw_header: bytes[..RAW_HEADER_LEN].to_vec(),
        trailing_attributes: bytes[trailer_start..].to_vec(),
    })
}

/// Follow the linked-list of 0x800-byte TOC blocks to find the first data page.
/// Each block stores a pointer to the next block at offset VULZ_TOC_NEXT_PTR_OFFSET within it.
/// When the pointer no longer equals current_offset + VULZ_TOC_BLOCK_SIZE, we've reached
/// the first data page.
fn vulz_first_page_offset(bytes: &[u8]) -> usize {
    let mut offset = VULZ_TOC_NEXT_PTR_OFFSET;
    loop {
        if offset + 4 > bytes.len() {
            break;
        }
        let ptr = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        if ptr == offset + VULZ_TOC_BLOCK_SIZE {
            offset = ptr;
        } else {
            break;
        }
    }
    offset
}

fn decode_vulz_to_raw(bytes: &[u8]) -> Result<Vec<u8>, ReadError> {
    if bytes.len() < 0x40 {
        return Err(ReadError::ShortFile {
            needed: 0x40,
            actual: bytes.len(),
        });
    }

    let total_len = read_le_u32(bytes, VULZ_TOTAL_EXPANDED_OFFSET)? as usize;
    let mut decoded = Vec::with_capacity(total_len);
    let mut search_start = vulz_first_page_offset(bytes);

    while decoded.len() < total_len {
        let Some(page) = find_next_vulz_page(bytes, search_start) else {
            return Err(ReadError::MissingVulzPage {
                search_start,
                decoded_bytes: decoded.len(),
                total_bytes: total_len,
            });
        };

        let remaining = total_len - decoded.len();
        decoded.extend_from_slice(&page.decoded[..remaining.min(page.decoded.len())]);
        search_start = page.offset + 8 + page.stored_len;
    }

    Ok(decoded)
}

struct DecodedVulzPage {
    offset: usize,
    stored_len: usize,
    decoded: Vec<u8>,
}

fn find_next_vulz_page(bytes: &[u8], search_start: usize) -> Option<DecodedVulzPage> {
    let search_end = bytes
        .len()
        .saturating_sub(8)
        .min(search_start.saturating_add(VULZ_NEXT_PAGE_SCAN_LIMIT));

    for offset in search_start..search_end {
        let stored_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        let expanded_len = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;

        if stored_len == 0
            || stored_len > VULZ_PAGE_EXPANDED_LEN
            || expanded_len < stored_len
            || expanded_len > stored_len + 1024
            || offset + 8 + stored_len > bytes.len()
        {
            continue;
        }

        if let Ok(decoded) = fastlz1_decompress(&bytes[offset + 8..offset + 8 + stored_len])
            && decoded.len() == VULZ_PAGE_EXPANDED_LEN
        {
            return Some(DecodedVulzPage {
                offset,
                stored_len,
                decoded,
            });
        }
    }

    None
}

fn fastlz1_decompress(input: &[u8]) -> Result<Vec<u8>, ReadError> {
    let mut input_offset = 0;
    let mut output = Vec::with_capacity(VULZ_PAGE_EXPANDED_LEN);

    while input_offset < input.len() {
        let op_offset = input_offset;
        let control = input[input_offset];
        input_offset += 1;

        if control < 32 {
            let len = control as usize + 1;
            let end = input_offset + len;
            if end > input.len() {
                return Err(ReadError::InvalidVulzPage {
                    offset: op_offset,
                    message: "literal run exceeds page input".to_string(),
                });
            }
            output.extend_from_slice(&input[input_offset..end]);
            input_offset = end;
        } else {
            let mut len = (control >> 5) as usize;
            let mut reference_offset = ((control & 0x1f) as usize) << 8;

            if len == 7 {
                if input_offset >= input.len() {
                    return Err(ReadError::InvalidVulzPage {
                        offset: op_offset,
                        message: "missing extended match length".to_string(),
                    });
                }
                len += input[input_offset] as usize;
                input_offset += 1;
            }

            if input_offset >= input.len() {
                return Err(ReadError::InvalidVulzPage {
                    offset: op_offset,
                    message: "missing match offset byte".to_string(),
                });
            }

            reference_offset += input[input_offset] as usize;
            input_offset += 1;
            len += 2;

            if reference_offset >= output.len() {
                return Err(ReadError::InvalidVulzPage {
                    offset: op_offset,
                    message: "match reference is before output start".to_string(),
                });
            }

            let start = output.len() - reference_offset - 1;
            for reference_index in start..start + len {
                if reference_index >= output.len() {
                    return Err(ReadError::InvalidVulzPage {
                        offset: op_offset,
                        message: "match reference exceeded output".to_string(),
                    });
                }
                output.push(output[reference_index]);
            }
        }

        if output.len() > VULZ_PAGE_EXPANDED_LEN {
            return Err(ReadError::InvalidVulzPage {
                offset: op_offset,
                message: "expanded page is larger than expected".to_string(),
            });
        }
    }

    Ok(output)
}

fn detect_index_base(faces: &[Face], vertex_count: usize) -> Result<IndexBase, ReadError> {
    let max_index = faces
        .iter()
        .flat_map(|face| face.indices)
        .max()
        .unwrap_or(0);
    let min_index = faces
        .iter()
        .flat_map(|face| face.indices)
        .min()
        .unwrap_or(0);

    if min_index == 0 && max_index < vertex_count as u32 {
        Ok(IndexBase::Zero)
    } else if min_index >= 1 && max_index <= vertex_count as u32 {
        Ok(IndexBase::One)
    } else {
        Err(ReadError::InvalidFaceIndices {
            min: min_index,
            max: max_index,
            vertex_count,
        })
    }
}

fn read_be_u32(bytes: &[u8], offset: usize) -> Result<u32, ReadError> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or(ReadError::UnexpectedEof { offset, needed: 4 })?;
    Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_le_u32(bytes: &[u8], offset: usize) -> Result<u32, ReadError> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or(ReadError::UnexpectedEof { offset, needed: 4 })?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_be_f64(bytes: &[u8], offset: usize) -> Result<f64, ReadError> {
    let slice = bytes
        .get(offset..offset + 8)
        .ok_or(ReadError::UnexpectedEof { offset, needed: 8 })?;
    Ok(f64::from_be_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}
