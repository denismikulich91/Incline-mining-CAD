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
const VULZ_PAGE_SIZE_OFFSET: usize = 0x14;
const VULZ_TOTAL_EXPANDED_OFFSET: usize = 0x20;
const VULZ_AUX_OFFSET_OFFSET: usize = 0x2c;
const VULZ_AUX_LEN_OFFSET: usize = 0x34;
const VULZ_WALK_START: usize = 0x3c;
/// Typical decoded page size; only used as an allocation hint.
const VULZ_PAGE_EXPANDED_LEN: usize = 25_600;

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
    MissingVulzPages {
        missing: usize,
        total: usize,
    },
}

impl Triangulation {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ReadError> {
        let bytes = fs::read(path).map_err(ReadError::Io)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReadError> {
        if bytes.starts_with(VULZ_MAGIC) {
            let raw = decode_vulz_bytes(bytes)?;
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
            ReadError::MissingVulzPages { missing, total } => {
                write!(
                    f,
                    "vulZ container stores only {} of {total} pages; the absent pages are freed/journaled space",
                    total - missing
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

/// Fully decode a vulZ container; errors if any logical page is absent
/// (triangulations are stored contiguously, so a hole means corruption).
pub fn decode_vulz_bytes(bytes: &[u8]) -> Result<Vec<u8>, ReadError> {
    let archive = decode_vulz_archive(bytes)?;
    if archive.missing_pages > 0 {
        return Err(ReadError::MissingVulzPages {
            missing: archive.missing_pages,
            total: archive.total_pages,
        });
    }
    Ok(archive.data)
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

/// A decoded vulZ container.
///
/// `data` is the logical database image: each compressed page carries its
/// logical page number in the first trailer word (zero in older files, which
/// store pages strictly sequentially) and is placed at `number * page_size`.
/// Journaled files keep only live pages, so the image can contain zero-filled
/// holes where pages were freed; `missing_pages` counts them.
///
/// `aux` is the auxiliary stream outside the page-numbered space (in design
/// databases: the PNG layer-preview gallery, which also carries layer names).
pub(crate) struct VulzArchive {
    pub data: Vec<u8>,
    pub aux: Vec<u8>,
    pub missing_pages: usize,
    pub total_pages: usize,
}

/// Largest stored/decoded page size considered plausible.
const MAX_VULZ_PAGE_LEN: usize = 4 << 20;
/// Largest gap between a page's stored data and the next page header
/// (page-number word, end marker, zero padding).
const MAX_VULZ_PAGE_TRAILER: usize = 0x4000;
/// Cap on the logical image size to guard against corrupt headers.
const MAX_VULZ_TOTAL_LEN: usize = 1 << 30;

/// Decode a vulZ container by walking its page chain exactly.
///
/// The file is a linked structure starting at [`VULZ_WALK_START`]: an 8-byte
/// record is either a pointer (`second == 0`; table-of-contents blocks and
/// section links both use this form, `first == 0` terminates) or a page
/// header `(stored_len, advance)` followed by one FastLZ stream. `advance`
/// spans the stored data plus the trailer, so the next record always sits at
/// `offset + 8 + advance`.
pub(crate) fn decode_vulz_archive(bytes: &[u8]) -> Result<VulzArchive, ReadError> {
    if bytes.len() < 0x40 {
        return Err(ReadError::ShortFile {
            needed: 0x40,
            actual: bytes.len(),
        });
    }
    let page_size = read_le_u32(bytes, VULZ_PAGE_SIZE_OFFSET)? as usize;
    if !(1024..=MAX_VULZ_PAGE_LEN).contains(&page_size) {
        return Err(ReadError::InvalidVulzPage {
            offset: VULZ_PAGE_SIZE_OFFSET,
            message: format!("implausible page size {page_size}"),
        });
    }
    let total_len = read_le_u32(bytes, VULZ_TOTAL_EXPANDED_OFFSET)? as usize
        | ((read_le_u32(bytes, VULZ_TOTAL_EXPANDED_OFFSET + 4)? as usize) << 32);
    if total_len == 0 || total_len > MAX_VULZ_TOTAL_LEN {
        return Err(ReadError::InvalidVulzPage {
            offset: VULZ_TOTAL_EXPANDED_OFFSET,
            message: format!("implausible total expanded length {total_len}"),
        });
    }
    let aux_offset = read_le_u32(bytes, VULZ_AUX_OFFSET_OFFSET)? as usize;
    let aux_len = read_le_u32(bytes, VULZ_AUX_LEN_OFFSET)? as usize;

    let total_pages = total_len.div_ceil(page_size);
    let mut data = vec![0u8; total_pages * page_size];
    let mut covered = vec![false; total_pages];
    let mut aux = Vec::new();
    let mut next_sequential = 0usize;
    let mut offset = VULZ_WALK_START;
    let mut visited = std::collections::HashSet::new();

    while offset + 8 <= bytes.len() {
        if !visited.insert(offset) {
            return Err(ReadError::InvalidVulzPage {
                offset,
                message: "cycle in page chain".to_string(),
            });
        }
        let first = read_le_u32(bytes, offset)? as usize;
        let second = read_le_u32(bytes, offset + 4)? as usize;

        if second == 0 {
            if first == 0 || first + 8 > bytes.len() {
                break;
            }
            offset = first;
            continue;
        }

        let (stored_len, advance) = (first, second);
        if stored_len > MAX_VULZ_PAGE_LEN
            || advance < stored_len
            || advance > stored_len + MAX_VULZ_PAGE_TRAILER
            || offset + 8 + stored_len > bytes.len()
        {
            break;
        }

        let page =
            fastlz_decompress(&bytes[offset + 8..offset + 8 + stored_len]).map_err(|error| {
                ReadError::InvalidVulzPage {
                    offset,
                    message: format!("page failed to decompress: {error}"),
                }
            })?;

        if aux_offset != 0 && offset >= aux_offset && aux.len() < aux_len {
            aux.extend_from_slice(&page);
        } else {
            if page.len() != page_size {
                return Err(ReadError::InvalidVulzPage {
                    offset,
                    message: format!(
                        "page decompressed to {} bytes, expected {page_size}",
                        page.len()
                    ),
                });
            }
            let number = bytes
                .get(offset + 8 + stored_len..offset + 12 + stored_len)
                .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]) as usize)
                .unwrap_or(0);
            let index = if number == 0 { next_sequential } else { number };
            // Sample files carry one spare page past the stated total;
            // anything beyond the logical image is ignored.
            if index < total_pages {
                data[index * page_size..(index + 1) * page_size].copy_from_slice(&page);
                covered[index] = true;
            }
            next_sequential = index + 1;
        }

        offset += 8 + advance;
    }

    aux.truncate(aux_len);
    data.truncate(total_len);
    let missing_pages = covered.iter().filter(|&&hit| !hit).count();
    Ok(VulzArchive {
        data,
        aux,
        missing_pages,
        total_pages,
    })
}

/// Decompress one FastLZ stream (as Vulcan stores per page). The level is
/// carried in the top three bits of the first control byte — 0 for level 1,
/// 1 for level 2 (used for less-compressible/newer content). Level 2 extends
/// level 1 with a looped extended match length and a 16-bit far-distance
/// escape, per the reference `fastlz.c`.
fn fastlz_decompress(input: &[u8]) -> Result<Vec<u8>, ReadError> {
    let Some(&first) = input.first() else {
        return Ok(Vec::new());
    };
    let level2 = match first >> 5 {
        0 => false,
        1 => true,
        other => {
            return Err(ReadError::InvalidVulzPage {
                offset: 0,
                message: format!("unsupported FastLZ level {}", other + 1),
            });
        }
    };

    let mut input_offset = 0;
    let mut output = Vec::with_capacity(VULZ_PAGE_EXPANDED_LEN);
    let mut first_op = true;

    while input_offset < input.len() {
        let op_offset = input_offset;
        let mut control = input[input_offset];
        if first_op {
            // The level marker lives in the first op's top bits; the op
            // itself is always a literal run.
            control &= 0x1f;
            first_op = false;
        }
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
                loop {
                    if input_offset >= input.len() {
                        return Err(ReadError::InvalidVulzPage {
                            offset: op_offset,
                            message: "missing extended match length".to_string(),
                        });
                    }
                    let code = input[input_offset];
                    input_offset += 1;
                    len += code as usize;
                    if !level2 || code != 255 {
                        break;
                    }
                }
            }

            if input_offset >= input.len() {
                return Err(ReadError::InvalidVulzPage {
                    offset: op_offset,
                    message: "missing match offset byte".to_string(),
                });
            }

            let code = input[input_offset];
            input_offset += 1;
            reference_offset += code as usize;

            if level2 && code == 255 && control & 0x1f == 0x1f {
                // Far match: distance is a 16-bit value biased past the
                // 13-bit range of the short form.
                if input_offset + 2 > input.len() {
                    return Err(ReadError::InvalidVulzPage {
                        offset: op_offset,
                        message: "missing far match offset".to_string(),
                    });
                }
                let hi = input[input_offset] as usize;
                let lo = input[input_offset + 1] as usize;
                input_offset += 2;
                reference_offset = (hi << 8) + lo + 8191;
            }

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

        if output.len() > MAX_VULZ_PAGE_LEN {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastlz_level2_decodes_extended_length_and_far_distance() {
        // Level-2 stream (level marker in the first byte's top bits):
        //  1. literal "ABCD"
        //  2. short match: 4 bytes at distance 4 -> "ABCDABCD"
        //  3. extended-length match (looped 255-continuation): 268 x 'D'
        let mut input = vec![0x20 | 3, b'A', b'B', b'C', b'D'];
        input.extend_from_slice(&[0x40, 3]); // len=2 (+2 bytes), distance 3+1
        input.extend_from_slice(&[0xE0, 255, 4, 0]); // len=7+255+4 (+2), distance 1
        let decoded = fastlz_decompress(&input).expect("valid level-2 stream");
        let mut expected = b"ABCDABCD".to_vec();
        expected.extend(std::iter::repeat_n(b'D', 268));
        assert_eq!(decoded, expected);

        // Far-distance escape: grow the output past the 13-bit range with an
        // RLE match, mark a position, then copy from distance 9002.
        let mut input = vec![0x20, b'X'];
        // one level-2 extended match: 7 + 255*35 + 66 + 2 = 9000 x 'X'
        input.push(0xE0);
        input.extend(std::iter::repeat_n(255u8, 35));
        input.extend_from_slice(&[66, 0]);
        input.extend_from_slice(&[0x00, b'Y']);
        // control len=1 (+2 = 3 bytes), short-distance bits all set + code 255
        // triggers the far form: distance = (3*256 + 42) + 8191 + 1 = 9002.
        input.extend_from_slice(&[0x3f, 255, 3, 42]);
        let decoded = fastlz_decompress(&input).expect("valid far-distance stream");
        assert_eq!(decoded.len(), 9001 + 1 + 3);
        assert!(decoded[..9001].iter().all(|&b| b == b'X'));
        assert_eq!(decoded[9001], b'Y');
        assert_eq!(&decoded[9002..], b"XXX", "far match copies from offset 0");
    }

    #[test]
    fn fastlz_level1_still_reads_a_plain_literal_stream() {
        let input = vec![0x02, b'a', b'b', b'c'];
        assert_eq!(fastlz_decompress(&input).unwrap(), b"abc");
    }

    /// Level-1 compress `len` copies of `fill` (a literal then RLE matches),
    /// as a stored page body.
    fn l1_rle_page(fill: u8, len: usize) -> Vec<u8> {
        assert!(len >= 4);
        let mut out = vec![0x00, fill];
        let mut produced = 1usize;
        while produced < len {
            let m = (len - produced).min(264);
            assert!(m >= 3, "helper requires match-sized remainders");
            if m - 2 <= 7 {
                out.extend_from_slice(&[(((m - 2) as u8) << 5), 0]);
            } else {
                out.extend_from_slice(&[0xE0, (m - 2 - 7) as u8, 0]);
            }
            produced += m;
        }
        out
    }

    const TEST_PAGE: usize = 4096;

    /// Build a synthetic vulZ container. Each page is (logical page number,
    /// fill byte); number 0 places sequentially like older Vulcan files.
    fn build_vulz(total_len: usize, pages: &[(u32, u8)], aux: &[u8]) -> Vec<u8> {
        let mut file = vec![0u8; 0x40];
        file[..8].copy_from_slice(VULZ_MAGIC);
        file[VULZ_PAGE_SIZE_OFFSET..VULZ_PAGE_SIZE_OFFSET + 4]
            .copy_from_slice(&(TEST_PAGE as u32).to_le_bytes());
        file[VULZ_TOTAL_EXPANDED_OFFSET..VULZ_TOTAL_EXPANDED_OFFSET + 8]
            .copy_from_slice(&(total_len as u64).to_le_bytes());
        // Walk start holds a pointer chain of one hop, like real files (the
        // pointer's second word, at 0x40, stays zero so it reads as a jump).
        file.resize(0x44, 0);
        file[VULZ_WALK_START..VULZ_WALK_START + 4].copy_from_slice(&0x44u32.to_le_bytes());
        for &(number, fill) in pages {
            append_page(&mut file, number, &l1_rle_page(fill, TEST_PAGE));
        }
        if !aux.is_empty() {
            let aux_offset = file.len();
            file[VULZ_AUX_OFFSET_OFFSET..VULZ_AUX_OFFSET_OFFSET + 4]
                .copy_from_slice(&(aux_offset as u32).to_le_bytes());
            file[VULZ_AUX_LEN_OFFSET..VULZ_AUX_LEN_OFFSET + 4]
                .copy_from_slice(&(aux.len() as u32).to_le_bytes());
            let mut body = vec![(aux.len() as u8).saturating_sub(1).min(31)];
            body.extend_from_slice(aux);
            append_page(&mut file, 0, &body);
        }
        file
    }

    fn append_page(file: &mut Vec<u8>, number: u32, body: &[u8]) {
        let trailer = [
            number.to_le_bytes().as_slice(),
            &[0xfa, 0xfb, 0xfc, 0xfd, 0x00, 0x00],
        ]
        .concat();
        file.extend_from_slice(&(body.len() as u32).to_le_bytes());
        file.extend_from_slice(&((body.len() + trailer.len()) as u32).to_le_bytes());
        file.extend_from_slice(body);
        file.extend_from_slice(&trailer);
    }

    #[test]
    fn vulz_places_sequential_zero_numbered_pages() {
        let file = build_vulz(2 * TEST_PAGE, &[(0, b'Q'), (0, b'R')], &[]);
        let decoded = decode_vulz_bytes(&file).expect("sequential pages");
        assert_eq!(decoded.len(), 2 * TEST_PAGE);
        assert!(decoded[..TEST_PAGE].iter().all(|&b| b == b'Q'));
        assert!(decoded[TEST_PAGE..].iter().all(|&b| b == b'R'));
    }

    #[test]
    fn vulz_places_journaled_pages_by_number_and_last_write_wins() {
        // Page 1 is written twice; the later version must win.
        let file = build_vulz(
            3 * TEST_PAGE,
            &[(0, b'A'), (1, b'X'), (2, b'C'), (1, b'B')],
            &[],
        );
        let decoded = decode_vulz_bytes(&file).expect("journaled pages");
        assert!(decoded[..TEST_PAGE].iter().all(|&b| b == b'A'));
        assert!(decoded[TEST_PAGE..2 * TEST_PAGE].iter().all(|&b| b == b'B'));
        assert!(decoded[2 * TEST_PAGE..].iter().all(|&b| b == b'C'));
    }

    #[test]
    fn vulz_missing_pages_error_strictly_but_leave_holes_in_the_archive() {
        let file = build_vulz(3 * TEST_PAGE, &[(0, b'A'), (2, b'C')], &[]);
        let error = decode_vulz_bytes(&file).expect_err("page 1 is absent");
        assert!(matches!(
            error,
            ReadError::MissingVulzPages {
                missing: 1,
                total: 3
            }
        ));

        let archive = decode_vulz_archive(&file).expect("archive tolerates holes");
        assert_eq!(archive.missing_pages, 1);
        assert!(archive.data[..TEST_PAGE].iter().all(|&b| b == b'A'));
        assert!(
            archive.data[TEST_PAGE..2 * TEST_PAGE]
                .iter()
                .all(|&b| b == 0)
        );
        assert!(archive.data[2 * TEST_PAGE..].iter().all(|&b| b == b'C'));
    }

    #[test]
    fn vulz_splits_the_aux_preview_stream_from_the_page_image() {
        let aux = b"gallery bytes with layer names";
        let file = build_vulz(TEST_PAGE, &[(0, b'D')], aux);
        let archive = decode_vulz_archive(&file).expect("aux page decodes");
        assert_eq!(archive.data.len(), TEST_PAGE);
        assert!(archive.data.iter().all(|&b| b == b'D'));
        assert_eq!(archive.aux, aux);
    }

    #[test]
    fn decodes_repo_sample_vulz_files() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let samples = [
            root.join("test/Vulcan/00t/phase_1_topo.00t"),
            root.join("test/Vulcan/00t/topo.00t"),
        ];
        for path in samples {
            if !path.exists() {
                continue;
            }
            let triangulation = Triangulation::from_path(&path).expect("sample should decode");
            assert!(triangulation.vertex_count() > 0);
            assert!(triangulation.face_count() > 0);
        }

        let isis = root.join("test/Vulcan/dgd_isis/thorarea1.dgd.isis");
        if isis.exists() {
            let bytes = std::fs::read(&isis).unwrap();
            let total = read_le_u32(&bytes, VULZ_TOTAL_EXPANDED_OFFSET).unwrap() as usize;
            let decoded = decode_vulz_bytes(&bytes).expect("sample dgd should decode");
            assert_eq!(decoded.len(), total);
        }
    }
}
