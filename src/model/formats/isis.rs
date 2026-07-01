use std::{error::Error, fmt, fs, io, path::Path};

use super::tri00t::{VULZ_MAGIC, decode_vulz_bytes};

// ── Public types ──────────────────────────────────────────────────────────────

/// A single coordinate record from a Vulcan `.dgd.isis` design database.
///
/// Records with `seg_type == 0` are segment-header points (start of a new
/// polyline/polygon). Records with `seg_type == 1` are continuation points.
/// Group consecutive records by name and break on `seg_type == 0` to
/// reconstruct individual polylines.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignPoint {
    /// Layer / segment name (trimmed of trailing whitespace).
    pub name: String,
    /// 0 = segment header (first point of a new feature), 1 = continuation.
    pub seg_type: u8,
    pub x: f64,
    pub y: f64,
    pub z: f64,
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
/// Use `seg_type == 0` records as segment boundaries: a new polyline starts
/// whenever `seg_type` transitions back to 0.
pub fn read_dgd_points(path: impl AsRef<Path>) -> Result<Vec<DesignPoint>, IsisError> {
    let bytes = fs::read(path).map_err(IsisError::Io)?;
    read_dgd_points_bytes(&bytes)
}

pub fn read_dgd_points_bytes(bytes: &[u8]) -> Result<Vec<DesignPoint>, IsisError> {
    let data = decompress_if_vulz(bytes)?;
    Ok(scan_dgd_points(&data))
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn decompress_if_vulz(bytes: &[u8]) -> Result<Vec<u8>, IsisError> {
    if bytes.starts_with(VULZ_MAGIC) {
        decode_vulz_bytes(bytes).map_err(|e| IsisError::Decompress(e.to_string()))
    } else {
        Ok(bytes.to_vec())
    }
}

/// Scan a raw (decompressed) DGD ISIS stream for SEGCRD coordinate records.
///
/// Record layout (117 bytes each):
/// ```text
/// [0x05]  [0x20 0x20 0x20 digit]  [X:f64be]  [Y:f64be]  [Z:f64be]
/// [8 bytes extra]  [name0: 40 bytes space-padded]  [name1: 40 bytes space-padded]
/// ```
/// `digit` is ASCII '0' for segment-header, '1' (or higher) for continuation.
fn scan_dgd_points(data: &[u8]) -> Vec<DesignPoint> {
    const RECORD_LEN: usize = 117;
    const MIN_SCAN_OFFSET: usize = 0x1000;
    const COORD_OFFSET: usize = 5;
    const NAME_OFFSET: usize = 37;
    const NAME_LEN: usize = 40;

    let mut out = Vec::new();

    let limit = data.len().saturating_sub(RECORD_LEN);
    let mut i = MIN_SCAN_OFFSET;

    while i < limit {
        if data[i] == 0x05
            && data[i + 1] == 0x20
            && data[i + 2] == 0x20
            && data[i + 3] == 0x20
            && data[i + 4].is_ascii_digit()
            && let Some((x, y, z)) = try_read_xyz(data, i + COORD_OFFSET)
            && is_plausible_coord(x, y, z)
        {
            let seg_type = data[i + 4] - b'0';
            let name = decode_name(&data[i + NAME_OFFSET..i + NAME_OFFSET + NAME_LEN]);
            out.push(DesignPoint {
                name,
                seg_type,
                x,
                y,
                z,
            });
            i += RECORD_LEN;
            continue;
        }
        i += 1;
    }

    out
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
