use std::{collections::BTreeMap, error::Error, fmt, fs, path::Path, sync::Arc};

use glam::{DMat3, DVec3};

use crate::{model::block_model::BlockBounds, userspace_warn};

const FILE_HEADER_LEN: usize = 0x800;
const PAGE_STRIDE: usize = 0x808;
const PAGE_HEADER_LEN: usize = 8;
const PAGE_PAYLOAD_LEN: usize = 0x800;
const METADATA_PAGE_KIND: [u8; 2] = [0x00, 0x02];

#[derive(Clone, Debug)]
pub(crate) struct BmfModel {
    pub(crate) metadata: BmfMetadata,
    /// Precomputed from `metadata.orientation`, which is fixed at construction
    /// time. Computing bearing/dip/plunge trig here once avoids redoing it for
    /// every corner of every block on every bounds query.
    rotation: DMat3,
    bytes: Arc<BmfBytes>,
}

enum BmfBytes {
    _Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl fmt::Debug for BmfBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BmfBytes")
            .field("len", &self.as_ref().len())
            .finish_non_exhaustive()
    }
}

impl AsRef<[u8]> for BmfBytes {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::_Owned(bytes) => bytes,
            Self::Mapped(map) => map,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BmfMetadata {
    pub(crate) n_blocks: usize,
    pub(crate) origin: DVec3,
    pub(crate) orientation: DVec3,
    pub(crate) lower: DVec3,
    pub(crate) upper: DVec3,
    pub(crate) dims: [usize; 3],
    pub(crate) is_irregular: bool,
    pub(crate) schemas: Vec<BmfSchema>,
    pub(crate) variables: Vec<BmfVariable>,
    pub(crate) raw_top_level: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BmfSchema {
    pub(crate) name: String,
    pub(crate) lower: DVec3,
    pub(crate) upper: DVec3,
    pub(crate) dims: [usize; 3],
    pub(crate) min_size: DVec3,
    pub(crate) max_size: DVec3,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BmfVariable {
    pub(crate) name: String,
    pub(crate) physical_type: String,
    pub(crate) description: String,
    pub(crate) location: u64,
    pub(crate) default: String,
    pub(crate) global: String,
    pub(crate) strings: BTreeMap<u32, String>,
    pub(crate) special: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BdfDefinition {
    pub(crate) sections: Vec<BdfSection>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BdfSection {
    pub(crate) name: String,
    pub(crate) fields: BTreeMap<String, String>,
}

#[derive(Debug)]
pub(crate) enum BmfError {
    Io(std::io::Error),
    Invalid(String),
}

impl fmt::Display for BmfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Invalid(message) => write!(f, "{message}"),
        }
    }
}

impl Error for BmfError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<std::io::Error> for BmfError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl BmfModel {
    pub(crate) fn from_path(path: impl AsRef<Path>) -> Result<Self, BmfError> {
        let file = fs::File::open(path)?;
        // Mapping avoids eagerly reading multi-GB block models into a Vec.
        // The parser still sees a normal byte slice, but the OS pages data in
        // on demand as metadata, bounds, and selected variables are decoded.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_storage(BmfBytes::Mapped(map))
    }

    pub(crate) fn _from_bytes(bytes: Vec<u8>) -> Result<Self, BmfError> {
        Self::from_storage(BmfBytes::_Owned(bytes))
    }

    fn from_storage(bytes: BmfBytes) -> Result<Self, BmfError> {
        let bytes = Arc::new(bytes);
        let slice = bytes.as_ref().as_ref();
        if slice.len() < FILE_HEADER_LEN || !slice.starts_with(b"TBMS2.0\0") {
            return Err(BmfError::Invalid("not a Vulcan TBMS2.0 block model".into()));
        }
        let root = parse_bmf_metadata_root(slice)?;
        let metadata = BmfMetadata::from_node(&root)?;
        let rotation = compute_rotation_matrix(metadata.orientation);
        Ok(Self {
            metadata,
            rotation,
            bytes,
        })
    }

    pub(crate) fn variable(&self, name: &str) -> Option<&BmfVariable> {
        self.metadata.variables.iter().find(|var| var.name == name)
    }

    pub(crate) fn numeric_variables(&self) -> Vec<&BmfVariable> {
        self.metadata
            .variables
            .iter()
            .filter(|var| is_numeric_type(&var.physical_type))
            .collect()
    }

    pub(crate) fn numeric_values(&self, name: &str) -> Result<Vec<f64>, BmfError> {
        self.numeric_values_range(name, 0, self.metadata.n_blocks)
    }

    /// Values for blocks `start..end` only, decoding just the value pages
    /// covering that range. Lets a paged viewer show a slice of a large
    /// model without decoding (or holding) whole variables.
    pub(crate) fn numeric_values_range(
        &self,
        name: &str,
        start: usize,
        end: usize,
    ) -> Result<Vec<f64>, BmfError> {
        let variable = self
            .variable(name)
            .ok_or_else(|| BmfError::Invalid(format!("unknown block variable '{name}'")))?;
        if is_numeric_type(&variable.physical_type) {
            self.decode_numeric_variable(variable, start, end)
        } else {
            Err(BmfError::Invalid(format!(
                "variable '{}' is not numeric ({})",
                variable.name, variable.physical_type
            )))
        }
    }

    pub(crate) fn named_code_values(&self, name: &str) -> Result<Vec<u32>, BmfError> {
        let variable = self
            .variable(name)
            .ok_or_else(|| BmfError::Invalid(format!("unknown block variable '{name}'")))?;
        match variable.physical_type.as_str() {
            "namedbyte" | "namedshort" => self.decode_named_variable(variable),
            other => Err(BmfError::Invalid(format!(
                "variable '{}' is not named/categorical ({other})",
                variable.name
            ))),
        }
    }

    /// Variables whose `type` this reader doesn't know how to decode as
    /// either numeric or named/categorical. These are still listed in
    /// [`BmfMetadata::variables`] (and shown in the UI's variable table) but
    /// have no value column, so callers should surface this list rather than
    /// let it pass silently.
    pub(crate) fn unsupported_variables(&self) -> Vec<&BmfVariable> {
        self.metadata
            .variables
            .iter()
            .filter(|var| !Self::variable_type_supported(&var.physical_type))
            .collect()
    }

    pub(crate) fn variable_type_supported(physical_type: &str) -> bool {
        is_numeric_type(physical_type) || matches!(physical_type, "namedbyte" | "namedshort")
    }

    pub(crate) fn renderable_block_indices(&self) -> Result<Vec<usize>, BmfError> {
        let Some(variable) = self.empty_marker_variable() else {
            return Ok((0..self.metadata.n_blocks).collect());
        };
        let empty_codes: std::collections::HashSet<u32> = variable
            .strings
            .iter()
            .filter_map(|(&code, label)| is_empty_block_label(label).then_some(code))
            .collect();
        if empty_codes.is_empty() {
            return Ok((0..self.metadata.n_blocks).collect());
        }
        let codes = self.named_code_values(&variable.name)?;
        Ok(codes
            .into_iter()
            .enumerate()
            .filter_map(|(index, code)| (!empty_codes.contains(&code)).then_some(index))
            .collect())
    }

    pub(crate) fn block_bounds(&self) -> Result<Vec<BlockBounds>, BmfError> {
        if ![
            "__lower_x",
            "__lower_y",
            "__lower_z",
            "__upper_x",
            "__upper_y",
            "__upper_z",
        ]
        .into_iter()
        .all(|name| self.variable(name).is_some())
        {
            if self.metadata.is_irregular {
                // A sub-blocked (is_irregular) model without explicit per-block
                // bounds cannot be reconstructed as a uniform grid: doing so
                // would silently render every sub-block at parent-cell size in
                // the wrong place. Surface that instead of guessing.
                return Err(BmfError::Invalid(
                    "BMF is sub-blocked (is_irregular) but is missing explicit __lower_x/y/z \
                     and __upper_x/y/z block-bound variables; cannot reconstruct block geometry"
                        .into(),
                ));
            }
            return self.regular_block_bounds();
        }

        // Decode the six bound columns in bounded slices rather than holding
        // six full N-length f64 temporaries at once: on a multi-GB model the
        // whole-column form peaks at ~48 B × N of scratch on top of the output
        // `Vec<BlockBounds>`. One slice is ~48 MB of temporaries regardless of
        // model size.
        const BOUNDS_SLICE_BLOCKS: usize = 1 << 20;
        let n = self.metadata.n_blocks;
        let mut blocks = Vec::with_capacity(n);
        let mut start = 0;
        while start < n {
            let end = (start + BOUNDS_SLICE_BLOCKS).min(n);
            let lower_x = self.numeric_values_range("__lower_x", start, end)?;
            let lower_y = self.numeric_values_range("__lower_y", start, end)?;
            let lower_z = self.numeric_values_range("__lower_z", start, end)?;
            let upper_x = self.numeric_values_range("__upper_x", start, end)?;
            let upper_y = self.numeric_values_range("__upper_y", start, end)?;
            let upper_z = self.numeric_values_range("__upper_z", start, end)?;
            let rows = lower_x
                .into_iter()
                .zip(lower_y)
                .zip(lower_z)
                .zip(upper_x)
                .zip(upper_y)
                .zip(upper_z);
            for (((((lx, ly), lz), ux), uy), uz) in rows {
                blocks.push(BlockBounds {
                    lower: DVec3::new(lx, ly, lz),
                    upper: DVec3::new(ux, uy, uz),
                });
            }
            start = end;
        }
        Ok(blocks)
    }

    fn regular_block_bounds(&self) -> Result<Vec<BlockBounds>, BmfError> {
        let [dim_x, dim_y, dim_z] = self.metadata.dims;
        if dim_x == 0 || dim_y == 0 || dim_z == 0 {
            return Err(BmfError::Invalid(
                "BMF has no block bounds variables or regular-grid dimensions".into(),
            ));
        }
        let expected = dim_x
            .checked_mul(dim_y)
            .and_then(|value| value.checked_mul(dim_z))
            .ok_or_else(|| BmfError::Invalid("BMF regular-grid dimensions overflow".into()))?;
        if expected != self.metadata.n_blocks {
            return Err(BmfError::Invalid(format!(
                "BMF regular-grid dimensions imply {expected} blocks, metadata reports {}",
                self.metadata.n_blocks
            )));
        }

        let cell = DVec3::new(
            (self.metadata.upper.x - self.metadata.lower.x) / dim_x as f64,
            (self.metadata.upper.y - self.metadata.lower.y) / dim_y as f64,
            (self.metadata.upper.z - self.metadata.lower.z) / dim_z as f64,
        );
        let mut blocks = Vec::with_capacity(self.metadata.n_blocks);
        for z in 0..dim_z {
            for y in 0..dim_y {
                for x in 0..dim_x {
                    let lower = self.metadata.lower
                        + DVec3::new(x as f64 * cell.x, y as f64 * cell.y, z as f64 * cell.z);
                    blocks.push(BlockBounds {
                        lower,
                        upper: lower + cell,
                    });
                }
            }
        }
        Ok(blocks)
    }

    pub(crate) fn local_to_world(&self, local: DVec3) -> DVec3 {
        self.metadata.origin + self.rotation * local
    }

    /// The model's local→world rotation and origin, so a renderer can push the
    /// affine placement (`origin + rotation * local`) to the GPU and expand
    /// axis-aligned local block bounds into oriented world boxes in a shader
    /// instead of pre-rotating every corner on the CPU. Orthonormal, so its
    /// inverse is its transpose.
    pub(crate) fn rotation(&self) -> DMat3 {
        self.rotation
    }

    pub(crate) fn origin(&self) -> DVec3 {
        self.metadata.origin
    }

    /// `true` when the model's dip/plunge are (approximately) zero, i.e. the
    /// bearing-only rotation this reader has verified against real Vulcan
    /// samples is sufficient. See [`compute_rotation_matrix`] for the caveat
    /// on tilted models.
    pub(crate) fn has_verified_rotation(&self) -> bool {
        const EPSILON: f64 = 1e-6;
        self.metadata.orientation.x.abs() < EPSILON && self.metadata.orientation.y.abs() < EPSILON
    }

    fn decode_numeric_variable(
        &self,
        variable: &BmfVariable,
        start: usize,
        end: usize,
    ) -> Result<Vec<f64>, BmfError> {
        let end = end.min(self.metadata.n_blocks);
        let start = start.min(end);
        if variable.location == 0 {
            let value = parse_default_f64(variable);
            return Ok(vec![value; end - start]);
        }

        let values_per_page = match variable.physical_type.as_str() {
            "float" => 512,
            "short" => 1024,
            "int" => 512,
            "longlong" => 256,
            "double" => 256,
            other => {
                return Err(BmfError::Invalid(format!(
                    "unsupported numeric storage type {other}"
                )));
            }
        };
        let first_page = start / values_per_page;
        let last_page = end.div_ceil(values_per_page);
        let page_offsets = self.value_page_offsets(variable.location, first_page..last_page)?;
        let mut values = Vec::with_capacity(end - start + values_per_page);
        for offset in page_offsets {
            if offset == 0 {
                values.extend(std::iter::repeat_n(
                    parse_default_f64(variable),
                    values_per_page,
                ));
                continue;
            }
            let payload = self.page_payload(offset)?;
            match variable.physical_type.as_str() {
                "float" => {
                    for chunk in payload.chunks_exact(4) {
                        values.push(f32::from_le_bytes(read_chunk(chunk)?) as f64);
                    }
                }
                "short" => {
                    for chunk in payload.chunks_exact(2) {
                        values.push(i16::from_le_bytes(read_chunk(chunk)?) as f64);
                    }
                }
                "int" => {
                    for chunk in payload.chunks_exact(4) {
                        values.push(i32::from_le_bytes(read_chunk(chunk)?) as f64);
                    }
                }
                "longlong" => {
                    for chunk in payload.chunks_exact(8) {
                        values.push(i64::from_le_bytes(read_chunk(chunk)?) as f64);
                    }
                }
                "double" => {
                    for chunk in payload.chunks_exact(8) {
                        values.push(f64::from_le_bytes(read_chunk(chunk)?));
                    }
                }
                _ => unreachable!(),
            }
        }
        // `values` starts at block `first_page * values_per_page`; trim to
        // the requested block range.
        let skip = start - first_page * values_per_page;
        values.drain(..skip.min(values.len()));
        values.truncate(end - start);
        Ok(values)
    }

    fn decode_named_variable(&self, variable: &BmfVariable) -> Result<Vec<u32>, BmfError> {
        if variable.location == 0 {
            let value = parse_default_code(variable);
            return Ok(vec![value; self.metadata.n_blocks]);
        }

        let values_per_page = match variable.physical_type.as_str() {
            "namedbyte" => 2048,
            "namedshort" => 1024,
            other => {
                return Err(BmfError::Invalid(format!(
                    "unsupported named storage type {other}"
                )));
            }
        };
        let required_pages = self.metadata.n_blocks.div_ceil(values_per_page);
        let page_offsets = self.value_page_offsets(variable.location, 0..required_pages)?;
        let mut values = Vec::with_capacity(self.metadata.n_blocks);
        for offset in page_offsets {
            if offset == 0 {
                values.extend(std::iter::repeat_n(
                    parse_default_code(variable),
                    values_per_page,
                ));
                continue;
            }
            let payload = self.page_payload(offset)?;
            match variable.physical_type.as_str() {
                "namedbyte" => values.extend(payload.iter().map(|&value| u32::from(value))),
                "namedshort" => {
                    for chunk in payload.chunks_exact(2) {
                        values.push(u32::from(u16::from_le_bytes(read_chunk(chunk)?)));
                    }
                }
                _ => unreachable!(),
            }
        }
        values.truncate(self.metadata.n_blocks);
        Ok(values)
    }

    fn empty_marker_variable(&self) -> Option<&BmfVariable> {
        self.metadata
            .variables
            .iter()
            .filter(|variable| {
                matches!(variable.physical_type.as_str(), "namedbyte" | "namedshort")
            })
            .filter(|variable| {
                variable
                    .strings
                    .values()
                    .any(|label| is_empty_block_label(label))
            })
            .max_by_key(|variable| {
                let name = variable.name.to_ascii_lowercase();
                usize::from(name == "geology" || name == "rock" || name == "material")
            })
    }

    /// Value-page file offsets for the pages in `pages`, in order. Only
    /// walks the child tables covering the requested range, so a small
    /// range on a huge model stays cheap.
    fn value_page_offsets(
        &self,
        table_offset: u64,
        pages: std::ops::Range<usize>,
    ) -> Result<Vec<u64>, BmfError> {
        let page = self.page(table_offset)?;
        let kind = &page[..2];
        let payload = &page[PAGE_HEADER_LEN..PAGE_HEADER_LEN + PAGE_PAYLOAD_LEN];
        match kind {
            [0x01, 0x01] => Ok(read_u64_slots(payload)?
                .into_iter()
                .take(pages.end)
                .skip(pages.start)
                .collect()),
            [0x02, 0x01] => {
                let child_tables = read_u64_slots(payload)?;
                let mut offsets = Vec::with_capacity(pages.len());
                for page_index in pages {
                    let child_index = page_index / 256;
                    let child_slot = page_index % 256;
                    let child_offset = child_tables.get(child_index).copied().unwrap_or(0);
                    if child_offset == 0 {
                        offsets.push(0);
                        continue;
                    }
                    let child = self.page(child_offset)?;
                    if child[..2] != [0x01, 0x01] {
                        return Err(BmfError::Invalid("expected BMF leaf page table".into()));
                    }
                    let slots = read_u64_slots(
                        &child[PAGE_HEADER_LEN..PAGE_HEADER_LEN + PAGE_PAYLOAD_LEN],
                    )?;
                    offsets.push(slots.get(child_slot).copied().unwrap_or(0));
                }
                Ok(offsets)
            }
            _ => Err(BmfError::Invalid(format!(
                "expected page table at offset {table_offset}"
            ))),
        }
    }

    fn page(&self, offset: u64) -> Result<&[u8], BmfError> {
        let offset = usize::try_from(offset)
            .map_err(|_| BmfError::Invalid("BMF page offset does not fit in memory".into()))?;
        let bytes = self.bytes.as_ref().as_ref();
        bytes
            .get(offset..offset + PAGE_STRIDE)
            .ok_or_else(|| BmfError::Invalid(format!("BMF page offset {offset} is outside file")))
    }

    fn page_payload(&self, offset: u64) -> Result<&[u8], BmfError> {
        let page = self.page(offset)?;
        Ok(&page[PAGE_HEADER_LEN..PAGE_HEADER_LEN + PAGE_PAYLOAD_LEN])
    }
}

impl BmfMetadata {
    fn from_node(node: &MetaNode) -> Result<Self, BmfError> {
        let object = node
            .as_object()
            .ok_or_else(|| BmfError::Invalid("BMF metadata root is not an object".into()))?;
        let dims = [
            object.usize("dim_x").unwrap_or(0),
            object.usize("dim_y").unwrap_or(0),
            object.usize("dim_z").unwrap_or(0),
        ];
        let inferred_blocks = dims
            .into_iter()
            .try_fold(1usize, |acc, value| {
                (value > 0).then(|| acc.checked_mul(value)).flatten()
            })
            .unwrap_or(0);
        let mut metadata = Self {
            n_blocks: object.usize("n_blocks").unwrap_or(inferred_blocks),
            origin: DVec3::new(
                object.f64("origin_x").unwrap_or(0.0),
                object.f64("origin_y").unwrap_or(0.0),
                object.f64("origin_z").unwrap_or(0.0),
            ),
            orientation: DVec3::new(
                object.f64("orientation_1").unwrap_or(0.0),
                object.f64("orientation_2").unwrap_or(0.0),
                object.f64("orientation_3").unwrap_or(0.0),
            ),
            lower: DVec3::new(
                object.f64("lower_x").unwrap_or(0.0),
                object.f64("lower_y").unwrap_or(0.0),
                object.f64("lower_z").unwrap_or(0.0),
            ),
            upper: DVec3::new(
                object.f64("upper_x").unwrap_or(0.0),
                object.f64("upper_y").unwrap_or(0.0),
                object.f64("upper_z").unwrap_or(0.0),
            ),
            dims,
            is_irregular: object.usize("is_irregular").unwrap_or(0) != 0,
            schemas: Vec::new(),
            variables: Vec::new(),
            raw_top_level: BTreeMap::new(),
        };
        for (key, value) in object {
            if let Some(text) = value.as_scalar() {
                metadata.raw_top_level.insert(key.clone(), text.clone());
            }
            if key.starts_with("schema_") {
                if let Some(schema) = BmfSchema::from_node(value) {
                    metadata.schemas.push(schema);
                }
            } else if (key.starts_with("var_") || key.starts_with("special_"))
                && let Some(mut variable) = BmfVariable::from_node(value)
            {
                variable.special = key.starts_with("special_");
                metadata.variables.push(variable);
            }
        }
        Ok(metadata)
    }
}

impl BmfSchema {
    fn from_node(node: &MetaNode) -> Option<Self> {
        let object = node.as_object()?;
        Some(Self {
            name: object.string("description").unwrap_or_default(),
            lower: DVec3::new(
                object.f64("lower_x").unwrap_or(0.0),
                object.f64("lower_y").unwrap_or(0.0),
                object.f64("lower_z").unwrap_or(0.0),
            ),
            upper: DVec3::new(
                object.f64("upper_x").unwrap_or(0.0),
                object.f64("upper_y").unwrap_or(0.0),
                object.f64("upper_z").unwrap_or(0.0),
            ),
            dims: [
                object.usize("dim_x").unwrap_or(0),
                object.usize("dim_y").unwrap_or(0),
                object.usize("dim_z").unwrap_or(0),
            ],
            min_size: DVec3::new(
                object.f64("min_size_x").unwrap_or(0.0),
                object.f64("min_size_y").unwrap_or(0.0),
                object.f64("min_size_z").unwrap_or(0.0),
            ),
            max_size: DVec3::new(
                object.f64("max_size_x").unwrap_or(0.0),
                object.f64("max_size_y").unwrap_or(0.0),
                object.f64("max_size_z").unwrap_or(0.0),
            ),
        })
    }
}

impl BmfVariable {
    fn from_node(node: &MetaNode) -> Option<Self> {
        let object = node.as_object()?;
        let mut strings = BTreeMap::new();
        for (key, value) in object {
            if let Some(index) = key
                .strip_prefix("string_")
                .and_then(|n| n.parse::<u32>().ok())
                && let Some(label) = value.as_scalar()
            {
                strings.insert(index, label.clone());
            }
        }
        Some(Self {
            name: object.string("name").unwrap_or_default().trim().to_owned(),
            // Vulcan writes some metadata strings space-padded to a fixed
            // width; normalize so type matching doesn't depend on padding or
            // case ("float " must read as float).
            physical_type: object
                .string("type")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase(),
            description: object.string("description").unwrap_or_default(),
            location: object.u64("location").unwrap_or(0),
            default: object.string("default").unwrap_or_default(),
            global: object.string("global").unwrap_or_default(),
            strings,
            special: false,
        })
    }
}

pub(crate) fn parse_bdf(path: impl AsRef<Path>) -> Result<BdfDefinition, BmfError> {
    let text = fs::read_to_string(path)?;
    let mut sections = Vec::new();
    let mut current: Option<BdfSection> = None;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('*') {
            continue;
        }
        if let Some(name) = line.strip_prefix("BEGIN$DEF") {
            current = Some(BdfSection {
                name: name.trim().to_owned(),
                fields: BTreeMap::new(),
            });
        } else if line.starts_with("END$DEF") {
            if let Some(section) = current.take() {
                sections.push(section);
            }
        } else if let Some(section) = current.as_mut() {
            if let Some((key, value)) = line.split_once('=') {
                section.fields.insert(
                    key.trim().to_owned(),
                    value.trim().trim_matches('\'').to_owned(),
                );
            } else {
                section.fields.insert(line.to_owned(), String::new());
            }
        }
    }
    Ok(BdfDefinition { sections })
}

pub(crate) fn same_stem_bdf_path(bmf_path: &Path) -> Option<std::path::PathBuf> {
    let mut path = bmf_path.to_path_buf();
    path.set_extension("bdf");
    path.is_file().then_some(path)
}

pub(crate) fn same_stem_bmf_path(bdf_path: &Path) -> Option<std::path::PathBuf> {
    let mut path = bdf_path.to_path_buf();
    path.set_extension("bmf");
    path.is_file().then_some(path)
}

const HEADER_PRIMARY_TABLE_POINTER: usize = 0x18;

/// A candidate metadata root text, plus the file offset of the last `00 02`
/// page it was built from (in text order). That offset lets callers check a
/// candidate against the file header's table pointer deterministically,
/// instead of only ranking candidates by how "complete" they look.
struct MetadataCandidate {
    text: String,
    /// The maximum file offset among the `00 02` pages this candidate is
    /// built from — i.e. the last page the allocator wrote for this text,
    /// regardless of which order the pages read correctly in.
    max_page_offset: usize,
    /// Number of pages in this candidate that start a root object. Candidates
    /// with multiple roots are useful for heuristic parsing, but pointer
    /// anchoring should prefer an unambiguous single-root candidate when two
    /// candidates end at the same page.
    root_starts: usize,
}

/// Vulcan orients a block model with three angles: bearing, dip, and
/// plunge, stored as `orientation_3`, `orientation_1`, and `orientation_2`
/// respectively. Bearing is measured clockwise from north/world Y about the
/// vertical (Z) axis; converting it to the usual mathematical angle
/// (counter-clockwise from east/world X) is confirmed against sample files.
/// Dip and plunge are applied here as rotations about the (post-bearing)
/// local X and Y axes, per Maptek's documented bearing/dip/plunge
/// convention. No sample file in this repo has non-zero dip/plunge, so that
/// part of the transform is unverified against real data;
/// [`BmfModel::has_verified_rotation`] reports when a model relies on it.
fn compute_rotation_matrix(orientation: DVec3) -> DMat3 {
    let bearing_angle = (90.0 - orientation.z).to_radians();
    let dip_angle = orientation.x.to_radians();
    let plunge_angle = orientation.y.to_radians();
    DMat3::from_rotation_z(bearing_angle)
        * DMat3::from_rotation_x(dip_angle)
        * DMat3::from_rotation_y(plunge_angle)
}

/// Read the metadata root, preferring the candidate whose last page sits
/// nearest before the file header's primary table pointer (offset `0x18`),
/// and falling back to heuristic scoring when that anchor is unavailable.
///
/// BMF files observed in this repo can contain multiple, fully-formed
/// metadata "root" objects left over from earlier incremental saves (e.g. a
/// pre-`index_model()` snapshot and a post-`index_model()` snapshot in the
/// same file, or a stale copy from a resize), and the pages making up the
/// *current* root are not always contiguous: appending a new variable can
/// leave its remaining text scattered behind unrelated data pages with no
/// link field to follow. What is reliable, verified against every `.bmf`
/// sample in `test/Vulcan/bmf_bdf/`, is that the primary table pointer at
/// `0x18` always points to a page table (`01 01`/`02 01`) that sits after the
/// last `00 02` page of the *live* root's text. In repo samples it sits
/// immediately after that page; company files have also been observed with
/// unrelated pages between the root and the table. Choosing the nearest parsed
/// root before the pointer is strictly more grounded than picking whichever
/// candidate happens to mention the most `var_`/`schema_` keys.
fn parse_bmf_metadata_root(bytes: &[u8]) -> Result<MetaNode, BmfError> {
    let candidates = extract_metadata_candidates(bytes)?;
    let pointer = header_primary_table_pointer(bytes);

    let mut pointer_anchor = None;
    let mut last_error = None;
    let mut best_root = None;
    let mut best_score = 0usize;
    let mut parsed_page_ends: Vec<usize> = Vec::new();
    for candidate in &candidates {
        let parsed = match parse_metadata_root(&candidate.text) {
            Ok(parsed) => parsed,
            Err(error) => {
                last_error = Some(error);
                continue;
            }
        };
        parsed_page_ends.push(candidate.max_page_offset);
        if let Some(pointer) = pointer
            && candidate.max_page_offset < pointer
        {
            let gap = pointer - candidate.max_page_offset;
            if gap % PAGE_STRIDE == 0 {
                let replace_anchor = match pointer_anchor.as_ref() {
                    Some((best_gap, best_root_starts, _)) => {
                        gap < *best_gap
                            || (gap == *best_gap && candidate.root_starts < *best_root_starts)
                    }
                    None => true,
                };
                if replace_anchor {
                    pointer_anchor = Some((gap, candidate.root_starts, parsed.clone()));
                }
            }
        }
        let score = metadata_root_score(&parsed);
        if score > best_score || best_root.is_none() {
            best_root = Some(parsed);
            best_score = score;
        }
    }

    if let Some((_, _, root)) = pointer_anchor {
        return Ok(root);
    }
    if let Some(pointer) = pointer {
        parsed_page_ends.sort_unstable();
        parsed_page_ends.dedup();
        let ends = parsed_page_ends
            .iter()
            .map(|offset| format!("0x{offset:x}"))
            .collect::<Vec<_>>()
            .join(", ");
        userspace_warn!(
            "BMF metadata root was not found where the file header's table pointer expected it \
             (pointer 0x{pointer:x}; parsed candidate roots end at [{ends}]); falling back to \
             heuristic candidate scoring, which may pick the wrong root in an atypical or \
             malformed file. If block values look wrong, please report these offsets."
        );
    }
    best_root.ok_or_else(|| {
        last_error.unwrap_or_else(|| BmfError::Invalid("could not parse BMF metadata root".into()))
    })
}

/// Reads the little-endian `u64` at file offset `0x18` and validates it
/// looks like a genuine page-table pointer (page-aligned, in range, and
/// pointing at a `01 01`/`02 01` page). Returns `None` rather than a
/// mismatched offset so callers only trust it when it's structurally sound.
fn header_primary_table_pointer(bytes: &[u8]) -> Option<usize> {
    let pointer = u64::from_le_bytes(
        bytes
            .get(HEADER_PRIMARY_TABLE_POINTER..HEADER_PRIMARY_TABLE_POINTER + 8)?
            .try_into()
            .ok()?,
    );
    let pointer = usize::try_from(pointer).ok()?;
    if pointer % PAGE_STRIDE != 0 || pointer < PAGE_STRIDE || pointer + PAGE_STRIDE > bytes.len() {
        return None;
    }
    match bytes.get(pointer..pointer + 2)? {
        [0x01, 0x01] | [0x02, 0x01] => Some(pointer),
        _ => None,
    }
}

fn extract_metadata_candidates(bytes: &[u8]) -> Result<Vec<MetadataCandidate>, BmfError> {
    let pages = collect_metadata_pages(bytes);
    let mut candidates = Vec::new();
    let mut current_run: Vec<&MetadataPage> = Vec::new();

    for page in &pages {
        if page.contiguous_with_previous {
            current_run.push(page);
        } else if !current_run.is_empty() {
            push_metadata_run_candidates(&current_run, &mut candidates);
            current_run.clear();
        }
        if !page.contiguous_with_previous {
            current_run.push(page);
        }
    }
    if !current_run.is_empty() {
        push_metadata_run_candidates(&current_run, &mut candidates);
    }

    push_threaded_metadata_candidates(&pages, &mut candidates);

    candidates.retain(|candidate| is_metadata_candidate(&candidate.text));
    candidates.sort_by_key(|candidate| {
        let trimmed = candidate.text.trim_start();
        (
            !trimmed.starts_with('{'),
            !candidate.text.contains("\"var_"),
            metadata_marker_index(&candidate.text),
        )
    });
    candidates.dedup_by(|a, b| a.text == b.text);
    if !candidates.is_empty() {
        return Ok(candidates);
    }

    Err(BmfError::Invalid(
        "BMF metadata pages were not found".into(),
    ))
}

#[derive(Clone, Debug)]
struct MetadataPage {
    text: String,
    offset: usize,
    starts_root: bool,
    contiguous_with_previous: bool,
}

fn collect_metadata_pages(bytes: &[u8]) -> Vec<MetadataPage> {
    let mut pages = Vec::new();
    let mut offset = FILE_HEADER_LEN + PAGE_HEADER_LEN;
    let mut previous_metadata_offset = None;
    while offset + PAGE_STRIDE <= bytes.len() {
        let page = &bytes[offset..offset + PAGE_STRIDE];
        if page[..2] == METADATA_PAGE_KIND {
            let payload_len =
                usize::from(u16::from_le_bytes([page[2], page[3]])).clamp(1, PAGE_PAYLOAD_LEN);
            let payload = &page[PAGE_HEADER_LEN..PAGE_HEADER_LEN + payload_len];
            let text = String::from_utf8_lossy(payload).replace('\0', "");
            let starts_root = text.trim_start().starts_with('{');
            let contiguous_with_previous =
                previous_metadata_offset.is_some_and(|previous| previous + PAGE_STRIDE == offset);
            pages.push(MetadataPage {
                text,
                offset,
                starts_root,
                contiguous_with_previous,
            });
            previous_metadata_offset = Some(offset);
        }
        offset += PAGE_STRIDE;
    }
    pages
}

fn push_threaded_metadata_candidates(
    pages: &[MetadataPage],
    candidates: &mut Vec<MetadataCandidate>,
) {
    for (index, page) in pages.iter().enumerate() {
        if !page.starts_root {
            continue;
        }
        // Materialize a candidate only at pages where a root object closes
        // (brace balance returns to zero), plus the full thread as a fallback.
        // The parser tolerates trailing text and the header table pointer
        // anchors on the page where the live root's text *ends*, so prefixes
        // that cut mid-object add no information — and cloning one per page
        // made this quadratic in the number of metadata pages.
        let mut text = page.text.clone();
        let mut max_page_offset = page.offset;
        let mut scanner = BraceScanner::default();
        scanner.feed(&page.text);
        candidates.push(MetadataCandidate {
            text: text.clone(),
            max_page_offset,
            root_starts: 1,
        });
        let mut pushed_up_to_date = true;
        for continuation in pages.iter().skip(index + 1) {
            if continuation.starts_root {
                continue;
            }
            text.push_str(&continuation.text);
            max_page_offset = max_page_offset.max(continuation.offset);
            pushed_up_to_date = false;
            if scanner.feed(&continuation.text) {
                candidates.push(MetadataCandidate {
                    text: text.clone(),
                    max_page_offset,
                    root_starts: 1,
                });
                pushed_up_to_date = true;
            }
        }
        if !pushed_up_to_date {
            candidates.push(MetadataCandidate {
                text,
                max_page_offset,
                root_starts: 1,
            });
        }
    }
}

/// Tracks brace depth across streamed text chunks, honouring the same string
/// and escape rules as `tokenize` so braces inside quoted values don't count.
#[derive(Default)]
struct BraceScanner {
    depth: u32,
    in_string: bool,
    escaped: bool,
}

impl BraceScanner {
    /// Feed a chunk; returns true if a top-level object closed (depth
    /// returned to zero) anywhere within it.
    fn feed(&mut self, text: &str) -> bool {
        let mut closed_root = false;
        for ch in text.chars() {
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if ch == '\\' {
                    self.escaped = true;
                } else if ch == '"' {
                    self.in_string = false;
                }
                continue;
            }
            match ch {
                '"' => self.in_string = true,
                '{' => self.depth += 1,
                '}' => {
                    self.depth = self.depth.saturating_sub(1);
                    if self.depth == 0 {
                        closed_root = true;
                    }
                }
                _ => {}
            }
        }
        closed_root
    }
}

fn is_metadata_candidate(text: &str) -> bool {
    text.contains("\"n_blocks\"")
        || (text.contains("\"dim_x\"") && text.contains("\"dim_y\"") && text.contains("\"dim_z\""))
}

fn metadata_marker_index(text: &str) -> usize {
    text.find("\"n_blocks\"")
        .or_else(|| text.find("\"dim_x\""))
        .unwrap_or(usize::MAX)
}

fn push_metadata_run_candidates(run: &[&MetadataPage], candidates: &mut Vec<MetadataCandidate>) {
    // `run` is always in ascending file-offset order (it's built by a
    // forward scan), so the maximum offset is the same for both the forward
    // and reverse text arrangement below.
    let max_page_offset = run.last().map_or(0, |page| page.offset);
    let root_starts = run.iter().filter(|page| page.starts_root).count().max(1);
    let forward = run
        .iter()
        .map(|page| page.text.as_str())
        .collect::<String>();
    if !forward.trim().is_empty() {
        candidates.push(MetadataCandidate {
            text: forward,
            max_page_offset,
            root_starts,
        });
    }
    if run.len() > 1 {
        let reverse = run
            .iter()
            .rev()
            .map(|page| page.text.as_str())
            .collect::<String>();
        if !reverse.trim().is_empty() {
            candidates.push(MetadataCandidate {
                text: reverse,
                max_page_offset,
                root_starts,
            });
        }
    }
}

fn read_u64_slots(payload: &[u8]) -> Result<Vec<u64>, BmfError> {
    payload
        .chunks_exact(8)
        .map(|chunk| Ok(u64::from_le_bytes(read_chunk(chunk)?)))
        .collect()
}

/// Converts a byte slice into a fixed-size array, returning a `BmfError`
/// instead of panicking if the slice is unexpectedly short. `chunks_exact`
/// callers can't hit this today (chunk length always matches `N`), but this
/// keeps numeric decoding from panicking if that invariant is ever broken by
/// a future refactor or an unexpectedly truncated page.
fn read_chunk<const N: usize>(chunk: &[u8]) -> Result<[u8; N], BmfError> {
    chunk
        .try_into()
        .map_err(|_| BmfError::Invalid("BMF value chunk has an unexpected length".into()))
}

fn parse_default_f64(variable: &BmfVariable) -> f64 {
    variable
        .global
        .trim()
        .parse::<f64>()
        .or_else(|_| variable.default.trim().parse::<f64>())
        .unwrap_or(0.0)
}

fn parse_default_code(variable: &BmfVariable) -> u32 {
    let text = if variable.global.trim().is_empty() {
        variable.default.trim()
    } else {
        variable.global.trim()
    };
    if let Ok(code) = text.parse::<u32>() {
        return code;
    }
    variable
        .strings
        .iter()
        .find_map(|(&code, label)| label.eq_ignore_ascii_case(text).then_some(code))
        .unwrap_or(0)
}

fn is_numeric_type(physical_type: &str) -> bool {
    matches!(
        physical_type,
        "float" | "short" | "int" | "longlong" | "double"
    )
}

fn is_empty_block_label(label: &str) -> bool {
    matches!(
        label.trim().to_ascii_lowercase().as_str(),
        "air" | "delete" | "deleted" | "void" | "empty" | "null"
    )
}

#[derive(Clone, Debug)]
enum MetaNode {
    Object(BTreeMap<String, MetaNode>),
    Scalar(String),
}

impl MetaNode {
    fn as_object(&self) -> Option<&BTreeMap<String, MetaNode>> {
        match self {
            Self::Object(object) => Some(object),
            Self::Scalar(_) => None,
        }
    }

    fn as_scalar(&self) -> Option<&String> {
        match self {
            Self::Scalar(value) => Some(value),
            Self::Object(_) => None,
        }
    }
}

trait MetaObjectExt {
    fn string(&self, key: &str) -> Option<String>;
    fn f64(&self, key: &str) -> Option<f64>;
    fn u64(&self, key: &str) -> Option<u64>;
    fn usize(&self, key: &str) -> Option<usize>;
}

impl MetaObjectExt for BTreeMap<String, MetaNode> {
    fn string(&self, key: &str) -> Option<String> {
        self.get(key).and_then(MetaNode::as_scalar).cloned()
    }

    fn f64(&self, key: &str) -> Option<f64> {
        self.string(key)?.trim().parse().ok()
    }

    fn u64(&self, key: &str) -> Option<u64> {
        self.string(key)?.trim().parse().ok()
    }

    fn usize(&self, key: &str) -> Option<usize> {
        self.string(key)?.trim().parse().ok()
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Open,
    Close,
    Equal,
    Comma,
    Value(String),
}

fn parse_metadata_root(text: &str) -> Result<MetaNode, BmfError> {
    let tokens = tokenize(text);
    let mut best = None;
    let mut best_score = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        if *token != Token::Open {
            continue;
        }
        let mut parser = Parser {
            tokens: &tokens,
            pos: index,
        };
        if let Ok(node) = parser.parse_object()
            && is_metadata_root(&node)
        {
            let score = metadata_root_score(&node);
            if score > best_score || best.is_none() {
                best = Some(node);
                best_score = score;
            }
        }
    }
    best.ok_or_else(|| BmfError::Invalid("could not parse BMF metadata root".into()))
}

fn is_metadata_root(node: &MetaNode) -> bool {
    node.as_object().is_some_and(|object| {
        object.contains_key("n_blocks")
            || (object.contains_key("dim_x")
                && object.contains_key("dim_y")
                && object.contains_key("dim_z"))
    })
}

fn metadata_root_score(node: &MetaNode) -> usize {
    let Some(object) = node.as_object() else {
        return 0;
    };
    let variables = object
        .keys()
        .filter(|key| key.starts_with("var_") || key.starts_with("special_"))
        .count();
    let schemas = object
        .keys()
        .filter(|key| key.starts_with("schema_"))
        .count();
    variables * 10
        + schemas * 3
        + usize::from(object.contains_key("n_blocks"))
        + usize::from(object.contains_key("dim_x"))
}

fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '{' => tokens.push(Token::Open),
            '}' => tokens.push(Token::Close),
            '=' => tokens.push(Token::Equal),
            ',' => tokens.push(Token::Comma),
            '"' => {
                let mut value = String::new();
                while let Some(next) = chars.next() {
                    match next {
                        '"' => break,
                        '\\' => {
                            if let Some(escaped) = chars.next() {
                                value.push(escaped);
                            }
                        }
                        _ => value.push(next),
                    }
                }
                tokens.push(Token::Value(value));
            }
            c if c.is_whitespace() => {}
            _ => {
                let mut value = String::from(ch);
                while let Some(&next) = chars.peek() {
                    if next.is_whitespace() || matches!(next, '{' | '}' | '=' | ',') {
                        break;
                    }
                    value.push(next);
                    chars.next();
                }
                tokens.push(Token::Value(value));
            }
        }
    }
    tokens
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl Parser<'_> {
    fn parse_object(&mut self) -> Result<MetaNode, BmfError> {
        self.expect_open()?;
        let mut object = BTreeMap::new();
        loop {
            self.skip_commas();
            if self.peek() == Some(&Token::Close) {
                self.pos += 1;
                break;
            }
            let key = self.take_value()?;
            self.expect_equal()?;
            let value = self.parse_value()?;
            object.insert(key, value);
            self.skip_commas();
        }
        Ok(MetaNode::Object(object))
    }

    fn parse_value(&mut self) -> Result<MetaNode, BmfError> {
        match self.peek() {
            Some(Token::Open) => self.parse_object(),
            Some(Token::Value(_)) => Ok(MetaNode::Scalar(self.take_value()?)),
            _ => Err(BmfError::Invalid("expected BMF metadata value".into())),
        }
    }

    fn expect_open(&mut self) -> Result<(), BmfError> {
        if self.peek() == Some(&Token::Open) {
            self.pos += 1;
            Ok(())
        } else {
            Err(BmfError::Invalid("expected '{'".into()))
        }
    }

    fn expect_equal(&mut self) -> Result<(), BmfError> {
        if self.peek() == Some(&Token::Equal) {
            self.pos += 1;
            Ok(())
        } else {
            Err(BmfError::Invalid("expected '='".into()))
        }
    }

    fn take_value(&mut self) -> Result<String, BmfError> {
        match self.tokens.get(self.pos) {
            Some(Token::Value(value)) => {
                self.pos += 1;
                Ok(value.clone())
            }
            _ => Err(BmfError::Invalid("expected metadata key/value".into())),
        }
    }

    fn skip_commas(&mut self) {
        while self.peek() == Some(&Token::Comma) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
}
