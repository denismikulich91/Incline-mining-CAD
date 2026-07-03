# Vulcan `.dgd.isis` Design Database Format

`.dgd.isis` files store Vulcan design geometry. Incline currently reads the
coordinate records needed to reconstruct linework and imports them as `.pidb`
layers and polylines.

The reader supports both raw data and `vulZ`-wrapped compressed data. A
compressed file is identified by the same 8-byte magic used by compressed
`.00t` files:

```text
ea fb a7 8a 76 75 6c 5a
```

Compressed files are decompressed first, then scanned as raw ISIS data.

## Coordinate Records

Incline scans the raw stream for Vulcan `SEGCRD` coordinate records. Scanning
starts at offset `0x1000`; bytes before that are treated as file/header data.

Each coordinate record is 117 bytes:

| Relative offset | Size | Type | Description |
| --- | ---: | --- | --- |
| `0x00` | 1 byte | byte | Record marker. Expected value: `0x05`. |
| `0x01` | 3 bytes | bytes | Padding/signature. Expected value: `20 20 20`. |
| `0x04` | 1 byte | ASCII digit | Segment type. `0` starts a new feature; `1` or higher continues it. |
| `0x05` | 8 bytes | `f64` | X coordinate, big-endian. |
| `0x0d` | 8 bytes | `f64` | Y coordinate, big-endian. |
| `0x15` | 8 bytes | `f64` | Z coordinate, big-endian. |
| `0x1d` | 8 bytes | bytes | Extra record data. Currently ignored. |
| `0x25` | 40 bytes | text | Layer/segment name, space-padded. |
| `0x4d` | 40 bytes | text | Secondary name/attribute field. Used as a fallback layer name when the first name is generated. |

Only records with finite and plausible coordinates are accepted:

- `abs(x)` must be greater than `100.0` and less than `1e8`
- `abs(y)` must be greater than `100.0` and less than `1e8`
- `abs(z)` must be less than `50,000.0`

The name field is decoded as printable ASCII, stops at the first NUL byte, and
is trimmed of trailing spaces. Non-printable bytes are replaced with `?`.

## Segment Reconstruction

Each decoded coordinate becomes a `DesignPoint`:

```text
offset, name, secondary_name, seg_type, geometry_kind, x, y, z
```

Records are processed in file order. A point with `seg_type == 0` starts a new
segment, completing the previous segment first. A non-coordinate gap between
two accepted records also starts a new segment, even when the second record's
`seg_type` says continuation; those gaps usually contain DGD object/header data
and continuing through them fabricates long lines across unrelated objects.
Contiguous points with `seg_type > 0` are appended to the current segment even
when their point-name field changes. This matters for files that name vertices
`POINT_1`, `POINT_2`, etc.; grouping by that field connects unrelated vertices
into a web.

When a nearby object header identifies a coordinate run as `POLYPOINT`, or the
resolved layer name is a known point-collection name such as `POINTS`,
`REFERENCE_POINTS`, or `*_PTS`, the run is imported as individual point
objects instead of a polyline.

At end of file, any still-open polylines are completed.

## ISIX Sidecar

When importing `foo.dgd.isis`, Incline automatically looks for
`foo.dgd.isix` in the same directory, including a case-insensitive filename
fallback for sidecars with uppercase/mixed-case extensions. The sidecar is
optional; missing sidecars fall back to the decoded segment-header names
described above.

Observed `.dgd.isix` files store 48-byte index entries, usually beginning at
`0x400`:

| Relative offset | Size | Type | Description |
| --- | ---: | --- | --- |
| `0x00` | 4 bytes | `u32` | Big-endian offset into the decompressed ISIS stream. |
| `0x04` | 4 bytes | bytes | Marker. Observed value: `ff ff ff ff`. |
| `0x08` | 40 bytes | text | Space-padded layer/object name. |

Incline scans for these entries rather than relying on one fixed table start.
Entries are sorted by their ISIS offset. For each reconstructed segment whose
record names are generated labels, Incline uses the nearest sidecar entry at
the segment-header offset as the PIDB layer name. This avoids treating generated
vertex labels such as `POINT_1`, `POINT_2`, etc. as layers when Vulcan layer
names are available in the sidecar.

## PIDB Import

When imported as a `.pidb`, layer names are chosen in this order:

1. the segment header's first name, when meaningful
2. the automatic `.dgd.isix` sidecar name, when available
3. the segment header's secondary name, when meaningful
4. `DGD Import`

Empty names, generated point labels such as `POINT_123`, and scale labels such
as `1:1250` are not treated as meaningful layer names. Numeric bookkeeping
fields such as `0         6` are ignored too.

- layer name: decoded segment header name, or `DGD Import`
- layer color: white
- layer visibility: enabled

Each reconstructed segment becomes individual point objects when DGD identifies
it as point-like. Otherwise it becomes an open polyline when it has more than
one vertex, or a point object when it has exactly one vertex:

- vertices use the decoded XYZ coordinates
- `closed` is `false`
- object color is `ByLayer`
- fill style is `Clear`
- line weight is `1.0`

The import does not currently preserve ISIS styling, closed-polygon state, or
fill information.

## `vulZ` Compression Wrapper

For compressed `.dgd.isis` files, Incline reuses the same `vulZ` page decoder
used for compressed `.00t` files: FastLZ level 1/2 pages, variable observed
page sizes, and initial or mid-stream 0x800-byte TOC chains.

Some observed DGD files advertise an expanded byte count at offset `0x20` that
is larger than the compressed page runs physically present in the file. For
ISIS only, Incline accepts a clean end-of-file trailer after complete page runs
and scans the decompressed bytes that are present. Strict `.00t` decoding still
requires the advertised expanded length.

After decompression, the expanded byte stream is scanned for coordinate records
using the layout above.
