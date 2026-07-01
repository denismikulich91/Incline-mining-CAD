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
| `0x4d` | 40 bytes | text | Additional name/attribute field. Currently ignored. |

Only records with finite and plausible coordinates are accepted:

- `abs(x)` must be greater than `100.0` and less than `1e8`
- `abs(y)` must be greater than `100.0` and less than `1e8`
- `abs(z)` must be less than `50,000.0`

The name field is decoded as printable ASCII, stops at the first NUL byte, and
is trimmed of trailing spaces. Non-printable bytes are replaced with `?`.

## Segment Reconstruction

Each decoded coordinate becomes a `DesignPoint`:

```text
name, seg_type, x, y, z
```

Records are processed in file order. Incline groups points by `name`, using the
name as the target PIDB layer. A point with `seg_type == 0` starts a new
polyline for that layer. Any open polyline for the same layer is completed
before the new one begins. Points with `seg_type > 0` are appended to the
current polyline for that layer.

At end of file, any still-open polylines are completed.

## PIDB Import

When imported as a `.pidb`, each distinct ISIS name becomes a PIDB layer:

- layer name: decoded ISIS name
- layer color: white
- layer visibility: enabled

Each reconstructed segment becomes an open polyline:

- vertices use the decoded XYZ coordinates
- `closed` is `false`
- object color is `ByLayer`
- fill style is `Clear`
- line weight is `1.0`

The import does not currently preserve ISIS styling, closed-polygon state, fill
information, or the second 40-byte name/attribute field.

## `vulZ` Compression Wrapper

For compressed `.dgd.isis` files, Incline reuses the same `vulZ` decoder used
for compressed `.00t` files. The wrapper stores the total expanded byte length
as a little-endian `u32` at offset `0x20`, followed by FastLZ level-1 pages.
After decompression, the expanded byte stream is scanned for coordinate records
using the layout above.
