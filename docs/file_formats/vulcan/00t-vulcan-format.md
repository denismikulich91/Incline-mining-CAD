# Vulcan `.00t` Triangulation Format

`.00t` files store Vulcan triangulations: triangle meshes made from 3D vertices
and triangular faces. Incline supports two variants:

- raw `.00t` data
- `vulZ`-wrapped compressed data, identified by the first 8 bytes:
  `ea fb a7 8a 76 75 6c 5a`

Compressed `vulZ` files are decompressed first, then interpreted using the same
raw layout described below.

## Raw Layout

All numeric fields in the raw mesh payload are big-endian.

| Offset | Size | Type | Description |
| --- | ---: | --- | --- |
| `0x00` | 120 bytes | bytes | Header. Incline preserves this when possible. |
| `0x48` | 4 bytes | `u32` | Vertex count. |
| `0x60` | 4 bytes | `u32` | Face count. |
| `0x78` | `24 * vertex_count` | vertices | Vertex table. |
| after vertices | `24 * face_count` | faces | Face table. |
| after faces | remaining bytes | bytes | Trailing attributes, preserved on write. |

## Vertices

Each vertex record is 24 bytes:

| Relative offset | Size | Type | Description |
| --- | ---: | --- | --- |
| `0x00` | 8 bytes | `f64` | X coordinate. |
| `0x08` | 8 bytes | `f64` | Y coordinate. |
| `0x10` | 8 bytes | `f64` | Z coordinate. |

Coordinates must be finite floating point values.

## Faces

Each face record is 24 bytes:

| Relative offset | Size | Type | Description |
| --- | ---: | --- | --- |
| `0x00` | 4 bytes | `u32` | First vertex index. |
| `0x04` | 4 bytes | `u32` | Second vertex index. |
| `0x08` | 4 bytes | `u32` | Third vertex index. |
| `0x0c` | 12 bytes | bytes | Unused/attribute bytes. Incline writes zeroes here. |

Incline accepts both zero-based and one-based face indices. The index base is
detected from the minimum and maximum face index:

- zero-based if the minimum index is `0` and the maximum is less than
  `vertex_count`
- one-based if the minimum index is at least `1` and the maximum is less than or
  equal to `vertex_count`

Files with indices outside either range are rejected.

## `vulZ` Compression Wrapper

A compressed `.00t` starts with the `vulZ` magic bytes above. The total expanded
raw length is stored as a little-endian `u32` at offset `0x20`.

The compressed stream is stored in FastLZ pages. Older samples use level 1 and
25,600-byte expanded pages; newer observed files can use FastLZ level 2 and a
different, but consistent, expanded page size. Incline locks onto the first
accepted page size, appends pages until the advertised total expanded length is
reconstructed, and keeps `.00t` decoding strict if the stream ends early.

Table-of-contents blocks are 0x800 bytes. A little-endian pointer at offset
`0x3c` points to the next block during the initial TOC chain; when that pointer
no longer equals the next 0x800-byte block, scanning begins for the first
compressed page. Large files can also insert linked TOC chains between page
runs; Incline skips those chains and resumes at the next compressed page run.

## Writing

When Incline writes `.00t` files it writes the raw, uncompressed form:

1. A 120-byte header, reusing the source header when available.
2. Updated vertex and face counts at `0x48` and `0x60`.
3. Big-endian `f64` vertices.
4. Big-endian zero-based face indices.
5. Twelve zero bytes after each face index triple.
6. Any trailing attributes preserved from the source file.

Newly-created triangulations use a zero-filled header except for the vertex and
face counts.
