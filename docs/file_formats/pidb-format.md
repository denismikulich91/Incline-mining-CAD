# PIDB Project Format

`.pidb` files are Incline project files. They are UTF-8 JSON documents produced
with pretty-printed `serde_json`. A PIDB stores editable design geometry:
layers, points, polylines/polygons, text, and roads.

PIDBs do not store triangulations. Triangulations are loaded and saved as
separate mesh files such as `.00t`, `.obj`, `.stl`, or `.ply`.

## Top-Level Object

```json
{
  "format_version": 2,
  "document": {},
  "metadata": {
    "name": "example.pidb"
  }
}
```

| Field | Type | Description |
| --- | --- | --- |
| `format_version` | `u32` | PIDB schema version. Current value is `2`. |
| `document` | object | Layers, objects, and ID counters. |
| `metadata.name` | string | Display name for the project. If empty on load, Incline falls back to the file name. |

Incline rejects files whose `format_version` is not `2`.

## Document

```json
{
  "layers": [],
  "objects": [],
  "next_layer_id": 0,
  "next_object_id": 0
}
```

| Field | Type | Description |
| --- | --- | --- |
| `layers` | array | Ordered list of layer records. |
| `objects` | array | Design objects. Each object references a layer by ID. |
| `next_layer_id` | `u64` | Next layer ID to allocate. |
| `next_object_id` | `u64` | Next object ID to allocate. |

`revision` is runtime-only editor bookkeeping and is not serialized.

## IDs

Layer and object IDs are Rust newtype wrappers around `u64`. In JSON fields,
they serialize as plain numbers:

```json
3
```

Internally, open projects may receive a runtime namespace in the high 32 bits so
multiple PIDBs can share one scene without ID collisions. Before saving, Incline
normalizes the namespace back to `0`, so on-disk IDs remain local to the file.

## Layers

```json
{
  "id": 0,
  "name": "pit_design",
  "color_index": 7,
  "color": [1.0, 1.0, 1.0, 1.0],
  "visible": true,
  "elevation": 0.0
}
```

| Field | Type | Description |
| --- | --- | --- |
| `id` | `LayerId` | Stable layer ID used by objects. |
| `name` | string | Layer name. |
| `color_index` | `u8` or `null` | Optional DXF/ACI color index. |
| `color` | `[f32; 4]` | Resolved RGBA color used for by-layer rendering. |
| `visible` | bool | Layer visibility. |
| `elevation` | `f32` | Layer elevation value. |

Colors are stored as linear RGBA floats.

## Object Encoding

Objects are internally a Rust enum. In JSON, serde uses externally tagged enum
encoding, so each object has exactly one variant key:

```json
{
  "Polyline": {
    "id": 0,
    "layer": 0,
    "verts": [],
    "closed": false,
    "color": "ByLayer",
    "fill": "Clear",
    "fill_color": null,
    "line_weight": 1.0
  }
}
```

All object variants contain an `id` and `layer`. The `layer` must refer to a
layer ID in `document.layers`.

## Object Colors

```json
"ByLayer"
```

or:

```json
{ "Fixed": [0.2, 0.4, 0.8, 1.0] }
```

`ByLayer` uses the owning layer's `color`. `Fixed` stores an explicit linear
RGBA value.

## Polyline Vertices

```json
{
  "pos": [1000.0, 2000.0, 350.0],
  "bulge": 0.0
}
```

`pos` is an XYZ coordinate. `bulge` uses the DXF arc convention:
`tan(included_angle / 4)`. A value of `0.0` is a straight segment.

## Object Variants

### Point

```json
{
  "Point": {
    "id": 0,
    "layer": 0,
    "pos": [1000.0, 2000.0, 350.0],
    "color": "ByLayer"
  }
}
```

### Polyline

```json
{
  "Polyline": {
    "id": 1,
    "layer": 0,
    "verts": [
      { "pos": [1000.0, 2000.0, 350.0], "bulge": 0.0 },
      { "pos": [1010.0, 2000.0, 350.0], "bulge": 0.0 }
    ],
    "closed": false,
    "color": "ByLayer",
    "fill": "Clear",
    "fill_color": null,
    "line_weight": 1.0
  }
}
```

A polyline with `closed: true` represents a polygon.

`fill` is one of:

- `Clear`
- `Crosses`
- `Slashes`
- `Solid`

For older files, missing `fill` defaults to `Clear`, missing `fill_color`
defaults to `null`, and missing `line_weight` defaults to `1.0`.

### Text

```json
{
  "Text": {
    "id": 2,
    "layer": 0,
    "pos": [1000.0, 2000.0, 350.0],
    "content": "Pit floor",
    "height": 2.5,
    "rotation": 0.0,
    "color": "ByLayer"
  }
}
```

`rotation` is stored in radians.

### Road

```json
{
  "Road": {
    "id": 3,
    "layer": 0,
    "color": "ByLayer",
    "centerline": [
      { "pos": [1000.0, 2000.0, 350.0], "bulge": 0.0 },
      { "pos": [1100.0, 2050.0, 345.0], "bulge": 0.0 }
    ],
    "width": 30.0,
    "camber_degrees": 3.0,
    "shape": "Crown"
  }
}
```

`shape` is one of:

- `Crown`
- `CrossFallRight`
- `CrossFallLeft`

Roads are native Incline objects. When exported to DXF, they are converted into
plain linework for the centerline and road edges.

## Minimal Example

```json
{
  "format_version": 2,
  "document": {
    "layers": [
      {
        "id": 0,
        "name": "0",
        "color_index": 7,
        "color": [1.0, 1.0, 1.0, 1.0],
        "visible": true,
        "elevation": 0.0
      }
    ],
    "objects": [
      {
        "Point": {
          "id": 0,
          "layer": 0,
          "pos": [1000.0, 2000.0, 350.0],
          "color": "ByLayer"
        }
      }
    ],
    "next_layer_id": 1,
    "next_object_id": 1
  },
  "metadata": {
    "name": "example.pidb"
  }
}
```
