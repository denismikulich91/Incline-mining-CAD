# Vulcan `.bmf` / `.bdf` Block Model Format

Vulcan block models are stored as a pair of files:

- `.bdf`: an ASCII block definition file. It describes the model origin,
  schemas, variables, and construction rules.
- `.bmf`: a binary model file. It stores the per-block arrays and an embedded
  metadata copy that is more authoritative for the binary payload than the
  companion `.bdf`.

This document is based on the Thor tutorial files in
`test/thorData_forTutorial/`:

- `bm50x50.bdf` / `bm50x50.bmf`
- `high_grade_add.bdf` / `high_grade_add.bmf`
- `low_grade_add.bdf` / `low_grade_add.bmf`
- `pit.bdf` without a matching `.bmf`

The same files (plus `shawmodel_reg.bmf`, a `.bmf`-only sample from the Maptek
Python SDK examples) also live in `test/Vulcan/bmf_bdf/`. `shawmodel_reg.bmf`
is the source for the `double` and `int` variable types documented below,
which do not appear in the Thor tutorial files.

The decoded details below are observational. Unknown header fields and page
management fields are labelled as such.

## `.bdf` Definition File

`.bdf` files are plain text. They contain comments beginning with `*`, a
sequence of named definition sections, and a final `END$FILE`.

```text
BEGIN$DEF header
 bearing=60.000000000000
 n_schemas=2.000000000000
 n_variables=2.000000000000
 x_origin=77880.000000000000
 y_origin=4460.000000000000
 z_origin=-40.000000000000
END$DEF header
```

Sections use `key=value` fields. Quoted values are single-quoted. Numeric values
are written as decimal text, commonly with 12 decimal places.

### Header Section

The `header` section defines the model coordinate frame and counts:

| Key | Meaning |
| --- | --- |
| `bearing`, `dip`, `plunge` | Model orientation. In the `.bmf` metadata these appear as `orientation_3`, `orientation_1`, and `orientation_2` respectively in the observed files. Observed `bearing` is measured clockwise from north/world Y. |
| `n_schemas` | Number of `schema_N` sections. |
| `n_variables` | Number of `variable_N` sections in the `.bdf`. |
| `x_origin`, `y_origin`, `z_origin` | World coordinate origin for local block coordinates. |
| `NO_align_blocks` | Flag-style field with no value in the observed files. |

### Variable Sections

Each `variable_N` section describes a model attribute:

| Key | Meaning |
| --- | --- |
| `name` | Attribute name. |
| `type` | Logical type. Observed `.bdf` values are `float` and `name`. |
| `default` | Default/global value when no per-block data is stored. |
| `description` | Free text description. |

In `.bmf` metadata, `.bdf` `type='name'` becomes a dictionary-coded numeric
type such as `namedbyte` or `namedshort`, with `string_N` entries defining the
code-to-name mapping.

### Schema Sections

Each `schema_N` section describes a parent or sub-block grid:

| Key group | Meaning |
| --- | --- |
| `schema_min_x/y/z`, `schema_max_x/y/z` | Local model extents for the schema. |
| `block_min_x/y/z`, `block_max_x/y/z` | Minimum and maximum block dimensions. |
| `description` | Schema label, for example `PARENT` or `SUB`. |

The `.bmf` embedded metadata stores derived dimensions such as `dim_x`, `dim_y`,
and `dim_z`.

### Boundary, Limit, And Exception Sections

`boundary_N`, `limit_N`, and `exception_N` sections describe how the model was
constructed. Boundaries can reference Vulcan triangulations:

```text
BEGIN$DEF boundary_1
 projection='Y'
 triangulation='ore_tq1.00t'
 value='hi'
 variable='zone'
END$DEF boundary_1
```

These sections are useful provenance, but the `.bmf` file contains the resulting
per-block values.

## `.bmf` Binary File

All observed `.bmf` files start with the ASCII magic `TBMS2.0` followed by a
NUL byte. Numeric payload values are little-endian unless noted otherwise.

### File Header

The first `0x800` bytes are a file header.

| Offset | Size | Type | Observed meaning |
| --- | ---: | --- | --- |
| `0x00` | 8 bytes | bytes | Magic: `54 42 4d 53 32 2e 30 00` (`TBMS2.0\0`). |
| `0x0c` | 4 bytes | `u32le` | Observed value `1`. Purpose unknown. |
| `0x10` | 2 bytes | bytes | Observed value `08 08`, matching the `0x808` page stride. |
| `0x18` | 8 bytes | `u64le` | Pointer to a top-level page/table near the end of the file. |
| `0x28` | 8 bytes | `u64le` | Pointer to another top-level page/table. |
| `0x30` | 8 bytes | `u64le` | Logical end of allocated file data; equals physical file size in the Thor samples. |
| other | bytes | bytes | Zero in the observed files. |

The two top-level page pointers reference page tables and metadata pages. For
normal block-value reading, the embedded metadata `location` fields are the most
useful entry points.

### Paged Storage

The `0x800`-byte file header is followed by 8 reserved/zero-filled bytes
(observed zero in all sample files). The page array itself starts at `0x808`,
with each page using a `0x808` stride:

```text
8-byte page header
2048-byte page payload
```

The first two bytes of the page header identify the observed page kind:

| Page kind bytes | Meaning | Payload |
| --- | --- | --- |
| `00 02` | Text/metadata page | ASCII metadata fragment. |
| `00 03` | Double data page | 256 little-endian `f64` values. |
| `00 04` | Float data page | 512 little-endian `f32` values. |
| `00 05` | Long integer data page | 256 little-endian `u64` values. |
| `00 06` | Int data page | 512 little-endian `i32` values. |
| `00 09` | Named-byte data page | 2048 `u8` dictionary codes. |
| `00 0a` | Named-short data page | 1024 little-endian `u16` dictionary codes. |
| `01 01` | Leaf page table | Little-endian `u64` data-page offsets; zero entries can mean implicit default pages. |
| `02 01` | Parent page table | Little-endian `u64` child-table offsets; observed in large arrays. |

Page tables (`01 01`/`02 01`) hold 256 `u64` entries per page regardless of
the target variable's data type, since the table itself always stores
8-byte offsets.

Data pages store dense slices of a single variable. The variable's metadata
`location` points to a page table, not directly to values. Small arrays can use
a single `01 01` leaf table. Larger arrays can use a `02 01` parent table whose
entries point to `01 01` child tables.

Table position matters. A zero entry inside the page range needed for a
variable should be treated as an implicit page filled with the variable's
default/global value, not simply discarded. Zero entries beyond the required
page count are unused.

For example, `high_grade_add.bmf` stores `zone` as `namedshort`:

```text
"n_blocks" = 2837
"var_0" =
  "name" = "zone"
  "location" = 61680
  "string_0" = "none"
  "string_1" = "hi"
  "type" = "namedshort"
```

At file offset `61680` (`0x0000f0f0`) there is a leaf page table. Its first
three offsets point to `00 0a` pages. Each page stores 1024 `u16` codes, so
three pages are enough to store 2837 block values.

### Embedded Metadata

The `.bmf` contains text metadata pages using a brace-and-assignment syntax:

```text
{
 "created" = "Wed Sep 13 10:01:36 2017",
 "is_irregular" = 1,
 "n_blocks" = 2837,
 "n_schemas" = 2,
 "origin_x" = 77880,
 "origin_y" = 4460,
 "origin_z" = -40,
 ...
}
```

This metadata duplicates and extends the `.bdf`. It includes:

- `n_blocks`, the number of stored block records.
- Origin and orientation values.
- Schema dimensions and extents.
- Built-in coordinate variables such as `__lower_x` and `__upper_z`.
- User variables, their types, dictionary strings, defaults, global values, and
  storage locations.
- Voxel dimensions and extents.

The embedded metadata can differ from the `.bdf`. In `bm50x50.bdf`,
`n_variables=26`, but `bm50x50.bmf` includes additional stored variables named
`flag` and `zone`. For decoding the `.bmf` payload, prefer the embedded `.bmf`
metadata.

### Variable Records

Observed variable metadata fields:

| Field | Meaning |
| --- | --- |
| `name` | Variable name. |
| `type` | Physical storage type: `float`, `double`, `int`, `namedbyte`, `namedshort`, or `longlong` in the observed files. |
| `location` | Offset of the page table for per-block values. A value of `0` means there is no stored array. |
| `default` | Default value as text. |
| `global` | Global value used when `location=0`. |
| `string_N` | Dictionary label for named byte/short code `N`. |
| `read_only` | Flag; observed but not decoded beyond `0` in user variables. |
| `description` | Free text description. |

Built-in coordinate variables use the same mechanism:

- `__lower_x`, `__upper_x`
- `__lower_y`, `__upper_y`
- `__lower_z`, `__upper_z`

These arrays store local block bounds. Combine them with the model origin and
orientation metadata to place blocks in world coordinates. For the observed
Thor files, the XY placement transform is:

```text
angle = 90 degrees - bearing
world_x = origin_x + cos(angle) * local_x - sin(angle) * local_y
world_y = origin_y + sin(angle) * local_x + cos(angle) * local_y
world_z = origin_z + local_z
```

This places `high_grade_add.bmf` over its referenced `ore_tq1.00t`
triangulation. The dip/plunge contribution remains undocumented.

### Value Decoding

To decode a stored variable:

1. Read `n_blocks` from embedded metadata.
2. Find the variable by `name`.
3. If `location == 0`, use the variable's `global` or `default` value for every
   block.
4. Compute the number of value pages required from `n_blocks` and the type's
   values-per-page capacity.
5. Read the page table at `location`.
6. If the table is `02 01`, use table positions to visit child `01 01` tables.
   One parent-table slot covers 256 child-table slots in the observed files.
7. For each required leaf-table slot:
   - non-zero offset: read the pointed-to data page and decode its payload
   - zero offset: synthesize one page of the variable's global/default value
8. Concatenate values in table-slot order and truncate to `n_blocks`.
9. For named variables, map numeric codes through the `string_N` entries.

Observed capacities per data page:

| Type | Page kind | Values per page | Value encoding |
| --- | --- | ---: | --- |
| `double` | `00 03` | 256 | `f64le` |
| `float` | `00 04` | 512 | `f32le` |
| `longlong` | `00 05` | 256 | `u64le` |
| `int` | `00 06` | 512 | `i32le` |
| `namedbyte` | `00 09` | 2048 | `u8` dictionary code |
| `namedshort` | `00 0a` | 1024 | `u16le` dictionary code |

## Thor Sample Notes

`high_grade_add.bmf`:

- `n_blocks = 2837`
- `zone`: `namedshort`, codes `0=none`, `1=hi`
- `density`: `location=0`, global/default `-99.0`

`low_grade_add.bmf`:

- `n_blocks = 11076`
- `zone`: `namedshort`, codes `0=none`, `1=low`
- `density`: `location=0`, global/default `-99.0`

`bm50x50.bmf`:

- `n_blocks = 248948`
- user variables include `geology`, `density`, `au_krg`, `class`,
  `aui_a_cdist`, `aui_w_cdist`, `auk_a_cdist`, `auk_w_cdist`, `au_ids`,
  `flag`, and `zone`
- `geology` is `namedbyte` with codes for `delete`, `waste`, `tq1`, `tq1a`,
  `tq2`, and `air`
- several `.bdf` variables have `location=0` in the `.bmf`, meaning the model
  stores only a global/default value for those variables

`shawmodel_reg.bmf` (regular model, no `.bdf` companion in the sample set):

- `dim_x`/`dim_y`/`dim_z` = 87/73/60, no `n_blocks` field observed (regular
  models with `is_irregular = 0` omit it; block count is `dim_x * dim_y * dim_z`)
- `lens`: `namedshort` with codes `0=none`, `1=waste`, `2=bot_ore`, `3=se_ore`,
  `4=mid_ore`, `5=top_ore`
- `value0`/`value1`/`value2`: `double`, decoded from `00 03` pages
- `lg_pit`: `int`, decoded from `00 06` pages
- large arrays here use `02 01` parent tables over `01 01` leaf tables, unlike
  the smaller Thor samples which fit in a single leaf table

## Open Questions

The following fields are not fully decoded:

- The purpose of the `0x0c` header value.
- The full semantics of the top-level pointers at `0x18` and `0x28`.
- Free-list or allocation metadata outside the page tables.
- The exact 3D rotation transform implied by non-zero `dip` and `plunge`.

These unknowns do not prevent reading the observed block-value arrays.
