# Incline Tutorial
This tutorial is recommended for new users and contributors.

## Core ideas

### PIDBs

A PIDB, Project Items Database, holds information related to a project. It is suggested you create a new PIDB for each design. Its file extension is `.pidb`.

A PIDB can store many layers, each containing:

- points
- lines
- polylines (polygons)
- text labels
- roads
- colours and visibility settings

PIDBs do not hold Triangulations. A triangulation is usually a topology (surface) you use to assist in your design, by providing context, terrain, or a target for snapping and analysis.

Use `File > New .pidb` to create a new PIDB. Use `File > Open .pidb(s)` to open existing PIDBs. Use `File > Save All` or the save button in the top toolbar to save your edits.

### Layers

Layers are named containers inside a PIDB. Every editable object belongs to one
layer.

Layers help you separate types of work. For example, in a PIDB named `bravo_pit_designs.pidb`, you may have the following layers:

- `topology_contour_lines`
- `bravo_pit_polygons`
- `bravo_haul_roads`
- `windrow_designs`
- `water_bore_and_pipes`
- `eletrical_infrastructure`

The active layer is chosen from the `Layer` dropdown in the top toolbar. New objects are created on the active layer.

Layers can be loaded or unloaded from the Explorer. Loading a layer shows it in the scene and makes it available for editing. Unloading a layer hides it and can reduce the amount of geometry Incline has to draw. A layer with unsaved changes, refered to as a 'dirty-layer', is marked with `*`.

To make a new layer, click the `New layer` button on the left toolbar. To rename, save, load, unload, or delete a layer, use the layer's context menu in the Explorer by right-clicking the layer name.

Each layer can be exported as a DXF. Some types, like roads, do not exist in the DXF format, and so upon exporting they will get converted into polylines.

### Triangulations

A triangulation is a 3D mesh. In mine design software, triangulations are commonly used for:

- natural topography - often refered to as a surface, or a 'topo'/topology
- pit shells - the shape of the pit pit walls and floor
- ore bodies
- stockpile shells - the shape of a stockpile

Triangulations are shown in the `Triangulations` section of the Explorer. They are not layers inside a PIDB. They are separate surface files that can be loaded, hidden, exported, or used for surface operations. Incline has reverse engineered MapTek Vulcans .00t file format, and is therefore compatible in Vulcan based enviroments. Furthermore, Incline has support for .obj, .stl, .ply file formats.

Use `File > Import > As Triangulation ...` to import a triangulation file. Use `File > Add Triangulation Folder` to add a folder of triangulations. Use the `Triangulation` menu to create, cut, or contour triangulations.

### Points

A point is a single 3D position. Points are useful for:

- survey marks
- blast patterns
- water bores
- design reference locations

To create points:

1. Load or create a PIDB
2. Create a new layer with the new layer tool in the left toolbar
1. Choose an active layer from the layer selecter in the top toolbar.
2. Set the `Z` value in the top toolbar.
3. Select `Make Point` on the left toolbar.
4. Click in the viewport.
5. Right-Click, Esc, or Enter to finish the tool.

### Lines

A line is the simplest linear design object: a straight segment between two positions. In Incline, lines are stored as polylines with two vertices.

Lines are useful for:

- dig-to's
- simple boundaries

To create a line:

1. Choose an active layer.
2. Select `Make Line`.
3. Click the first point.
4. Click the second point.

Use `Cursor: Snap to point`, `Cursor: Snap to line`, or `Cursor: Snap to surface` when you need the line to connect exactly to existing geometry.

### Polylines and polygons

A polyline is a connected chain of vertices. It can be open or closed.

A closed polyline is refered to as a polygon and is useful for no-dig's, pit floor designs, and dump guidance.

In Incline, polylines can also preserve curved segments through DXF-style bulges. Closed polylines can have fill styles such as clear, crosses, slashes, or solid.

To create a polygon:

1. Choose an active layer.
2. Select `Make Polygon`.
3. Click each vertex around the boundary.
4. Finish the polygon when the shape is complete.

Useful editing tools for polylines and polygons include:

- `Offset` to create a parallel line or boundary
- `Auto-Bench` for bench/berm style offsets - useful for stockpile or pit designs
- `Relimit Line` to trim or extend a line against another
- `Fuse Lines Into Polygon` to turn connected lines into a closed polygon
- `Chamfer Polygon Corners` to bevel polygon corners
- `Bezier Polygon` to smooth or shape polygon edges
- `Explode Polygon to Lines` to split a polygon back into indiviudal lines

### Text

Text objects are labels placed at a 3D position. Use text for names, or design annotations.

To create text:

1. Choose the layer where labels should live.
2. Select `Make Text`.
3. Click where the label should be placed.
4. Enter the label content and properties when prompted.
5. Press the apply button or press Enter

### Roads

A road is a specialised design object unique to Incline. The properies of a road include:

- a centreline
- a width
- a camber angle (the shape that allows water to flow to the side of a road)
- a road shape

The available road shapes are:

- `Crown`: both edges drop away from the centreline. looks like ^
- `CrossFallRight`: left edge is high, right edge is low. looks like \
- `CrossFallLeft`: right edge is high, left edge is low. looks like /

To create a road:

1. Create or choose a layer such as `haul_roads`.
2. Select `Make Roads` in the left toolbar.
3. Draw the centreline by clicking along the route.
4. Set the road width, camber, and shape in the viewport dock.
5. Finish the road.

Use road objects when you want Incline to remember road design properties. Use plain lines or polylines when you only need simple geometry. The advantage of using a road type to create roads is the ease of appending new road designs to an existing network, or updating properties such as road-width.

## A basic workflow

### Step 1: Create a PIDB

Start Incline and choose `File > New .pidb`.

Save the new PIDB with a meaningful name, such as `training_design.pidb`. It is recomended you keep you PIDBs together in one directory.

### Step 2: Create working layers

Create a few layers:

- `pit_design`
- `haul_roads`
- `labels`
- `stockpile_design`

Select `pit_design` as the active layer.

### Step 3: Load a triangulation

Use `File > Import > As Triangulation ...` and choose a surface file. For this toutorial, load `tutorial/example_data/topology.obj`.

After loading the surface, use `Zoom to extents` on the right toolbar (looks like the earth).

### Step 4: Draw simple design geometry

With `pit_design` active: Select `Make Polygon` and draw a polygon on the topology using the Snap To Cursor button in the bottom toolbar.

Try the cursor modes in the bottom toolbar:

- `Cursor: Regular` uses normal placement, places at the Z value set in the top-toolbar.
- `Cursor: Snap to surface` snaps to the surface of a triangulation.
- `Cursor: Snap to line` snaps to linework, or triangulation edges.
- `Cursor: Snap to point` snaps to vertices or points.

Then use the Auto-Bench tool, select your pit polygon, and increase the benches value. You will see a projection of your benches. Benches are used in mining as a geotechnical safeguard by catching falling rock. select apply or press Enter.

### Step 5: Create a Pit Shell

use `Triangulations > Create Triangulation` and select all of the bench polygons. Change the generation setting from Surface to Solid.

### Step 6: Add a label

Switch the active layer to `labels`.

Use `Make Text` to add labels such as `Sierra Pit Floor`

Keeping labels on a separate layer lets you unload or hide them without touching the design geometry.

### Step 7: Create a road

Switch the active layer to `haul_roads`.

Use `Make Roads` to draw a haul road centreline. Use `Crown` for a road that drains away from the middle, or a cross-fall option when the road should drain to one side. You can snap the road to the pit floor using snap-to-surface. You can snap roads to exisiting roads to create T or L junctions using the snap-to-line tool. Try making a ramp going from the base of the pit floor, to the top surface.

### Step 8: Edit and inspect

Use these tools to refine the design:

- `Move` to reposition objects.
- `Offset` to create parallel design lines.
- `Auto-Bench` to build repeated bench/berm offsets.
- `Measure distance` to check spacing.
- `Hide selection`, `Freeze selection`, and `Reveal all elements` to control what is visible and selectable.
- `X-Ray Vision` to see through surfaces and dense geometry.

### Step 9: Save and export

Save your PIDB regularly.

To export your designs, use `File > Export > Layer to ...` when you only want one layer. Use `File > Export > .pidb to ...` when you want to export the full design database to another supported format, such as DXF.

## Recommended habits

- Put different design purposes on different layers. Dont design everything on one layer.
- Name layers clearly before drawing a lot of objects.
- Keep text labels separate from geometry.
- Use triangulations as reference surfaces.
- Use snap modes when exact placement matters.
- Export layers separately when sending focused design packages to another person or application.

## Quick glossary

`PIDB`: Incline's project database file.

`Layer`: A named container inside a PIDB. Every editable object belongs to one layer.

`Triangulation`: A 3D triangle mesh used to represent topologies.

`Point`: A single 3D coordinate.

`Line`: A straight two-vertex polyline.

`Polyline`: A connected chain of vertices. It may be open or closed.

`Polygon`: A closed polyline.

`Text`: A label or annotation placed in the design.

`Road`: A centreline-based design object with width, camber, and cross-section shape.

`Z`: The elevation used for drawing when you are not snapping to a surface.

## Shortcuts

`~ Key`: Samples the Z value of whatevers under the mouse and sets the working-Z-level to this value
`Enter`: Accept / Apply
`Del`: Delete Objects
`Esc`: Cancel
`Ctrl Z`: Undo
`Cntrl D`: Duplicate