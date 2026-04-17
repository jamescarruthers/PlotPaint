# PlotPaint — Feature Specification

A browser-based tool for designing artwork for pen plotters. Users write
JavaScript in a code editor, see a live 3D preview, and export G-code
for their machine.

---

## 1. Application Layout

The app is a single-page browser application with a **split-pane layout**:

- **Left pane: Code editor** — a CodeMirror 6 JavaScript editor with
  syntax highlighting (one-dark theme) and monospace font.
- **Right pane: 3D preview** — a Three.js WebGL scene showing the
  painting strokes in 3D.
- **Draggable splitter** between the panes (vertical split on desktop,
  horizontal split on screens narrower than 720px).
- **Toolbar** above the editor with:
  - Title ("PlotPaint")
  - Keyboard shortcut hint (Ctrl/⌘+Enter)
  - **Run** button (green) — evaluates the code
  - **Download gcode** button — exports the current painting as a
    `.gcode` file download
- **Log panel** below the editor — shows `console.log` output in green
  and errors in red; auto-scrolls; clears before each run.

The colour scheme is dark: background #111, panels #181818, accent green
#2a6, error red #ff5a5a.

---

## 2. Interactive Editor

### Code execution model

User code runs inside an `async` function. The following identifiers are
provided in scope — **no import statements needed**:

| Name | Description |
|------|-------------|
| `painting` | A fresh `Plotpaint` instance, pre-created for you |
| `Plotpaint` | The class, for creating additional instances |
| `npline` | Geometry utility object |
| `poly` | Drawing helper — regular polygon |
| `line` | Drawing helper — centred line |
| `concfill` | Drawing helper — concentric fill spiral |
| `contractexpand` | Drawing helper — contract-then-expand spiral |
| `npCurves` | Easing curves object |
| `draw` | Object containing all drawing helpers |

After the code finishes, `painting.strokes` is automatically rendered in
the 3D preview.

### Keyboard shortcut

**Ctrl+Enter** (or ⌘+Enter on Mac) runs the code — equivalent to
clicking the Run button.

### Persistence

Code is automatically saved to `localStorage` on every keystroke and
restored on page load. First-time visitors see a default example.

### Error handling

Runtime errors are caught and displayed in the log panel with a stack
trace in red. The preview is not updated on error.

---

## 3. 3D Preview

The preview is a Three.js scene with orbit controls:

- **Drag** to orbit the camera.
- **Right-click drag** to pan.
- **Scroll** to zoom.
- Camera position is **preserved** across code runs (only reset on first
  load). The scene content (strokes, grid, page outline) is rebuilt each
  time.

### Visual elements

| Element | Description |
|---------|-------------|
| Page outline | White wireframe rectangle at page boundaries |
| Grid | XY plane grid centred on page, 20 divisions |
| Axes | RGB axes indicator at the origin (30 mm) |
| Stroke lines | Coloured lines (cycles through 10 distinct colours) |
| Stroke points | Small dots at each vertex |
| Start marker | Green sphere (1.2 mm) at the first point of each stroke |
| End marker | Red sphere (1.2 mm) at the last point of each stroke |

The preview resizes automatically with the splitter or window.

---

## 4. Plotpaint Core — Configurable Properties

All distances in millimetres, feed rates in mm/min.

### Page

| Property | Default | Description |
|----------|---------|-------------|
| `pageWidth` | 420 | Page width (A3) |
| `pageHeight` | 297 | Page height (A3) |

### Pen motion (run-in / run-out)

The **run-in** is extra motion *before* the stroke where the pen
descends from `zUp` to `zDown` — giving the brush a runway. The
**run-out** is the reverse at the end.

| Property | Default | Description |
|----------|---------|-------------|
| `runIn` | 20 | Run-in distance. Set to 0 to disable. |
| `runInEase` | `backOut` | Easing curve for Z descent during run-in |
| `runOut` | 20 | Run-out distance. Set to 0 to disable. |
| `runOutEase` | `backIn` | Easing curve for Z ascent during run-out |

For **closed paths** (first point == last point), run-in and run-out
wrap around the path itself rather than extending outward.

For **open paths**, run-in extends behind the start and run-out extends
behind the end (both rotated 180°).

### Feed rate

| Property | Default | Description |
|----------|---------|-------------|
| `feed` | 1000 | Feed rate at the start/end of a stroke |
| `feedMax` | 10000 | Peak feed rate in the middle of a stroke |
| `feedEase` | `sineInOut` | Easing curve controlling how feed varies across the stroke. Called with values spanning [0, 2] to produce a full ramp-up-then-down wave. |

### Machine Z heights

| Property | Default | Description |
|----------|---------|-------------|
| `zUp` | 20 | Z position when pen is lifted |
| `zDown` | 5 | Z position when pen is drawing |

### Path processing

| Property | Default | Description |
|----------|---------|-------------|
| `interpolate` | 0.05 | Resolution in mm for densifying paths |
| `simplifyVal` | 0.02 | Tolerance in mm for Ramer-Douglas-Peucker simplification |
| `pathOffset` | 0 | Fraction (0–1) to rotate the start point of closed paths |

### Callback hooks

Users can assign functions to post-process sections of each stroke.
Each receives an array of `[x, y, z]` points and must return the same
shape.

| Property | Description |
|----------|-------------|
| `processRunin` | Process run-in points after Z interpolation |
| `processLine` | Process the main line points |
| `processRunout` | Process run-out points after Z interpolation |

### G-code header

| Property | Default |
|----------|---------|
| `gcodeHeader` | `G17 G21 G90 G54 M3\n` |

### Methods

| Method | Description |
|--------|-------------|
| `addStroke(line)` | Process a polyline and add it to the painting. Returns the simplified stroke `[[x,y,z,f], ...]`. |
| `output(filename)` | Generate G-code for all strokes. Returns `{ filename, text }`. |
| `reset()` | Reset all properties to defaults and clear strokes. |

---

## 5. Drawing Helpers

### `poly(center, radius, startAngle, numPoints)`

Creates a **closed regular polygon** (last point == first).

- `center` — `[x, y]`
- `radius` — distance from centre to each vertex (mm)
- `startAngle` — rotation in degrees (0° = right)
- `numPoints` — number of vertices (e.g. 3 = triangle, 36 ≈ circle)

### `line(center, length, angle)`

Creates a **straight line** centred on a point.

- `center` — `[x, y]`
- `length` — total length (mm)
- `angle` — direction in degrees

### `concfill(polygon, distance)`

Generates a **concentric inward spiral** fill of a closed polygon.
Iteratively offsets the polygon inward by `distance` and traces a
continuous spiral path. Stops when the offset self-intersects or the
perimeter drops below 10 mm.

### `contractexpand(polygon, rotations, distance)`

Generates a spiral that **contracts inward then expands outward** from
the polygon boundary. Returns a single continuous path.

---

## 6. Geometry Utilities (`npline`)

All functions work with points as plain arrays (`[x, y]` or
`[x, y, z]`). Closed polygons have last point equal to first.

| Function | Description |
|----------|-------------|
| `npline.distance(a, b)` | Euclidean distance between two points |
| `npline.interpolate(points, resolution)` | Densify a polyline to a given resolution (mm). Preserves original vertices. |
| `npline.rotate(points, angleDeg, origin)` | Rotate 2D points around an origin |
| `npline.scale(points, factors, origin)` | Scale points by per-axis factors around an origin |
| `npline.offset(polygon, distance)` | Inward/outward polygon offsetting using vertex bisectors. Positive = outward, negative = inward. |
| `npline.spiral(polygon, offset)` | Repeatedly offset a polygon inward, returning an array of rings |
| `npline.splitPath(path, distance)` | Split a polyline into chunks of approximately `distance` mm |
| `npline.polygonArea(points)` | Signed area of a polygon (shoelace formula) |
| `npline.isSelfIntersecting(polygon)` | Test if a polygon has self-intersections |
| `npline.reducepoints(coords, tolerance)` | Drop points closer than `tolerance` to their predecessor |

---

## 7. Easing Curves (`npCurves`)

Each function accepts an array (or single number) and returns eased
values. Inputs are **not clamped** to [0, 1] — this matters for the
feed ease which spans [0, 2].

**Available curves** (each has In, Out, and InOut variants):

- `linear`
- `quad` — quadratic
- `cubic`
- `quart` — quartic
- `quint` — quintic
- `sine`
- `expo` — exponential
- `circ` — circular arc
- `back` — overshoot
- `elastic` — spring oscillation
- `bounce` — bouncing ball

Example names: `npCurves.backOut`, `npCurves.sineInOut`,
`npCurves.elasticIn`.

---

## 8. G-Code Output Schema

### Units and coordinate system

- All dimensions in **millimetres** (G21).
- **Absolute positioning** (G90).
- Y-axis is **flipped** in the output: `Y_gcode = pageHeight − Y_input`.
- Numeric precision: X/Y/Z to 2 decimal places, F (feed) to 0.

### Header

Default: `G17 G21 G90 G54 M3`

| Code | Meaning |
|------|---------|
| G17 | XY plane selection |
| G21 | Metric (millimetres) |
| G90 | Absolute positioning |
| G54 | Work coordinate offset 1 |
| M3 | Spindle on |

Customisable via `painting.gcodeHeader`.

### Per-stroke structure

Each stroke produces the following sequence:

```
G00 X... Y... Z{zUp}          ← rapid move to the stroke's first XY at pen-up height
G01 X... Y... Z... F...       ← run-in: Z ramps from zUp → zDown
G01 X... Y... Z... F...       ← main line: Z stays at zDown
G01 X... Y... Z... F...       ← run-out: Z ramps from zDown → zUp
G00 Z{zUp}                    ← rapid lift (Z only)
```

- **G00** — rapid (non-cutting) move. Used for travel between strokes
  and final Z lift.
- **G01** — linear interpolated (cutting) move with feed rate F.

### Feed rate profile

Feed varies across each stroke using the `feedEase` curve:
- Starts at `feed` (default 1000 mm/min)
- Peaks at `feedMax` (default 10000 mm/min) near the middle
- Returns to `feed` at the end

### Bounds warning

If any point falls outside the page boundaries (negative X/Y, or
exceeds pageWidth/pageHeight) or has negative Z, a console warning is
logged: "Plot may be off the page, contains negative coordinates".

---

## 9. Node CLI Export

A headless Node script at `scripts/export-gcode.js` runs the example
painting and writes the G-code to disk:

```
node scripts/export-gcode.js [output-file]
```

Defaults to `painting.gcode` if no file is given.

---

## 10. Build & Deploy

| Command | Description |
|---------|-------------|
| `npm run dev` | Vite dev server with hot reload |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Preview the production build locally |
| `npm run export` | Run the Node CLI to write gcode |

A **GitHub Actions workflow** (`.github/workflows/deploy.yml`) deploys
the production build to GitHub Pages on every push to `main`. The Vite
`base` is set to `/PlotPaint/` for production builds so asset URLs
resolve correctly under the repo's Pages subpath.
