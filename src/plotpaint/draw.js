// Drawing helpers, ported from draw.py.
// The Python version leans on shapely.Polygon.buffer for contractexpand /
// concfill. This port uses npline.offset() as a lightweight stand-in —
// good enough for convex / lightly-concave polygons but not a full
// replacement for shapely.

import { npline } from './npline.js';

// Closed regular polygon centred on `center`.
export function poly(center, radius, startAngle, numPoints) {
  const coords = [];
  const step = (2 * Math.PI) / numPoints;
  const start = (startAngle * Math.PI) / 180;
  for (let i = 0; i < numPoints; i++) {
    const a = i * step + start;
    coords.push([
      center[0] + radius * Math.cos(a),
      center[1] + radius * Math.sin(a)
    ]);
  }
  coords.push(coords[0].slice());
  return coords;
}

// A line of `length`, centred on `center`, at `angle` degrees.
// Preserves the (small) quirk in the Python source that both endpoints are
// derived from center[0]; kept for bit-for-bit output parity.
export function line(center, length, angle) {
  const rad = ((angle - 90) * Math.PI) / 180;
  const radius = length / 2;
  const x0 = center[0] - radius * Math.cos(rad);
  const y0 = center[0] - radius * Math.sin(rad);
  const x1 = center[0] - radius * Math.cos(rad + Math.PI);
  const y1 = center[0] - radius * Math.sin(rad + Math.PI);
  return [
    [x0, y0],
    [x1, y1]
  ];
}

// Perimeter length of a (closed or open) polyline.
function perimeter(points) {
  let len = 0;
  for (let i = 1; i < points.length; i++) {
    const dx = points[i][0] - points[i - 1][0];
    const dy = points[i][1] - points[i - 1][1];
    len += Math.hypot(dx, dy);
  }
  return len;
}

// Point at arc-length `dist` along a closed polyline (wraps at end).
function interpolateAlong(points, dist) {
  const total = perimeter(points);
  if (total === 0) return points[0].slice();
  let d = ((dist % total) + total) % total;
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1];
    const [x1, y1] = points[i];
    const segLen = Math.hypot(x1 - x0, y1 - y0);
    if (d <= segLen || i === points.length - 1) {
      const t = segLen === 0 ? 0 : d / segLen;
      return [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t];
    }
    d -= segLen;
  }
  return points[points.length - 1].slice();
}

// Reorder a closed polygon so that it starts at the vertex closest to `pt`.
function reorderPolygon(polygon, pt) {
  // polygon is expected to be closed (last == first). Work on open ring.
  const ring = polygon.slice(0, -1);
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < ring.length; i++) {
    const dx = ring[i][0] - pt[0];
    const dy = ring[i][1] - pt[1];
    const d = dx * dx + dy * dy;
    if (d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  }
  const reordered = ring.slice(bestIdx).concat(ring.slice(0, bestIdx));
  reordered.push(reordered[0].slice());
  return reordered;
}

// Produce a spiral that contracts inward and then expands outward from
// `polygon`. Ported from draw.py — uses iterative offsetting in place of
// shapely.Polygon.buffer.
export function contractexpand(polygon, rotations, distance) {
  const inner = [];

  let d = 0;
  let r = 0;
  let current = polygon.slice();

  while (d < distance * rotations) {
    let inset = d === 0 ? current : npline.offset(polygon, -d);
    if (npline.isSelfIntersecting(inset)) break;
    const len = perimeter(inset);
    if (len < 10) break;

    const rotRes = len;
    const distStep = distance / rotRes;
    const rotStep = 1 / rotRes;

    if (d > 0 && inner.length) inset = reorderPolygon(inset, inner[0]);
    inner.push(interpolateAlong(inset, r * len));

    r += rotStep;
    if (r > 1) r = 0;
    d += distStep;
  }

  const outer = [];
  d = 0;
  r = 0;

  while (d < distance * rotations) {
    let inset = d === 0 ? polygon.slice() : npline.offset(polygon, d);
    if (npline.isSelfIntersecting(inset)) break;
    const len = perimeter(inset);
    if (len < 10) break;

    const rotRes = len * 2;
    const distStep = distance / rotRes;
    const rotStep = 1 / rotRes;

    outer.push(interpolateAlong(inset, r * len));

    r += rotStep;
    if (r > 1) r = 0;
    d += distStep;
  }

  outer.reverse();
  return outer.concat(inner);
}

// Concentric fill spiralling inward from the polygon boundary.
export function concfill(polygon, distance) {
  const newPoly = polygon.slice(0, -1).map((p) => p.slice());

  let d = 0;
  let r = 0;

  while (true) {
    const inset = npline.offset(polygon, -d);
    if (npline.isSelfIntersecting(inset)) break;
    const len = perimeter(inset);
    if (len < 10) break;

    const rotRes = len;
    const distStep = distance / rotRes;
    const rotStep = 1 / rotRes;

    const reordered = reorderPolygon(inset, newPoly[0]);
    const pt = interpolateAlong(reordered, (r - Math.floor(r)) * len);
    newPoly.push(pt);

    r += rotStep;
    d += distStep;
  }

  return newPoly;
}

export { reorderPolygon };
