// Line / polygon utilities, ported from npline.py.
// Points are plain [x, y] (or [x, y, z, ...]) arrays. Collections of
// points are arrays of these arrays.

const copyPoint = (p) => p.slice();
const nearlyEqual = (a, b, eps = 1e-12) => Math.abs(a - b) <= eps;

function pointsEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function sub(a, b) {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - (b[i] ?? 0);
  return out;
}

function add(a, b) {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] ?? 0);
  return out;
}

function norm(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

export const npline = {
  distance(coord1, coord2) {
    return norm(sub(coord1, coord2));
  },

  reducepoints(coords, tolerance) {
    if (!coords.length) return [];
    const filtered = [copyPoint(coords[0])];
    for (let i = 1; i < coords.length; i++) {
      const d = npline.distance(filtered[filtered.length - 1], coords[i]);
      if (d >= tolerance) filtered.push(copyPoint(coords[i]));
    }
    return filtered;
  },

  // Rotate an array of 2D points around an origin by `angle` degrees.
  rotate(points, angle, origin = [0, 0]) {
    const rad = (angle * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const [ox, oy] = origin;
    const out = new Array(points.length);
    for (let i = 0; i < points.length; i++) {
      const [x, y] = points[i];
      const tx = x - ox;
      const ty = y - oy;
      out[i] = [tx * cos - ty * sin + ox, tx * sin + ty * cos + oy];
    }
    return out;
  },

  // Scale points around an origin.
  scale(points, factors, origin) {
    const dims = points[0].length;
    if (!origin) origin = new Array(dims).fill(0);
    const out = new Array(points.length);
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const np = new Array(dims);
      for (let j = 0; j < dims; j++) {
        np[j] = (p[j] - (origin[j] ?? 0)) * factors[j] + (origin[j] ?? 0);
      }
      out[i] = np;
    }
    return out;
  },

  // Interpolate points evenly along a path at the given resolution.
  // Mirrors npline.py's numpy implementation: includes original vertices.
  interpolate(points, resolution) {
    const n = points.length;
    if (n < 2) return points.map(copyPoint);

    const cum = new Array(n);
    cum[0] = 0;
    for (let i = 1; i < n; i++) {
      const dx = points[i][0] - points[i - 1][0];
      const dy = points[i][1] - points[i - 1][1];
      cum[i] = cum[i - 1] + Math.hypot(dx, dy);
    }
    const totalLen = cum[n - 1];
    if (totalLen === 0) return points.map(copyPoint);

    const numPoints = Math.max(2, Math.floor(totalLen / resolution));
    const extraCount = Math.max(0, numPoints - n + 1);

    // linspace(0, total, extraCount) — but only if we actually add points
    const evenly = [];
    if (extraCount > 0) {
      if (extraCount === 1) {
        evenly.push(0);
      } else {
        for (let i = 0; i < extraCount; i++) {
          evenly.push((totalLen * i) / (extraCount - 1));
        }
      }
    }

    // Merge original cumulative distances with evenly spaced ones and dedupe.
    const merged = cum.concat(evenly).sort((a, b) => a - b);
    const unique = [];
    for (let i = 0; i < merged.length; i++) {
      if (i === 0 || merged[i] - merged[i - 1] > 1e-12) unique.push(merged[i]);
    }

    // Linear interpolation of x and y at each distance.
    const out = new Array(unique.length);
    let seg = 0;
    for (let i = 0; i < unique.length; i++) {
      const d = unique[i];
      while (seg < n - 2 && cum[seg + 1] < d) seg++;
      const d0 = cum[seg];
      const d1 = cum[seg + 1];
      const t = d1 === d0 ? 0 : (d - d0) / (d1 - d0);
      const p0 = points[seg];
      const p1 = points[seg + 1];
      out[i] = [p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t];
    }
    return out;
  },

  // Inward/outward polygon offsetting using vertex bisectors.
  // old_points is expected to be a closed polygon (last == first).
  offset(oldPoints, offset, outerCcw = 1) {
    // drop repeated closing vertex
    let pts = oldPoints.slice();
    if (pts.length > 1 && pointsEqual(pts[0], pts[pts.length - 1])) {
      pts = pts.slice(0, -1);
    }
    const numPoints = pts.length;
    const newPoints = new Array(numPoints);

    for (let curr = 0; curr < numPoints; curr++) {
      const prev = (curr + numPoints - 1) % numPoints;
      const next = (curr + 1) % numPoints;

      const vn = sub(pts[next], pts[curr]);
      const vnNorm = norm(vn);
      const vnn = vnNorm === 0 ? [0, 0] : [vn[0] / vnNorm, vn[1] / vnNorm];
      const nnnX = vnn[1];
      const nnnY = -vnn[0];

      const vp = sub(pts[curr], pts[prev]);
      const vpNorm = norm(vp);
      const vpn = vpNorm === 0 ? [0, 0] : [vp[0] / vpNorm, vp[1] / vpNorm];
      const npnX = vpn[1] * outerCcw;
      const npnY = -vpn[0] * outerCcw;

      const bisX = (nnnX + npnX) * outerCcw;
      const bisY = (nnnY + npnY) * outerCcw;

      const bisLength = Math.hypot(bisX, bisY);
      if (bisLength !== 0 && !Number.isNaN(bisLength)) {
        const bisn = [bisX / bisLength, bisY / bisLength];
        const cosine = nnnX * npnX + nnnY * npnY;
        const bislen = offset / Math.sqrt((1 + cosine) / 2);
        if (!Number.isNaN(bislen) && Number.isFinite(bislen)) {
          newPoints[curr] = [
            pts[curr][0] + bislen * bisn[0],
            pts[curr][1] + bislen * bisn[1]
          ];
        } else {
          newPoints[curr] = copyPoint(pts[curr]);
        }
      } else {
        newPoints[curr] = copyPoint(pts[curr]);
      }
    }

    newPoints.push(copyPoint(newPoints[0]));
    return newPoints;
  },

  // Repeatedly offset a polygon inward until it self-intersects, producing
  // a spiral-like list of offset rings. Matches the Python implementation:
  // each iteration offsets the original polygon by a growing cumulative
  // distance (not the previously-offset polygon).
  spiral(polygon, offset) {
    const rings = [polygon];
    let currentOffset = offset;
    // safety guard
    for (let i = 0; i < 10000; i++) {
      const offsetPoly = npline.offset(polygon, currentOffset);
      if (npline.isSelfIntersecting(offsetPoly)) break;
      rings.push(offsetPoly);
      currentOffset += offset;
    }
    return rings;
  },

  polygonArea(points) {
    let closed = points;
    if (
      closed.length > 1 &&
      pointsEqual(closed[0], closed[closed.length - 1])
    ) {
      closed = closed.slice(0, -1);
    }
    let sum = 0;
    const n = closed.length;
    for (let i = 0; i < n; i++) {
      const [x, y] = closed[i];
      const [x1, y1] = closed[(i - 1 + n) % n];
      sum += x * y1 - y * x1;
    }
    return 0.5 * sum;
  },

  doLinesIntersect(p1, p2, q1, q2) {
    const ccw = (A, B, C) =>
      (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0]);
    return (
      ccw(p1, q1, q2) !== ccw(p2, q1, q2) &&
      ccw(p1, p2, q1) !== ccw(p1, p2, q2)
    );
  },

  isSelfIntersecting(polygon) {
    let pts = polygon;
    if (pts.length > 1 && pointsEqual(pts[0], pts[pts.length - 1])) {
      pts = pts.slice(0, -1);
    }
    const n = pts.length;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (
          i === j ||
          (i + 1) % n === j ||
          i === (j + 1) % n
        )
          continue;
        const p1 = pts[i];
        const p2 = pts[(i + 1) % n];
        const q1 = pts[j];
        const q2 = pts[(j + 1) % n];
        if (npline.doLinesIntersect(p1, p2, q1, q2)) return true;
      }
    }
    return false;
  },

  validateOffset(polygonPoints, offset) {
    const newPts = npline.offset(polygonPoints, offset);
    if (npline.isSelfIntersecting(newPts)) return [false, newPts];
    return [true, newPts];
  },

  isSimplePolygon(vertices) {
    return !npline.isSelfIntersecting(vertices);
  },

  // Split a path into chunks that are each approximately `distance` mm long.
  // (matches the Python version: interpolate to 0.1 mm, then slice every
  // `distance * 10` samples.)
  splitPath(path, distance) {
    const step = Math.max(1, Math.floor(distance * 10));
    const dense = npline.interpolate(path, 0.1);
    const newPaths = [];
    for (let i = 0; i < dense.length; i += step) {
      newPaths.push(dense.slice(i, i + step));
    }
    return newPaths;
  }
};

// Snake_case aliases for API parity with the Python version.
npline.reduce_points = npline.reducepoints;
npline.polygon_area = npline.polygonArea;
npline.do_lines_intersect = npline.doLinesIntersect;
npline.is_self_intersecting = npline.isSelfIntersecting;
npline.validate_offset = npline.validateOffset;
npline.is_simple_polygon = npline.isSimplePolygon;
npline.split_path = npline.splitPath;

export default npline;
