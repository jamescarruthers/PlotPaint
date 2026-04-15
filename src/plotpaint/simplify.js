// 3D Ramer-Douglas-Peucker simplification, a stand-in for the Python
// `simplify5d.simplify(points, tol, highQuality)` call. Operates on the
// first three coordinates of each point and preserves the full point.

function distToSegmentSquared(p, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = (b[2] ?? 0) - (a[2] ?? 0);
  const lenSq = dx * dx + dy * dy + dz * dz;
  if (lenSq === 0) {
    const ex = p[0] - a[0];
    const ey = p[1] - a[1];
    const ez = (p[2] ?? 0) - (a[2] ?? 0);
    return ex * ex + ey * ey + ez * ez;
  }
  let t =
    ((p[0] - a[0]) * dx +
      (p[1] - a[1]) * dy +
      ((p[2] ?? 0) - (a[2] ?? 0)) * dz) /
    lenSq;
  t = Math.max(0, Math.min(1, t));
  const px = a[0] + t * dx;
  const py = a[1] + t * dy;
  const pz = (a[2] ?? 0) + t * dz;
  const ex = p[0] - px;
  const ey = p[1] - py;
  const ez = (p[2] ?? 0) - pz;
  return ex * ex + ey * ey + ez * ez;
}

// Radial distance pre-pass (cheap) — drops consecutive near-duplicates.
function simplifyRadialDistance(points, tolSq) {
  if (points.length < 2) return points.slice();
  const out = [points[0]];
  let prev = points[0];
  for (let i = 1; i < points.length; i++) {
    const p = points[i];
    const dx = p[0] - prev[0];
    const dy = p[1] - prev[1];
    const dz = (p[2] ?? 0) - (prev[2] ?? 0);
    if (dx * dx + dy * dy + dz * dz > tolSq) {
      out.push(p);
      prev = p;
    }
  }
  if (prev !== points[points.length - 1]) out.push(points[points.length - 1]);
  return out;
}

// Iterative Douglas-Peucker, 3D.
function simplifyDouglasPeucker(points, tolSq) {
  const n = points.length;
  if (n < 3) return points.slice();
  const keep = new Uint8Array(n);
  keep[0] = 1;
  keep[n - 1] = 1;

  const stack = [[0, n - 1]];
  while (stack.length) {
    const [first, last] = stack.pop();
    let maxD = 0;
    let index = -1;
    for (let i = first + 1; i < last; i++) {
      const d = distToSegmentSquared(points[i], points[first], points[last]);
      if (d > maxD) {
        maxD = d;
        index = i;
      }
    }
    if (maxD > tolSq && index !== -1) {
      keep[index] = 1;
      stack.push([first, index]);
      stack.push([index, last]);
    }
  }

  const out = [];
  for (let i = 0; i < n; i++) if (keep[i]) out.push(points[i]);
  return out;
}

export function simplify(points, tolerance = 1, highQuality = false) {
  if (points.length <= 2) return points.slice();
  const tolSq = tolerance * tolerance;
  const prepared = highQuality
    ? points.slice()
    : simplifyRadialDistance(points, tolSq);
  return simplifyDouglasPeucker(prepared, tolSq);
}

export default simplify;
