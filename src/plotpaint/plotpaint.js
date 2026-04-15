// Core Plotpaint class, ported from plotpaint.py.

import { npline } from './npline.js';
import { npCurves } from './easing.js';
import { simplify } from './simplify.js';

// Linearly spaced values between start and stop (inclusive), `n` samples.
function linspace(start, stop, n) {
  if (n <= 0) return [];
  if (n === 1) return [start];
  const out = new Array(n);
  const step = (stop - start) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = start + step * i;
  return out;
}

// np.isin with all-columns-match, emulating the (xy, ez) match in Python.
function matchesRow(row, set, eps = 1e-9) {
  for (let i = 0; i < set.length; i++) {
    const r = set[i];
    if (
      Math.abs(r[0] - row[0]) <= eps &&
      Math.abs(r[1] - row[1]) <= eps &&
      Math.abs(r[2] - row[2]) <= eps
    )
      return true;
  }
  return false;
}

export class Plotpaint {
  constructor() {
    this.strokes = [];
    this.ease = npCurves;
    this.reset();
  }

  reset() {
    console.log('Reset to defaults');

    // default A3
    this.pageWidth = 420;
    this.pageHeight = 297;

    this.runIn = 20;
    this.runInEase = this.ease.backOut;
    this.runInAdj = 0;

    this.runOut = 20;
    this.runOutEase = this.ease.backIn;
    this.runOutAdj = 0;

    this.pathOffset = 0;
    this.feedEase = this.ease.sineInOut;

    this.interpolate = 0.05;
    this.simplifyVal = 0.02;

    this.processRunin = null;
    this.processLine = null;
    this.processRunout = null;

    // machine
    this.zUp = 20; // mm
    this.zDown = 5; // mm
    this.feed = 1000;
    this.feedMax = 10000;
    this.gcodeHeader = 'G17 G21 G90 G54 M3\n';
  }

  addStroke(line) {
    console.log('Interpolating... ');
    let t0 = performance.now();

    let xy = npline.interpolate(line, this.interpolate);

    const first = line[0];
    const last = line[line.length - 1];
    const closed =
      first.length === last.length &&
      first.every((v, i) => v === last[i]);

    let runIn;
    let runOut;

    if (closed) {
      if (this.pathOffset) {
        const numRows = xy.length;
        const shift = Math.round(this.pathOffset * numRows);
        xy = rollArray(xy, shift);
      }
      const outSteps = Math.max(1, Math.floor(this.runOut / this.interpolate));
      const inSteps = Math.max(1, Math.floor(this.runIn / this.interpolate));
      runIn = xy.slice(Math.max(0, xy.length - outSteps), xy.length - 1);
      runOut = xy.slice(1, inSteps);
    } else {
      const inSteps = Math.max(1, Math.floor(this.runIn / this.interpolate));
      const outSteps = Math.max(1, Math.floor(this.runOut / this.interpolate));
      runIn = xy.slice(1, inSteps);
      runOut = xy.slice(Math.max(0, xy.length - outSteps), xy.length - 1);

      runIn = npline.rotate(runIn, 180, xy[0]);
      runIn.reverse();
      runOut = npline.rotate(runOut, -180, xy[xy.length - 1]);
      runOut.reverse();
    }

    // build runIn with z
    {
      const ts = linspace(0, 1, runIn.length);
      const zs = this.runInEase(ts);
      const built = new Array(runIn.length);
      for (let i = 0; i < runIn.length; i++) {
        const z = this.zUp + (this.zDown - this.zUp) * zs[i];
        built[i] = [runIn[i][0], runIn[i][1], z];
      }
      runIn = built;
    }

    if (this.processRunin) runIn = this.processRunin(runIn);

    // the main line. In the Python source, this re-uses the raw input when
    // no processLine/pathOffset is set (faster for long paths).
    let lineXY = this.processLine || this.pathOffset ? xy : line;
    const mid = new Array(lineXY.length);
    for (let i = 0; i < lineXY.length; i++) {
      mid[i] = [lineXY[i][0], lineXY[i][1], this.zDown];
    }
    let mainLine = mid;
    if (this.processLine) mainLine = this.processLine(mainLine);

    {
      const ts = linspace(0, 1, runOut.length);
      const zs = this.runOutEase(ts);
      const built = new Array(runOut.length);
      for (let i = 0; i < runOut.length; i++) {
        const z = this.zDown + (this.zUp - this.zDown) * zs[i];
        built[i] = [runOut[i][0], runOut[i][1], z];
      }
      runOut = built;
    }
    if (this.processRunout) runOut = this.processRunout(runOut);

    let xyz = mainLine;
    if (this.runIn !== 0) xyz = runIn.concat(xyz);
    if (this.runOut !== 0) xyz = xyz.concat(runOut);

    console.log(`${Math.round(performance.now() - t0)} ms`);

    console.log('Simplifying...');
    t0 = performance.now();
    console.log(`Simplified from ${xyz.length} to `);
    const simpleLine = simplify(xyz, this.simplifyVal, true);
    console.log(`${simpleLine.length} points`);
    console.log(`${Math.round(performance.now() - t0)} ms`);

    // feed-rate curve across the original xyz
    const ts = linspace(0, 2, xyz.length);
    const fRaw = this.feedEase(ts);
    const fLine = new Array(xyz.length);
    for (let i = 0; i < xyz.length; i++) {
      const f = this.feed + (this.feedMax - this.feed) * fRaw[i];
      fLine[i] = [xyz[i][0], xyz[i][1], xyz[i][2], f];
    }

    // join feed back on to the simplified points by xyz equality
    const withFeed = new Array(simpleLine.length);
    // to stay fast, walk fLine with a pointer
    let p = 0;
    for (let i = 0; i < simpleLine.length; i++) {
      const s = simpleLine[i];
      // advance until we find a matching row (walks forward monotonically).
      while (
        p < fLine.length &&
        !(
          fLine[p][0] === s[0] &&
          fLine[p][1] === s[1] &&
          fLine[p][2] === s[2]
        )
      ) {
        p++;
      }
      if (p < fLine.length) {
        withFeed[i] = [s[0], s[1], s[2], fLine[p][3]];
      } else {
        // fall-back: linear search
        const idx = fLine.findIndex(
          (r) => r[0] === s[0] && r[1] === s[1] && r[2] === s[2]
        );
        withFeed[i] = [s[0], s[1], s[2], idx >= 0 ? fLine[idx][3] : this.feed];
        p = idx >= 0 ? idx : 0;
      }
    }

    this.strokes.push(withFeed);
    return withFeed;
  }

  output(filename) {
    console.log('Generating gcode');
    let out = this.gcodeHeader;
    for (const stroke of this.strokes) out += this.gcodePoints(stroke);
    // browser-friendly: caller writes the file. In Node there's
    // a helper in scripts/export-gcode.js.
    return { filename, text: out };
  }

  gcodePoints(points) {
    let out = '';
    let warning = false;

    const first = points[0];
    out += `G00 X${first[0].toFixed(2)} Y${(this.pageHeight - first[1]).toFixed(2)} Z${this.zUp}\n`;
    if (this.outsideBounds(first)) warning = true;

    for (const p of points) {
      out += `G01 X${p[0].toFixed(2)} Y${(this.pageHeight - p[1]).toFixed(2)} Z${p[2].toFixed(2)} F${p[3].toFixed(0)}\n`;
      if (this.outsideBounds(p)) warning = true;
    }

    out += `G00 Z${this.zUp}\n`;
    if (warning) {
      console.log('Plot may be off the page, contains negative coordinates');
    }
    return out;
  }

  outsideBounds(coord) {
    return (
      coord[0] < 0 ||
      coord[0] > this.pageWidth ||
      coord[1] < 0 ||
      coord[1] > this.pageHeight ||
      (coord[2] !== undefined && coord[2] < 0)
    );
  }
}

function rollArray(arr, n) {
  const len = arr.length;
  if (len === 0) return arr;
  const k = ((n % len) + len) % len;
  return arr.slice(len - k).concat(arr.slice(0, len - k));
}

export default Plotpaint;
