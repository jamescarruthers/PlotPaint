// Headless gcode export — the Node equivalent of running example.py.
// Usage: node scripts/export-gcode.js [outfile]

import { writeFileSync } from 'node:fs';
import { Plotpaint, npline, poly } from '../src/plotpaint/index.js';

const painting = new Plotpaint();

painting.runIn = 0;
painting.runOut = 0;

const circle = poly([200, 200], 100, 0, 8);
const paths = npline.split_path(circle, 500);
for (const path of paths) painting.addStroke(path);

painting.runIn = 0;
painting.runOut = 0;
for (const path of paths) painting.addStroke(poly(path[0], 10, 0, 4));

console.log(`${paths.length} paths`);

const outfile = process.argv[2] || 'painting.gcode';
const { text } = painting.output(outfile);
writeFileSync(outfile, text);
console.log(`Wrote ${outfile}`);
