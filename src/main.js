// Browser entry — mirrors example.py.

import { Plotpaint, npline, poly } from './plotpaint/index.js';
import { viz } from './viz.js';

// tee console.log to the in-page log pane
const logEl = document.getElementById('log');
const origLog = console.log.bind(console);
const formatLogArg = (a) => {
  if (a !== null && typeof a === 'object') {
    try {
      return JSON.stringify(a);
    } catch {
      return String(a);
    }
  }
  return String(a);
};
console.log = (...args) => {
  origLog(...args);
  if (logEl) {
    logEl.textContent += args.map(formatLogArg).join(' ') + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  }
};

const painting = new Plotpaint();

// ----- example (ported from example.py) ---------------------------------
painting.runIn = 0;
painting.runOut = 0;

const circle = poly([200, 200], 100, 0, 8);
const paths = npline.split_path(circle, 500);
for (const path of paths) painting.addStroke(path);

painting.runIn = 0;
painting.runOut = 0;

for (const path of paths) {
  painting.addStroke(poly(path[0], 10, 0, 4));
}

console.log(`${paths.length} paths`);
// ------------------------------------------------------------------------

viz(painting, painting.strokes, { page: true });

// wire up the download button
const btn = document.getElementById('download');
if (btn) {
  btn.addEventListener('click', () => {
    const { filename, text } = painting.output('painting.gcode');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  });
}
