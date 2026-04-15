// Interactive PlotPaint demo — code editor on the left, 3D preview on
// the right. Editing the code and pressing Run (or Ctrl/⌘+Enter)
// re-evaluates the script against a fresh Plotpaint instance.

import {
  Plotpaint,
  npline,
  poly,
  line,
  concfill,
  contractexpand,
  npCurves,
  draw
} from './plotpaint/index.js';
import { createViz } from './viz.js';
import { createEditor } from './editor.js';

const STORAGE_KEY = 'plotpaint.code.v1';

const INITIAL_CODE = `// PlotPaint demo — edit and hit Ctrl/⌘+Enter to run.
//
// In scope:
//   painting   — a fresh Plotpaint instance, pre-created for you
//   Plotpaint  — the class
//   npline     — line / polygon utilities
//   poly(center, radius, startAngle, numPoints)
//   line(center, length, angleDeg)
//   concfill, contractexpand
//   npCurves   — easing curves for feeds / run-in / run-out

painting.runIn = 0;
painting.runOut = 0;

const circle = poly([200, 200], 100, 0, 8);
const paths = npline.split_path(circle, 500);
for (const path of paths) painting.addStroke(path);

for (const path of paths) {
  painting.addStroke(poly(path[0], 10, 0, 4));
}

console.log(\`\${paths.length} paths\`);
`;

// ----- log panel ----------------------------------------------------------
const logEl = document.getElementById('log');
const origLog = console.log.bind(console);
const origErr = console.error.bind(console);

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

function appendLog(args, cls) {
  if (!logEl) return;
  const text = args.map(formatLogArg).join(' ') + '\n';
  if (cls) {
    const span = document.createElement('span');
    span.className = cls;
    span.textContent = text;
    logEl.appendChild(span);
  } else {
    logEl.appendChild(document.createTextNode(text));
  }
  logEl.scrollTop = logEl.scrollHeight;
}

function clearLog() {
  if (logEl) logEl.textContent = '';
}

console.log = (...args) => { origLog(...args); appendLog(args); };
console.error = (...args) => { origErr(...args); appendLog(args, 'err'); };

window.addEventListener('error', (e) => {
  appendLog([e.message || String(e.error || e)], 'err');
});
window.addEventListener('unhandledrejection', (e) => {
  appendLog([`Unhandled: ${e.reason?.message ?? e.reason}`], 'err');
});

// ----- viz + painting state ----------------------------------------------
const viz = createViz(document.getElementById('viz'));

let currentPainting = null;

async function runCode(code) {
  clearLog();
  const painting = new Plotpaint();
  currentPainting = painting;

  // Evaluate user code in an AsyncFunction so they can use `await` and
  // library helpers are in scope as identifiers (no imports needed).
  const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
  const fn = new AsyncFunction(
    'painting',
    'Plotpaint',
    'npline',
    'poly',
    'line',
    'concfill',
    'contractexpand',
    'npCurves',
    'draw',
    code
  );

  try {
    await fn(
      painting,
      Plotpaint,
      npline,
      poly,
      line,
      concfill,
      contractexpand,
      npCurves,
      draw
    );
    viz.update(painting, painting.strokes, { page: true });
  } catch (err) {
    console.error(err.stack || err.message || String(err));
  }
}

// ----- editor -------------------------------------------------------------
const saved = localStorage.getItem(STORAGE_KEY);
const editor = createEditor({
  parent: document.getElementById('editor'),
  doc: saved ?? INITIAL_CODE,
  onChange: (code) => {
    try { localStorage.setItem(STORAGE_KEY, code); } catch {}
  },
  onRun: () => runCode(editor.getCode())
});

document.getElementById('run').addEventListener('click', () => {
  runCode(editor.getCode());
});

document.getElementById('download').addEventListener('click', () => {
  if (!currentPainting) {
    console.error('Run the code first — nothing to export yet.');
    return;
  }
  const { filename, text } = currentPainting.output('painting.gcode');
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
});

// ----- splitter -----------------------------------------------------------
const app = document.getElementById('app');
const splitter = document.getElementById('splitter');
let dragging = false;

splitter.addEventListener('mousedown', (e) => {
  dragging = true;
  document.body.style.userSelect = 'none';
  document.body.style.cursor =
    window.innerWidth <= 720 ? 'row-resize' : 'col-resize';
  e.preventDefault();
});

window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  if (window.innerWidth <= 720) {
    // stacked layout — adjust top row height
    const h = Math.max(120, Math.min(window.innerHeight - 120, e.clientY));
    app.style.gridTemplateRows = `${h}px 6px 1fr`;
  } else {
    const w = Math.max(260, Math.min(window.innerWidth - 260, e.clientX));
    app.style.setProperty('--split', `${w}px`);
  }
});

window.addEventListener('mouseup', () => {
  if (!dragging) return;
  dragging = false;
  document.body.style.userSelect = '';
  document.body.style.cursor = '';
});

// ----- initial run --------------------------------------------------------
runCode(editor.getCode());
