// Three.js 3D visualisation for Plotpaint strokes. Replaces the
// matplotlib `Plotpaint.viz` method from the Python version.
//
// Exposes `createViz(container)` which sets up a persistent scene and
// returns an object with `update(painting, lines, opts)` — this lets the
// interactive editor re-render on every code change without tearing
// down the WebGL context or losing the user's camera position.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const STROKE_COLORS = [
  0x1f77b4, 0xff7f0e, 0x2ca02c, 0xd62728, 0x9467bd, 0x8c564b,
  0xe377c2, 0x7f7f7f, 0xbcbd22, 0x17becf
];

function disposeGroup(group) {
  group.traverse((obj) => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const m of mats) m.dispose?.();
    }
  });
  group.clear();
}

export function createViz(container) {
  const host = container || document.getElementById('viz');

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  const camera = new THREE.PerspectiveCamera(
    45,
    Math.max(host.clientWidth, 1) / Math.max(host.clientHeight, 1),
    0.1,
    5000
  );
  camera.up.set(0, 0, 1);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(host.clientWidth, host.clientHeight);
  host.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);

  const axes = new THREE.AxesHelper(30);
  scene.add(axes);

  // Persistent group that user-content (strokes, page outline, grid) is
  // added to. Cleared and repopulated on each `update()`.
  const content = new THREE.Group();
  scene.add(content);

  let cameraInitialised = false;

  function frameCamera(painting) {
    const cx = painting.pageWidth / 2;
    const cy = painting.pageHeight / 2;
    const radius = Math.max(painting.pageWidth, painting.pageHeight);
    camera.position.set(cx, cy - radius * 0.4, radius * 0.9);
    controls.target.set(cx, cy, 0);
    controls.update();
  }

  function update(painting, lines, { page = true, resetCamera = false } = {}) {
    // normalise: if a single stroke was passed in, wrap it
    if (lines.length && typeof lines[0][0] === 'number') lines = [lines];

    disposeGroup(content);

    // page outline in native paint-space coordinates (origin at bottom-left,
    // Y increases upward). OrbitControls handles the camera orientation.
    if (page) {
      const pageLines = [
        [0, 0, 0],
        [0, painting.pageHeight, 0],
        [painting.pageWidth, painting.pageHeight, 0],
        [painting.pageWidth, 0, 0],
        [0, 0, 0]
      ];
      const geom = new THREE.BufferGeometry().setFromPoints(
        pageLines.map((p) => new THREE.Vector3(p[0], p[1], p[2]))
      );
      const mat = new THREE.LineBasicMaterial({ color: 0xffffff });
      content.add(new THREE.Line(geom, mat));
    }

    // grid on the XY plane
    const grid = new THREE.GridHelper(
      Math.max(painting.pageWidth, painting.pageHeight),
      20,
      0x333333,
      0x222222
    );
    grid.rotation.x = Math.PI / 2;
    grid.position.set(painting.pageWidth / 2, painting.pageHeight / 2, 0);
    content.add(grid);

    lines.forEach((stroke, i) => {
      const points = stroke.map((p) =>
        new THREE.Vector3(p[0], p[1], p[2] ?? 0)
      );
      if (points.length < 2) return;
      const color = STROKE_COLORS[i % STROKE_COLORS.length];

      const geom = new THREE.BufferGeometry().setFromPoints(points);
      content.add(
        new THREE.Line(geom, new THREE.LineBasicMaterial({ color }))
      );
      content.add(
        new THREE.Points(
          geom,
          new THREE.PointsMaterial({ color, size: 0.6 })
        )
      );

      const startDot = new THREE.Mesh(
        new THREE.SphereGeometry(1.2, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0x00ff66 })
      );
      startDot.position.copy(points[0]);
      content.add(startDot);

      const endDot = new THREE.Mesh(
        new THREE.SphereGeometry(1.2, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0xff3355 })
      );
      endDot.position.copy(points[points.length - 1]);
      content.add(endDot);
    });

    if (!cameraInitialised || resetCamera) {
      frameCamera(painting);
      cameraInitialised = true;
    }
  }

  function resize() {
    const w = Math.max(host.clientWidth, 1);
    const h = Math.max(host.clientHeight, 1);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  // Observe the container so splitter drags / window resizes both work.
  const ro = new ResizeObserver(resize);
  ro.observe(host);

  let rafId = 0;
  let stopped = false;
  const tick = () => {
    if (stopped) return;
    rafId = requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  };
  tick();

  function dispose() {
    stopped = true;
    cancelAnimationFrame(rafId);
    ro.disconnect();
    controls.dispose();
    disposeGroup(content);
    renderer.dispose();
    if (renderer.domElement.parentNode === host) {
      host.removeChild(renderer.domElement);
    }
  }

  return {
    update,
    resetCamera: () => frameCamera,
    dispose,
    scene,
    camera,
    renderer,
    controls
  };
}

// Back-compat single-shot API — builds a fresh viz and renders once.
export function viz(painting, lines, { page = false, container } = {}) {
  const v = createViz(container);
  v.update(painting, lines, { page, resetCamera: true });
  return v;
}

export default createViz;
