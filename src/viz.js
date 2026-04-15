// Three.js 3D visualisation for Plotpaint strokes. Replaces the
// matplotlib `Plotpaint.viz` method from the Python version.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const STROKE_COLORS = [
  0x1f77b4, 0xff7f0e, 0x2ca02c, 0xd62728, 0x9467bd, 0x8c564b,
  0xe377c2, 0x7f7f7f, 0xbcbd22, 0x17becf
];

export function viz(painting, lines, { page = false, container } = {}) {
  // normalise: if a single stroke was passed in, wrap it
  if (typeof lines[0][0] === 'number') lines = [lines];

  const host = container || document.getElementById('app');
  host.innerHTML = '';

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  const camera = new THREE.PerspectiveCamera(
    45,
    host.clientWidth / host.clientHeight,
    0.1,
    5000
  );
  camera.up.set(0, 0, 1);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(host.clientWidth, host.clientHeight);
  host.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);

  // page outline (y is flipped to match matplotlib orientation in Python viz)
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
    scene.add(new THREE.Line(geom, mat));
  }

  // stroke lines + start/end markers
  lines.forEach((stroke, i) => {
    // handle 2D input by padding z=0
    const points = stroke.map((p) =>
      new THREE.Vector3(p[0], p[1], p[2] ?? 0)
    );
    if (points.length < 2) return;
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({
      color: STROKE_COLORS[i % STROKE_COLORS.length]
    });
    scene.add(new THREE.Line(geom, mat));

    // dots at each point (small)
    const dotsMat = new THREE.PointsMaterial({
      color: STROKE_COLORS[i % STROKE_COLORS.length],
      size: 0.6
    });
    scene.add(new THREE.Points(geom, dotsMat));

    // start / end markers
    const startDot = new THREE.Mesh(
      new THREE.SphereGeometry(1.2, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0x00ff66 })
    );
    startDot.position.copy(points[0]);
    scene.add(startDot);

    const endDot = new THREE.Mesh(
      new THREE.SphereGeometry(1.2, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xff3355 })
    );
    endDot.position.copy(points[points.length - 1]);
    scene.add(endDot);
  });

  // axes + grid for orientation
  const axes = new THREE.AxesHelper(30);
  scene.add(axes);

  const grid = new THREE.GridHelper(
    Math.max(painting.pageWidth, painting.pageHeight),
    20,
    0x333333,
    0x222222
  );
  // the three.js grid is in XZ, rotate to XY
  grid.rotation.x = Math.PI / 2;
  grid.position.set(painting.pageWidth / 2, painting.pageHeight / 2, 0);
  scene.add(grid);

  // frame the camera on the page
  const cx = painting.pageWidth / 2;
  const cy = painting.pageHeight / 2;
  const radius = Math.max(painting.pageWidth, painting.pageHeight);
  camera.position.set(cx, cy - radius * 0.4, radius * 0.9);
  controls.target.set(cx, cy, 0);
  controls.update();

  const onResize = () => {
    renderer.setSize(host.clientWidth, host.clientHeight);
    camera.aspect = host.clientWidth / host.clientHeight;
    camera.updateProjectionMatrix();
  };
  window.addEventListener('resize', onResize);

  const animate = () => {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };
  animate();

  return { scene, camera, renderer, controls };
}

export default viz;
