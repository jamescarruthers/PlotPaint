// Easing curves. Each function accepts an array of t values (typically in
// [0,1], but not clamped — the feed ease in Plotpaint is called with
// values spanning [0,2] to produce a full wave) and returns an array of
// eased values. Mirrors the easywaves.npCurves interface used by the
// Python version.

const vec = (fn) => (t) => Array.isArray(t) ? t.map(fn) : fn(t);

const PI = Math.PI;

export const npCurves = {
  linear: vec((x) => x),

  quadIn: vec((x) => x * x),
  quadOut: vec((x) => 1 - (1 - x) * (1 - x)),
  quadInOut: vec((x) =>
    x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2
  ),

  cubicIn: vec((x) => x * x * x),
  cubicOut: vec((x) => 1 - Math.pow(1 - x, 3)),
  cubicInOut: vec((x) =>
    x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2
  ),

  quartIn: vec((x) => x * x * x * x),
  quartOut: vec((x) => 1 - Math.pow(1 - x, 4)),
  quartInOut: vec((x) =>
    x < 0.5 ? 8 * x * x * x * x : 1 - Math.pow(-2 * x + 2, 4) / 2
  ),

  quintIn: vec((x) => x ** 5),
  quintOut: vec((x) => 1 - Math.pow(1 - x, 5)),
  quintInOut: vec((x) =>
    x < 0.5 ? 16 * x ** 5 : 1 - Math.pow(-2 * x + 2, 5) / 2
  ),

  sineIn: vec((x) => 1 - Math.cos((x * PI) / 2)),
  sineOut: vec((x) => Math.sin((x * PI) / 2)),
  sineInOut: vec((x) => -(Math.cos(PI * x) - 1) / 2),

  expoIn: vec((x) => (x === 0 ? 0 : Math.pow(2, 10 * x - 10))),
  expoOut: vec((x) => (x === 1 ? 1 : 1 - Math.pow(2, -10 * x))),
  expoInOut: vec((x) => {
    if (x === 0) return 0;
    if (x === 1) return 1;
    return x < 0.5
      ? Math.pow(2, 20 * x - 10) / 2
      : (2 - Math.pow(2, -20 * x + 10)) / 2;
  }),

  circIn: vec((x) => 1 - Math.sqrt(Math.max(0, 1 - x * x))),
  circOut: vec((x) => Math.sqrt(Math.max(0, 1 - Math.pow(x - 1, 2)))),
  circInOut: vec((x) =>
    x < 0.5
      ? (1 - Math.sqrt(Math.max(0, 1 - Math.pow(2 * x, 2)))) / 2
      : (Math.sqrt(Math.max(0, 1 - Math.pow(-2 * x + 2, 2))) + 1) / 2
  ),

  backIn: vec((x) => {
    const s = 1.70158;
    return (s + 1) * x * x * x - s * x * x;
  }),
  backOut: vec((x) => {
    const s = 1.70158;
    const u = x - 1;
    return 1 + (s + 1) * u * u * u + s * u * u;
  }),
  backInOut: vec((x) => {
    const s = 1.70158 * 1.525;
    return x < 0.5
      ? (Math.pow(2 * x, 2) * ((s + 1) * 2 * x - s)) / 2
      : (Math.pow(2 * x - 2, 2) * ((s + 1) * (x * 2 - 2) + s) + 2) / 2;
  }),

  elasticIn: vec((x) => {
    if (x === 0) return 0;
    if (x === 1) return 1;
    const c4 = (2 * PI) / 3;
    return -Math.pow(2, 10 * x - 10) * Math.sin((x * 10 - 10.75) * c4);
  }),
  elasticOut: vec((x) => {
    if (x === 0) return 0;
    if (x === 1) return 1;
    const c4 = (2 * PI) / 3;
    return Math.pow(2, -10 * x) * Math.sin((x * 10 - 0.75) * c4) + 1;
  }),
  elasticInOut: vec((x) => {
    if (x === 0) return 0;
    if (x === 1) return 1;
    const c5 = (2 * PI) / 4.5;
    return x < 0.5
      ? -(Math.pow(2, 20 * x - 10) * Math.sin((20 * x - 11.125) * c5)) / 2
      : (Math.pow(2, -20 * x + 10) * Math.sin((20 * x - 11.125) * c5)) / 2 + 1;
  }),

  bounceOut: vec((x) => {
    const n1 = 7.5625;
    const d1 = 2.75;
    if (x < 1 / d1) return n1 * x * x;
    if (x < 2 / d1) { x -= 1.5 / d1; return n1 * x * x + 0.75; }
    if (x < 2.5 / d1) { x -= 2.25 / d1; return n1 * x * x + 0.9375; }
    x -= 2.625 / d1; return n1 * x * x + 0.984375;
  }),
  bounceIn: vec((x) => 1 - npCurves.bounceOut(1 - x)),
  bounceInOut: vec((x) =>
    x < 0.5
      ? (1 - npCurves.bounceOut(1 - 2 * x)) / 2
      : (1 + npCurves.bounceOut(2 * x - 1)) / 2
  )
};

export default npCurves;
