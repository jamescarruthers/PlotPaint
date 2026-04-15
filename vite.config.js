import { defineConfig } from 'vite';

// For GitHub Pages under https://<user>.github.io/PlotPaint/ we need the
// built assets to resolve under /PlotPaint/. The dev server continues to
// serve from root. Override with the VITE_BASE env var if deploying
// somewhere else.
export default defineConfig(({ command }) => ({
  root: '.',
  base:
    process.env.VITE_BASE ?? (command === 'build' ? '/PlotPaint/' : '/'),
  server: {
    open: true,
    port: 5173
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
}));
