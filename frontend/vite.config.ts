import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: './',
  build: {
    outDir: 'dist',
    // Keep previous content-hashed bundles during in-place updates. A Home
    // Assistant iframe can still be running HTML from the previous build while
    // the new one is published, and deleting its bundle leaves a blank panel.
    emptyOutDir: false,
    sourcemap: false,
    target: 'es2022',
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8099',
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    clearMocks: true,
  },
});
