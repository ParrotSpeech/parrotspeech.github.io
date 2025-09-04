import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  optimizeDeps: {
    include: ['onnxruntime-web'],
    exclude: []
  },
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'credentialless',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
    fs: {
      allow: ['..']
    }
  },
  build: {
    target: "esnext",
    rollupOptions: {
      external: [],
      output: {
        format: 'es'
      }
    }
  },
  worker: {
    format: 'es',
    plugins: () => [react(), tailwindcss()],
    rollupOptions: {
      external: [],
      output: {
        format: 'es'
      }
    }
  },
  // Ensure WASM files are treated as assets
  assetsInclude: ['**/*.wasm']
})
