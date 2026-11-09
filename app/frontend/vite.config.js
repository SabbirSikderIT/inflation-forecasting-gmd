import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The browser calls /api/... on the Vite origin; Vite forwards it to FastAPI.
const api = { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } }

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: api },
  preview: { port: 4173, proxy: api },
})
