import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// For GitHub Pages: use your repository name as the base path
// Change 'crazyblocks-ai' to your actual repository name if different
const REPO_NAME = 'crazyblocks-ai'

export default defineConfig({
  plugins: [react()],
  // Use environment variable if set, otherwise use REPO_NAME
  base: process.env.VITE_BASE || (process.env.NODE_ENV === 'production' ? `/${REPO_NAME}/` : '/'),
  build: {
    outDir: 'dist',
  },
  publicDir: 'public',
})

