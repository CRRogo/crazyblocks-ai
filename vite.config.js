import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// For GitHub Pages: use your repository name as the base path
// Change 'crazyblocks' to your actual repository name if different
const REPO_NAME = 'crazyblocks'

export default defineConfig({
  plugins: [react()],
  base: process.env.NODE_ENV === 'production' ? `/${REPO_NAME}/` : '/',
  build: {
    outDir: 'dist',
  },
  publicDir: 'public',
})

