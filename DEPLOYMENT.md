# Deployment Guide

## GitHub Pages (Recommended - Free & Easy)

### Setup Steps:

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for GitHub Pages deployment"
   git push origin main
   ```

2. **Enable GitHub Pages**:
   - Go to your repository on GitHub
   - Click **Settings** → **Pages**
   - Under **Source**, select **GitHub Actions**
   - Save

3. **Deploy**:
   - The GitHub Actions workflow will automatically deploy when you push to `main`
   - Or manually trigger it: **Actions** tab → **Deploy to GitHub Pages** → **Run workflow**

4. **Your site will be available at**:
   ```
   https://YOUR_USERNAME.github.io/crazyblocks/
   ```

### Manual Deployment (Alternative):

If you prefer to deploy manually:

```bash
# Build the app
npm run build

# The dist folder contains your built app
# You can deploy this folder to any static hosting service
```

## Other Free Hosting Options:

### 1. **Vercel** (Easiest - Zero Config)
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Deploys automatically on every push
   - **URL**: `https://your-app.vercel.app`

### 2. **Netlify** (Also Very Easy)
   - Go to [netlify.com](https://netlify.com)
   - Drag and drop your `dist` folder after building
   - Or connect your GitHub repo for auto-deploy
   - **URL**: `https://your-app.netlify.app`

### 3. **Cloudflare Pages** (Fast & Free)
   - Go to [pages.cloudflare.com](https://pages.cloudflare.com)
   - Connect your GitHub repository
   - Auto-deploys on push
   - **URL**: `https://your-app.pages.dev`

## Notes:

- The strategy file (`crazyblocks-strategies.json`) needs to be in the `public` folder
- The GitHub Actions workflow will automatically copy it during build
- If you update the strategy file manually, push it to the `public` folder
- The base path is set to `/crazyblocks/` - change this in `vite.config.js` if your repo name is different

