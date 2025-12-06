# How to Disable the Default pages-build-deployment Workflow

The default `pages-build-deployment` workflow runs automatically when GitHub Pages is enabled. Here's how to disable it:

## Method 1: Repository Settings (Recommended)

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under **Source**, make sure it says **"GitHub Actions"** (NOT "Deploy from a branch")
4. If it's set to "Deploy from a branch", change it to "GitHub Actions"
5. The default `pages-build-deployment` workflow will stop running

## Method 2: Disable the Workflow File

1. Go to **Actions** tab
2. Click on **pages-build-deployment** in the left sidebar
3. Click the **...** (three dots) menu
4. Select **Disable workflow**

## Why Two Workflows Were Running

- **pages-build-deployment**: GitHub's default workflow (runs when Pages source is set to a branch)
- **Deploy to GitHub Pages**: Our custom workflow (runs when Pages source is set to GitHub Actions)

If both are running, it means Pages might be configured to use both methods, or the default one hasn't been disabled yet.

## After Fixing

Once you set Pages to use "GitHub Actions" only, you should see:
- ✅ Only "Deploy to GitHub Pages" workflow running
- ❌ "pages-build-deployment" will stop running

