# Troubleshooting: GitHub Pages Serving Source Files

## Problem
GitHub Pages is serving the source `index.html` (with `/src/main.jsx`) instead of the built `dist/index.html` (with `/crazyblocks-ai/assets/...`).

## Root Cause
GitHub Pages is configured to serve files from your repository instead of from the GitHub Actions deployment artifact.

## Solution Steps

### Step 1: Verify GitHub Pages Source
1. Go to: https://github.com/crrogo/crazyblocks-ai/settings/pages
2. Under **"Source"**, check what it says:
   - ❌ **"Deploy from a branch"** - This is WRONG and causes the issue
   - ✅ **"GitHub Actions"** - This is CORRECT

### Step 2: Change to GitHub Actions (if needed)
If it says "Deploy from a branch":
1. Click the dropdown
2. Select **"GitHub Actions"**
3. Click **Save**

### Step 3: Verify Deployment
1. Go to: https://github.com/crrogo/crazyblocks-ai/actions
2. Find the latest **"Deploy to GitHub Pages"** workflow run
3. Check if it has a ✅ green checkmark (success)
4. If it has ❌ red X, click on it to see the error

### Step 4: Wait for Deployment
After changing to "GitHub Actions":
- Wait 1-2 minutes for the deployment to complete
- The site should now serve the built files from `dist/`

### Step 5: Clear Browser Cache
- Hard refresh: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
- Or open in incognito/private mode

## How to Verify It's Working

After deployment, the HTML should contain:
```html
<script type="module" crossorigin src="/crazyblocks-ai/assets/index-XXXXX.js"></script>
```

NOT:
```html
<script type="module" src="/src/main.jsx"></script>
```

## If Still Not Working

1. **Check the workflow logs:**
   - Go to Actions → Latest "Deploy to GitHub Pages" run
   - Check the "Verify index.html content" step
   - It should show the correct paths with `/crazyblocks-ai/assets/`

2. **Manually trigger deployment:**
   - Go to Actions → "Deploy to GitHub Pages"
   - Click "Run workflow" → "Run workflow"

3. **Check deployment status:**
   - Go to Settings → Pages
   - Look for "Your site is live at..." message
   - Check the deployment history

