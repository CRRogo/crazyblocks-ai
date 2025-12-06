# Fix: Two Workflows Running

You're seeing both workflows because GitHub's default `pages-build-deployment` is still enabled. Here's how to fix it:

## Step-by-Step Fix:

### 1. Disable the Default Workflow

1. Go to your repository: `https://github.com/crrogo/crazyblocks-ai`
2. Click the **Actions** tab
3. In the left sidebar, find **"pages-build-deployment"**
4. Click on it
5. Click the **...** (three dots) menu in the top right
6. Select **"Disable workflow"**
7. Confirm by clicking **"Disable workflow"** in the popup

### 2. Verify Pages Settings

1. Go to **Settings** → **Pages**
2. Under **Source**, make sure it says:
   - ✅ **"GitHub Actions"** (this is what you want)
   - ❌ NOT "Deploy from a branch"

### 3. After Disabling

- The `pages-build-deployment` workflow will stop running
- Only your custom **"Deploy to GitHub Pages"** workflow will run
- Future pushes will only trigger one workflow

## Why This Happens

GitHub automatically enables the default `pages-build-deployment` workflow when Pages is first enabled. Even if you switch to "GitHub Actions" as the source, the default workflow file remains active until you explicitly disable it.

## Alternative: If You Can't Find the Disable Option

If the disable option isn't visible, you can also:

1. Go to **Settings** → **Actions** → **General**
2. Scroll down to **"Workflow permissions"**
3. Make sure it's set to allow workflows to run
4. Then go back to **Actions** tab and disable the workflow as described above

