# GitHub Pages Setup Notes

## Two Workflows Running?

You might see two GitHub Actions workflows:
1. **Deploy to GitHub Pages** (custom) - This is the one we want! It properly builds your Vite app.
2. **pages-build-deployment** (default) - This is GitHub's default workflow that tries to build Jekyll sites.

### To Use Only the Custom Workflow:

1. Go to **Settings** → **Pages**
2. Under **Source**, make sure it says **"GitHub Actions"** (not "Deploy from a branch")
3. The default `pages-build-deployment` will stop running automatically

### If You Want to Disable the Default Workflow:

You can disable it by going to **Actions** → **pages-build-deployment** → **...** → **Disable workflow**

## Required Files:

- ✅ `package-lock.json` - Must be committed (removed from .gitignore)
- ✅ `vite.config.js` - Configured with correct base path
- ✅ `.github/workflows/deploy.yml` - Custom deployment workflow

## After Committing package-lock.json:

1. Push the changes:
   ```bash
   git add package-lock.json .gitignore
   git commit -m "Add package-lock.json for CI/CD"
   git push origin main
   ```

2. The custom workflow should now succeed!

