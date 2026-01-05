# qsense-continuos Repository Enhancement Guide

This document provides instructions and files to enhance the qsense-continuos repository front-end.

## 🎯 Goals

1. Add repository "About" description
2. Add relevant topics/tags
3. Fix language distribution (reduce Jupyter Notebook dominance from 99.7%)

## 📝 Step 1: Add Repository About & Topics

### Via GitHub Web Interface:

1. Go to https://github.com/matibilkis/qsense-continuos
2. Click the ⚙️ (gear icon) next to "About" section
3. Add the following:

**Description:**
```
Advanced numerical framework for real-time parameter estimation in continuously-monitored quantum systems using Fisher information tracking, custom TensorFlow RNNs, and high-performance SDE solvers.
```

**Topics (add these):**
- quantum-sensing
- fisher-information
- parameter-estimation
- quantum-metrology
- tensorflow
- stochastic-differential-equations
- quantum-filtering
- continuous-measurement
- numba
- htcondor
- quantum-optics
- cramer-rao-bound

**Website (optional):**
Leave blank or add paper URL if available

## 📁 Step 2: Add .gitattributes File

Create `.gitattributes` in the repository root with the following content:

```gitattributes
# Language detection for GitHub
*.py linguist-detectable=true

# Mark analysis notebooks as documentation (not code)
comparing_results/**/*.ipynb linguist-documentation=true
fisher_lorentzian/**/*.ipynb linguist-documentation=true
*.ipynb linguist-documentation=true

# Exclude checkpoint files and cache
**/.ipynb_checkpoints/** linguist-vendored=true
**/__pycache__/** linguist-vendored=true
**/past/** linguist-vendored=true
**/trash/** linguist-vendored=true

# Mark test/example notebooks as documentation
**/*test*.ipynb linguist-documentation=true
**/*example*.ipynb linguist-documentation=true
**/*tutorial*.ipynb linguist-documentation=true

# Keep core Python code in language stats
numerics/**/*.py linguist-detectable=true
fisher_tf/**/*.py linguist-detectable=true
```

## 🧹 Step 3: Clean Up Checkpoint Files

Remove `.ipynb_checkpoints` directories from git:

```bash
cd qsense-continuos
git rm -r --cached **/.ipynb_checkpoints
git rm -r --cached .ipynb_checkpoints
```

Update `.gitignore` to include:
```
.ipynb_checkpoints/
**/.ipynb_checkpoints/
*.ipynb_checkpoints
```

## 📊 Expected Results

After these changes:
- **Language distribution**: Should show Python as primary language (43 Python files vs 107 notebooks marked as documentation)
- **About section**: Professional description visible
- **Topics**: Repository discoverable via relevant tags
- **Cleaner structure**: No checkpoint files in repository

## 🚀 Quick Implementation

1. Clone the repository (if not already):
   ```bash
   git clone https://github.com/matibilkis/qsense-continuos.git
   cd qsense-continuos
   ```

2. Add the `.gitattributes` file (see content above)

3. Update `.gitignore` to exclude checkpoints

4. Remove checkpoint files:
   ```bash
   git rm -r --cached **/.ipynb_checkpoints 2>/dev/null
   git rm -r --cached .ipynb_checkpoints 2>/dev/null
   ```

5. Commit and push:
   ```bash
   git add .gitattributes .gitignore
   git commit -m "chore: Add .gitattributes to fix language detection and clean up repository"
   git push origin master
   ```

6. Add About and Topics via GitHub web interface (see Step 1)

## 📈 Current vs Expected Language Distribution

**Current:**
- Jupyter Notebook: 99.7%
- Other: 0.3%

**Expected (after changes):**
- Python: ~70-80% (43 Python files)
- Jupyter Notebook: ~20-30% (notebooks marked as documentation)
- Shell: ~5% (if .sh files exist)

---

*Note: GitHub language detection updates may take a few minutes after pushing changes.*

