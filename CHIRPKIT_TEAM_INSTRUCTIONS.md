# Critical Instructions for ChirpKit Team

## 🚨 Current Issues Blocking Adoption

### 1. Repository Size Problem
- **Current**: 700GB+ training data mixed with 44MB runtime code
- **Impact**: 40+ second clones, 650MB+ Docker images
- **Affects**: All downstream projects (SoundCurious, containers, CI/CD)

### 2. Dependency Version Conflicts
- **ChirpKit requirements**: `numpy>=1.21.0,<2.0.0`, `torch>=1.9.0`
- **Modern projects need**: `numpy>=1.26.0`, `torch>=2.0.0`
- **Result**: Forces downgrades, duplicate installations, dependency hell

## 📋 Action Plan for ChirpKit Team

### 🏃 Quick Wins (1-2 weeks)

#### 1. Create Inference-Only Branch
```bash
# Create lean branch for production use
git checkout -b inference-only

# Remove heavy research files
rm -rf datasets/
rm -rf data/raw/
rm -rf notebooks/
rm -rf training_runs/
rm -rf experiments/

# Keep only essential runtime files:
# ✅ src/chirpkit/
# ✅ models/ (production models only)
# ✅ setup.py
# ✅ README.md
# ✅ requirements.txt (updated)

git add -A
git commit -m "Create inference-only branch for production deployments"
git push origin inference-only
```

#### 2. Fix Dependency Versions
```python
# setup.py - update to modern compatible versions
install_requires=[
    "numpy>=1.21.0",        # Remove upper bound for compatibility
    "torch>=2.0.0",         # Match modern PyTorch ecosystem
    "librosa>=0.9.0",       
    "scikit-learn>=1.0.0",
    "pandas>=1.3.0",
    "soundfile>=0.10.0",
    "joblib>=1.0.0",
    "requests>=2.32.4",
    "PyYAML>=5.4.0",
],
```

#### 3. Upload Models to GitHub Releases
```bash
# Package production models separately
cd models/
zip -r ../chirpkit-models-v0.2.0.zip trained/ birdnet/

# Upload to GitHub Releases at:
# https://github.com/prossm/chirpkit/releases/new
# Title: ChirpKit v0.2.0 - Production Models
# File: chirpkit-models-v0.2.0.zip (44MB)
```

### 🎯 Medium-term Improvements (1 month)

#### 4. Create PyPI Package Variants
```bash
# Option A: Separate packages
pip install chirpkit-inference  # Runtime only (5MB install)
pip install chirpkit-research   # Full research (700GB)

# Option B: Optional dependencies
pip install chirpkit             # Core package
pip install chirpkit[research]   # + training data
pip install chirpkit[full]       # + everything
```

#### 5. Implement Smart Model Downloads
```python
# Enhanced model downloader
from chirpkit.downloader import download_models

# Minimal deployment (inference only)
download_models(
    variant="inference",     # Just ensemble + BirdNET
    cache_dir="/models/",
    size="44MB"
)

# Full research setup
download_models(
    variant="research",      # All models + training data
    cache_dir="/data/", 
    size="700GB"
)
```

#### 6. Container-Optimized Images
```dockerfile
# Multi-stage Docker build
FROM python:3.11-slim as base
RUN pip install chirpkit-inference

FROM base as research
RUN pip install chirpkit[research]

# Result:
# chirpkit:inference -> 200MB (vs 850MB currently)
# chirpkit:research  -> 850MB (same functionality)
```

## 📊 Expected Impact

### Before (Current State)
```
Git clone: 40+ seconds
Docker image: 650MB+
pip install: Downloads 700GB+ data
Dependency conflicts: NumPy/PyTorch version mismatches
```

### After (With Fixes)
```
Git clone (inference): 5 seconds
Docker image: 200MB
pip install: Downloads 44MB models on-demand
Dependencies: Compatible with modern ecosystem
```

**Time Savings**: 60-90 seconds per build
**Storage Savings**: 600MB+ per deployment
**Compatibility**: ✅ Works with modern NumPy/PyTorch

## 🚀 Implementation Priority

### Phase 1: Immediate (This Week)
1. ✅ **Model configuration system** (Already implemented!)
2. 🔲 **Fix dependency versions** in setup.py
3. 🔲 **Create inference-only branch**

### Phase 2: Short-term (2 weeks)
4. 🔲 **Upload models to GitHub Releases**
5. 🔲 **Update documentation** for new installation methods
6. 🔲 **Test with SoundCurious integration**

### Phase 3: Medium-term (1 month)  
7. 🔲 **Create PyPI package variants**
8. 🔲 **Implement smart model downloads**
9. 🔲 **Optimize container images**

## 🧪 Testing Checklist

### Compatibility Testing
- [ ] SoundCurious integration (NumPy 1.26+, PyTorch 2.0+)
- [ ] Container builds (<300MB target)
- [ ] Fresh installs (models download correctly)
- [ ] Existing code (backwards compatibility)

### Performance Testing  
- [ ] Clone time (<10 seconds for inference branch)
- [ ] Install time (<30 seconds with model download)
- [ ] Runtime performance (no regression)
- [ ] Memory usage (inference models only)

## 💡 Additional Recommendations

### Repository Structure
```
chirpkit/
├── main branch (research - current)
├── inference-only branch (production)
└── releases/ (pre-built model packages)
```

### Documentation Updates
- Add "Quick Install" section for inference-only
- Container deployment guide with size comparisons  
- Migration guide for existing projects
- Performance benchmarks before/after

### Community Impact
- Easier adoption by production projects
- Faster CI/CD pipelines
- Reduced cloud storage costs
- Better compatibility with modern ML stack

---

**Bottom Line**: These changes will make ChirpKit much more adoptable for production use while maintaining full research capabilities. SoundCurious build time drops by 60-90 seconds, Docker images shrink by 600MB+, and dependency conflicts disappear! 🎯