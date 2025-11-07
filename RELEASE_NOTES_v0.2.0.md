# ChirpKit v0.2.0 Release Notes

## What's New

### Fixed Issues
✅ **Import Path Errors** - Fixed incorrect import paths (`models.chirpkit_ensemble` → `.models.chirpkit_ensemble`)
✅ **BirdNET Model Path Resolution** - Corrected path detection for both development and production environments
✅ **Label Encoder Version** - Updated to scikit-learn 1.7.2 (eliminates version mismatch warnings)
✅ **GPU Detection** - Fixed misleading MPS/Metal GPU warning on macOS
✅ **Added `is_available()` Method** - Consistent API for checking classifier status

### Security Updates
- Updated PyTorch to 2.6.0+ (fixes 7 security vulnerabilities)
- Updated requests to 2.32.4+ (fixes 2 vulnerabilities)
- Updated cryptography, pillow, jinja2, and 10+ other packages

### Model Distribution
- **Auto-download support** via GitHub Releases
- **Automatic fallback** to Git LFS if release download fails
- Models now available at: https://github.com/prossm/chirpkit/releases/tag/v0.2.0

## Installation

### Quick Install
```bash
pip install git+https://github.com/prossm/chirpkit.git@main
```

Models will auto-download on first use!

### Docker
```dockerfile
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Install ChirpKit
RUN pip install git+https://github.com/prossm/chirpkit.git@main

# Pre-download models
RUN python -c "from chirpkit.model_downloader import ModelDownloader; \
    ModelDownloader.download_all_models()"
```

## Release Assets

This release includes two model packages:

### 1. birdnet-models.zip (22.4 MB)
- BirdNET v2.4 embedding extractor
- Required for feature extraction from audio

### 2. chirpkit-ensemble.zip (18 MB) ⚠️ **UPDATED**
- 7-model ensemble (79.7% accuracy)
- 231 insect species
- Label encoder updated to scikit-learn 1.7.2

## For Existing Users

If you're updating from an earlier version:

```bash
# Update ChirpKit
pip install --upgrade --force-reinstall git+https://github.com/prossm/chirpkit.git@main

# Force re-download models (gets updated label encoder)
python -c "from chirpkit.model_downloader import ModelDownloader; \
    ModelDownloader.download_model('chirpkit-ensemble', force=True)"
```

## Breaking Changes

None - this release is backward compatible.

## Known Issues

None at this time.

## Model Performance

- **Ensemble Mode**: 79.6% accuracy
- **Ensemble + TTA Mode**: 79.7% accuracy
- **Single Model Mode**: 77% accuracy
- **Species Supported**: 231 insects

## Special Thanks

- SoundCurious team for reporting integration issues
- All contributors and testers

---

**Full Changelog**: https://github.com/prossm/chirpkit/commits/v0.2.0
