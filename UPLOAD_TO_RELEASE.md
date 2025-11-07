# Files to Upload to GitHub Release v0.2.0

## Current Status

✅ **Release created**: https://github.com/prossm/chirpkit/releases/tag/v0.2.0
✅ **birdnet-models.zip uploaded** (22.4 MB)
⚠️ **chirpkit-ensemble.zip NEEDS UPDATE** (18 MB)

## Action Required

### Update chirpkit-ensemble.zip

The existing `chirpkit-ensemble.zip` on the release needs to be replaced with the updated version that includes:
- ✅ Label encoder with scikit-learn 1.7.2 (eliminates version warning)

**Steps:**
1. Go to https://github.com/prossm/chirpkit/releases/tag/v0.2.0
2. Click "Edit release"
3. Delete the old `chirpkit-ensemble.zip` (if present)
4. Upload the new `chirpkit-ensemble.zip` from project root (18 MB)
5. Click "Update release"

**File location:**
```
/Volumes/PortableSSD/dev/chirpkit/chirpkit-ensemble.zip
```

**Verify the file:**
```bash
ls -lh /Volumes/PortableSSD/dev/chirpkit/chirpkit-ensemble.zip
# Should show: 18M Nov  6 19:58 chirpkit-ensemble.zip

# Check contents:
unzip -l /Volumes/PortableSSD/dev/chirpkit/chirpkit-ensemble.zip
# Should include:
#   - chirpkit-ensemble/ensemble_info.json
#   - chirpkit-ensemble/ensemble_model_1.pth through ensemble_model_7.pth
#   - chirpkit-ensemble/label_encoder.joblib (UPDATED)
```

## What Changed

### Code Fixes (Already in Repository)
1. ✅ Fixed import paths in `classifier.py` and `simple_ui.py`
2. ✅ Fixed BirdNET path resolution in `birdnet_embeddings.py`
3. ✅ Fixed GPU/MPS detection in `dependencies.py`
4. ✅ Added `is_available()` method to `InsectClassifier`
5. ✅ Updated `requirements.txt` with secure package versions
6. ✅ Improved `model_downloader.py` with automatic fallbacks

### Model Files (Need Release Upload)
1. ✅ `birdnet-models.zip` - Already uploaded, no changes needed
2. ⚠️ `chirpkit-ensemble.zip` - **Needs update** with new label_encoder.joblib

## Testing After Upload

After uploading the updated `chirpkit-ensemble.zip`, test the download:

```bash
# Test download URL
curl -I https://github.com/prossm/chirpkit/releases/download/v0.2.0/chirpkit-ensemble.zip

# Should return: HTTP/2 200 or 302 (redirect)

# Test full download and installation
python3 -m venv test_env
source test_env/bin/activate
pip install git+https://github.com/prossm/chirpkit.git@main

# Test model download
python -c "
from chirpkit.model_downloader import ModelDownloader
ModelDownloader.download_all_models()
"

# Verify no scikit-learn warning
python -c "
from chirpkit import InsectClassifier
c = InsectClassifier()
c.load_model()
print('✅ All models loaded successfully')
"
```

## Expected Results

After updating the release, users should see:
- ✅ No import errors
- ✅ No scikit-learn version warnings
- ✅ No misleading GPU warnings on macOS with MPS
- ✅ Fast model downloads from GitHub Release
- ✅ Automatic Git LFS fallback if needed

## SoundCurious Integration

Once the updated `chirpkit-ensemble.zip` is uploaded, notify SoundCurious that they can:
1. Update their Docker build to latest ChirpKit
2. Models will auto-download with all fixes
3. No more corruption or version mismatch errors

See `SOUNDCURIOUS_FIX.md` for their integration instructions.
