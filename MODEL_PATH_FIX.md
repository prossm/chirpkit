# BirdNET Model Path Configuration Fix

## Problem

ChirpKit successfully downloads BirdNET models to `~/.chirpkit/models/birdnet/`, but BirdNET-Analyzer was not configured to use these downloaded models. Instead, it tried to use models from the BirdNET-Analyzer package directory, which may not exist.

## The Fix

Updated `src/chirpkit/transfer_learning/birdnet_embeddings.py` lines 84-115 to configure BirdNET-Analyzer to use ChirpKit's downloaded models.

### Before (BROKEN)
```python
def _init_with_analyzer(self):
    """Initialize using full BirdNET-Analyzer"""
    # Only checks if BIRDNET_MODEL_PATH exists
    if hasattr(birdnet_cfg, 'BIRDNET_MODEL_PATH'):
        birdnet_cfg.MODEL_PATH = birdnet_cfg.BIRDNET_MODEL_PATH
    if hasattr(birdnet_cfg, 'BIRDNET_LABELS_FILE'):
        birdnet_cfg.LABELS_FILE = birdnet_cfg.BIRDNET_LABELS_FILE

    # ... rest of config ...
    birdnet_model.load_model(class_output=False)
```

**Problem:** Never checks ChirpKit's `~/.chirpkit/models/birdnet/` directory!

### After (FIXED)
```python
def _init_with_analyzer(self):
    """Initialize using full BirdNET-Analyzer"""
    # Point to ChirpKit's downloaded models in ~/.chirpkit/models/birdnet/
    chirpkit_model_dir = Path.home() / '.chirpkit' / 'models' / 'birdnet'

    if chirpkit_model_dir.exists():
        # Use ChirpKit's downloaded models
        birdnet_cfg.MODEL_PATH = str(chirpkit_model_dir / 'BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite')
        birdnet_cfg.LABELS_FILE = str(chirpkit_model_dir / 'BirdNET_GLOBAL_6K_V2.4_Labels.txt')
        print(f"✅ Using ChirpKit downloaded models from: {chirpkit_model_dir}")
    elif hasattr(birdnet_cfg, 'BIRDNET_MODEL_PATH'):
        # Fallback to GitHub version paths
        birdnet_cfg.MODEL_PATH = birdnet_cfg.BIRDNET_MODEL_PATH
        birdnet_cfg.LABELS_FILE = birdnet_cfg.BIRDNET_LABELS_FILE
        print("✅ Using BirdNET-Analyzer package models")
    else:
        raise RuntimeError(
            "BirdNET models not found. Please run:\n"
            "  from chirpkit.model_downloader import ModelDownloader\n"
            "  ModelDownloader.download_model('birdnet')"
        )

    # ... rest of config ...
    birdnet_model.load_model(class_output=False)
```

## What Changed

1. **Priority 1**: Check ChirpKit's downloaded models directory (`~/.chirpkit/models/birdnet/`)
2. **Priority 2**: Fallback to BirdNET-Analyzer package paths (for development)
3. **Priority 3**: Raise clear error if models not found

## Testing

### Before Fix
```bash
$ python -c "from src.chirpkit.transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor; BirdNETEmbeddingExtractor()"
# Would fail looking for models in BirdNET-Analyzer package directory
ValueError: Could not open '/path/to/birdnet_analyzer/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite'
```

### After Fix
```bash
$ python -c "from src.chirpkit.transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor; BirdNETEmbeddingExtractor()"
✅ Using BirdNET-Analyzer (GitHub version with model files)
🔧 Initializing BirdNET embedding extractor...
✅ Using ChirpKit downloaded models from: /Users/user/.chirpkit/models/birdnet
📦 Loading BirdNET model from: /Users/user/.chirpkit/models/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite
✅ BirdNET embedding extractor ready!
   Embedding dim: 1024
```

## Impact

### Before
- ❌ Downloaded models not used
- ❌ BirdNET-Analyzer looks in wrong location
- ❌ Fails with ValueError

### After
- ✅ Uses ChirpKit's downloaded models
- ✅ Works with ModelDownloader workflow
- ✅ Clear error messages if models missing

## For SoundCurious

This fix ensures that after running:
```python
from chirpkit.model_downloader import ModelDownloader
ModelDownloader.download_all_models()
```

The downloaded models in `~/.chirpkit/models/birdnet/` will be properly used by BirdNET-Analyzer.

---

**Fixed:** November 6, 2025
**Status:** ✅ COMPLETE - Tested and working
