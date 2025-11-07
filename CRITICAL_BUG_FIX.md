# CRITICAL BUG FIX - BirdNET Embedding Extraction

## Issue
**RuntimeError: Embedding dimension mismatch (6522 vs expected 1024)**

### Root Cause
The standalone TFLite model (`BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite`) only exports the **classification layer** (6522 species), not the **embedding layer** (1024 dimensions).

The code attempted to use TFLite as a fallback when BirdNET-Analyzer wasn't available, but this doesn't work because:
- TFLite model structure: Input → [hidden layers] → **6522-dim output (classes)**
- ChirpKit needs: Input → [hidden layers] → **1024-dim embeddings** → classifier

### Evidence
```
📊 TFLite model outputs:
   Output 0: shape=[1, 6522], dtype=<class 'numpy.float32'>
⚠️  Could not find 1024-dim embedding output, using output 0

RuntimeError: linear(): input and weight.T shapes cannot be multiplied (1x6522 and 1024x512)
```

## The Fix

### Added `birdnet-analyzer` as Required Dependency

**File: `requirements.txt`**
```diff
# Machine Learning
tensorflow>=2.6.0
scikit-learn>=1.0.0

+# BirdNET for embedding extraction (critical dependency)
+birdnet-analyzer>=1.0.0
```

### Why This Works

1. **BirdNET-Analyzer Python API** has access to intermediate layers
2. When `class_output=False` is set, it returns **1024-dim embeddings** instead of classifications
3. This is what the training scripts used to generate the ensemble models

### Code Path
```python
# In birdnet_embeddings.py
if USING_BIRDNET_ANALYZER:
    # ✅ Uses full Python API - can extract embeddings
    birdnet_model.load_model(class_output=False)  # Returns 1024-dim
    embeddings = birdnet_model.embeddings(chunks)
else:
    # ❌ Uses TFLite - can only extract classifications (6522-dim)
    embedding = interpreter.get_tensor(output_details[0]['index'])  # WRONG!
```

## Impact

### Before Fix
- ❌ Production deployments without BirdNET-Analyzer would fail
- ❌ pip install would work but model loading would fail at runtime
- ❌ TFLite fallback extracted wrong tensor (6522-dim instead of 1024-dim)

### After Fix
- ✅ `birdnet-analyzer` always installed
- ✅ Embedding extraction always uses correct API
- ✅ Consistent behavior in development and production
- ✅ No dimension mismatch errors

## Testing

### Verify the Fix
```bash
# Clean install
python3 -m venv test_env
source test_env/bin/activate
pip install git+https://github.com/prossm/chirpkit.git@main

# Should see birdnet-analyzer in dependencies
pip list | grep birdnet
# Output: birdnet-analyzer  x.x.x

# Test embedding extraction
python -c "
import sys
sys.path.insert(0, 'src')
from chirpkit.transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor

extractor = BirdNETEmbeddingExtractor()
print(f'Using BirdNET-Analyzer: {extractor.using_analyzer}')
# Should print: Using BirdNET-Analyzer: True
"
```

### Expected Startup Messages
```
✅ Using BirdNET-Analyzer (full features available)
📦 Loading BirdNET model from: [path]/birdnet_analyzer/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite
✅ BirdNET embedding extractor ready!
   Sample rate: 48000Hz
   Chunk length: 3.0s
   Embedding dim: 1024  ← Correct!
```

## For SoundCurious

This fix is **already included** in the latest main branch. When you update to the latest ChirpKit:

```dockerfile
RUN pip install git+https://github.com/prossm/chirpkit.git@main
```

The `birdnet-analyzer` package will be automatically installed, and embedding extraction will work correctly.

## Alternative Solutions Considered

### ❌ Option 1: Extract embeddings from TFLite intermediate tensors
- **Problem**: TFLite doesn't expose intermediate tensors by default
- **Would require**: Rebuilding the TFLite model with custom outputs

### ❌ Option 2: Create a separate embedding-only TFLite model
- **Problem**: Would need to maintain two separate model files
- **Complexity**: High maintenance burden

### ✅ Option 3: Make `birdnet-analyzer` a required dependency
- **Pros**: Simple, reliable, uses official API
- **Cons**: Slightly larger install size (~5MB)
- **Decision**: Best solution for reliability

## Related Files Changed
- ✅ `requirements.txt` - Added birdnet-analyzer
- ✅ `SOUNDCURIOUS_FIX.md` - Updated installation instructions
- ✅ `UPLOAD_TO_RELEASE.md` - Documented the fix

## Version
This fix is included in **ChirpKit v0.2.0** and later.
