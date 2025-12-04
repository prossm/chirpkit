# ChirpKit Model Configuration Migration Guide

This guide helps you migrate from ChirpKit's legacy model configuration to the new flexible configuration system that addresses the issues outlined in your modularity enhancement proposal.

## What Changed

### Before (Limited Flexibility)
```python
# Old way - limited options
from chirpkit import InsectClassifier

# Only these options were available:
classifier = InsectClassifier()  # Default paths only
classifier = InsectClassifier(model_path="/path/to/ensemble")  # Ensemble only

# Environment variables were inconsistent:
os.environ['CHIRPKIT_MODEL_DIR'] = '/models'  # Only supported for some models
```

### After (Full Flexibility)
```python
# New way - comprehensive configuration
from chirpkit import InsectClassifier

# All the options from your proposal:
classifier = InsectClassifier(
    model_root="/models/chirpkit",           # Root directory for all models
    birdnet_model_path="/models/chirpkit/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite",
    ensemble_path="/models/chirpkit/trained/chirpkit-ensemble",
    auto_download=False,                      # Disable auto-download if models exist
    validate_compatibility=True               # Check model dimensions match
)
```

## Migration Scenarios

### 1. SoundCurious Use Case (Your Original Request)

**Problem:** Re-downloads models even when they exist elsewhere

**Old Approach:**
```python
# Had to use symbolic links or copy models to default location
# Models would still be re-downloaded
classifier = InsectClassifier()
```

**New Solution:**
```python
# Option 1: Use existing downloaded models
classifier = InsectClassifier(
    model_root="/models/chirpkit",
    auto_download=False,  # Don't download if models exist
    validate_compatibility=True
)

# Option 2: Explicit paths for full control
classifier = InsectClassifier(
    birdnet_model_path="/models/chirpkit/birdnet/BirdNET_v2_3_1024dim.tflite",
    ensemble_path="/models/chirpkit/trained/chirpkit-ensemble"
)

# Option 3: Environment variables
export CHIRPKIT_ROOT_DIR="/models/chirpkit"
export CHIRPKIT_AUTO_DOWNLOAD="false"
# Then just: classifier = InsectClassifier()
```

### 2. Docker/Container Deployments

**Old Approach:**
```dockerfile
# Limited to default paths
ENV CHIRPKIT_MODEL_DIR=/models/chirpkit
VOLUME /models/chirpkit
```

**New Solution:**
```dockerfile
# Much more flexible
ENV CHIRPKIT_ROOT_DIR=/models/chirpkit
ENV CHIRPKIT_AUTO_DOWNLOAD=false
ENV CHIRPKIT_VALIDATE_COMPATIBILITY=true
VOLUME /models/chirpkit

# Or use configuration file
COPY chirpkit.config.yaml /app/
```

### 3. Configuration File Support

**New Feature:**
```yaml
# ~/.chirpkit/config.yaml
models:
  root_directory: "/models/chirpkit"
  birdnet:
    model_path: "birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
  ensemble:
    path: "trained/chirpkit-ensemble"
    mode: "ensemble_tta"
  download:
    auto_download: false
```

```python
# Configuration automatically loaded
classifier = InsectClassifier()
```

## Step-by-Step Migration

### Step 1: Assess Your Current Setup

1. **Find your current model location:**
   ```python
   from chirpkit import get_default_cache_dir
   print(get_default_cache_dir())  # Shows current default location
   ```

2. **Check what models you have:**
   ```python
   from chirpkit import ModelDownloader
   ModelDownloader.list_available_models()
   ```

### Step 2: Choose Your Migration Path

#### Path A: Keep Current Location (Easiest)
If you're happy with the current location, no changes needed:
```python
# This still works exactly as before
classifier = InsectClassifier()
```

#### Path B: Move to Custom Location
```python
# 1. Set new location
export CHIRPKIT_ROOT_DIR="/your/preferred/location"

# 2. Move existing models (optional)
mv ~/.chirpkit/models/* /your/preferred/location/

# 3. Use ChirpKit normally
classifier = InsectClassifier()
```

#### Path C: Use Explicit Paths (Maximum Control)
```python
classifier = InsectClassifier(
    birdnet_model_path="/exact/path/to/birdnet.tflite",
    ensemble_path="/exact/path/to/ensemble/directory"
)
```

### Step 3: Update Your Code

#### Environment Variables
```bash
# Old (still works)
export CHIRPKIT_MODEL_DIR="/models"

# New (recommended)
export CHIRPKIT_ROOT_DIR="/models/chirpkit"
export CHIRPKIT_BIRDNET_MODEL="/models/chirpkit/birdnet/model.tflite"
export CHIRPKIT_ENSEMBLE_DIR="/models/chirpkit/trained/chirpkit-ensemble"
export CHIRPKIT_AUTO_DOWNLOAD="false"
```

#### Constructor Changes
```python
# Old way (still supported)
classifier = InsectClassifier(model_path="/path/to/ensemble")

# New way (more flexible)
classifier = InsectClassifier(
    model_root="/models/chirpkit",  # Auto-discovers models
    mode="ensemble_tta",
    auto_download=False
)

# Or explicit paths
classifier = InsectClassifier(
    birdnet_model_path="/models/birdnet/model.tflite",
    ensemble_path="/models/ensemble/",
    validate_compatibility=True
)
```

## Configuration Priority (Highest to Lowest)

1. **Constructor parameters** (your code)
2. **Environment variables** 
3. **Configuration file**
4. **Defaults**

Example:
```python
# Environment says ensemble, but constructor overrides to ensemble_tta
os.environ['CHIRPKIT_MODE'] = 'ensemble'
classifier = InsectClassifier(mode='ensemble_tta')  # Uses ensemble_tta
```

## Validation and Troubleshooting

### Enable Validation
```python
classifier = InsectClassifier(validate_compatibility=True)
```

### Model Discovery
```python
from chirpkit import ModelDiscovery

# See what models ChirpKit can find
discovered = ModelDiscovery.find_models("/your/model/directory")
selection = ModelDiscovery.select_best_models(discovered)
print(f"ChirpKit would select: {selection}")
```

### Configuration Debugging
```python
from chirpkit import ConfigurationManager

config_manager = ConfigurationManager()
config = config_manager.resolve_configuration()
print(f"Final configuration: {config}")
```

## Benefits After Migration

✅ **No more hard-coded paths:** ChirpKit defaults to ~/.chirpkit/models/ with limited flexibility  
✅ **No redundant downloads:** Re-downloads models even when they exist elsewhere  
✅ **Environment variables work consistently:** Path conflicts across all model types resolved  
✅ **Model validation:** Verifies model compatibility before loading  
✅ **Configuration files:** Support for YAML/JSON configuration  
✅ **Smart model discovery:** Automatically finds compatible models  
✅ **Docker-friendly:** Works perfectly with containers and volumes  
✅ **Backwards compatible:** All existing code continues to work  

## Need Help?

1. **Run diagnostics:**
   ```bash
   chirpkit-doctor
   ```

2. **Check configuration:**
   ```python
   from chirpkit import create_example_config_file
   create_example_config_file("~/my_chirpkit_config.yaml")
   ```

3. **Test model discovery:**
   ```python
   python examples/model_configuration_examples.py
   ```

This new system solves all the issues you identified in your modularity enhancement proposal while maintaining complete backwards compatibility!