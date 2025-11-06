# ChirpKit Installation Guide

## Quick Start (Recommended)

### For Production Use

```bash
# Install ChirpKit
pip install git+https://github.com/prossm/chirpkit.git@main

# Models will auto-download on first use
# Or manually download:
python -c "from chirpkit.model_downloader import ModelDownloader; ModelDownloader.download_all_models()"
```

### For Development

```bash
# Clone with model files
git clone https://github.com/prossm/chirpkit.git
cd chirpkit
git lfs pull  # Download model files (~50MB)

# Install in development mode
pip install -e .
```

---

## Docker Installation

### Method 1: Auto-Download (Recommended)

```dockerfile
FROM python:3.11-slim

# Install ChirpKit
RUN pip install git+https://github.com/prossm/chirpkit.git@main

# Models will be downloaded automatically on first use
# Container needs internet access for initial download
```

### Method 2: Pre-Download Models

```dockerfile
FROM python:3.11-slim

# Install Git LFS
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Clone repository with models
RUN git clone https://github.com/prossm/chirpkit.git /tmp/chirpkit && \
    cd /tmp/chirpkit && \
    git lfs pull

# Install ChirpKit
RUN pip install -e /tmp/chirpkit

# Models are now bundled in the image
```

### Method 3: Mount Models as Volume

```dockerfile
FROM python:3.11-slim

RUN pip install git+https://github.com/prossm/chirpkit.git@main

# On host machine, download models once:
# python -c "from chirpkit.model_downloader import ModelDownloader; ModelDownloader.download_all_models()"

# Then mount the models directory in docker-compose.yml:
# volumes:
#   - ~/.chirpkit/models:/root/.chirpkit/models:ro
```

---

## Model Files

ChirpKit requires two model files (~50MB total):

### 1. BirdNET Embedding Extractor (~25MB)
- File: `BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite`
- Location: `models/birdnet/`
- Auto-downloads from: GitHub Release v0.2.0

### 2. ChirpKit Ensemble (~19MB)
- Files: 7 ensemble models + metadata
- Location: `models/trained/chirpkit-ensemble/`
- Auto-downloads from: GitHub Release v0.2.0

---

## Download Methods

ChirpKit uses multiple fallback methods to ensure reliable model downloads:

### Method 1: GitHub Release (Fast, Recommended)
- Downloads pre-packaged ZIP files
- Fastest method (~2 seconds)
- Requires: Internet access

### Method 2: Git LFS Fallback (Automatic)
- Clones repository and pulls LFS files
- Slower but more reliable (~1 minute)
- Requires: `git` and `git-lfs` installed

### Method 3: Manual Installation
If automatic downloads fail:

```bash
# Install Git LFS
# Ubuntu/Debian:
sudo apt-get install git-lfs

# macOS:
brew install git-lfs

# Clone and get models
git clone https://github.com/prossm/chirpkit.git
cd chirpkit
git lfs pull

# Install
pip install -e .
```

---

## Troubleshooting

### Error: "Model provided has model identifier 'ion ', should be 'TFL3'"

**Cause:** Corrupted or incomplete BirdNET model file

**Solution:**
```bash
# Force re-download the model
python -c "from chirpkit.model_downloader import ModelDownloader; ModelDownloader.download_model('birdnet', force=True)"

# Or manually verify the file
cd models/birdnet
ls -lh BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite  # Should be ~25MB
file BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite    # Should show "data"
```

### Error: "No module named 'models.chirpkit_ensemble'"

**Cause:** Using an older version of ChirpKit with outdated import paths

**Solution:**
```bash
# Update to latest version
pip install --upgrade --force-reinstall git+https://github.com/prossm/chirpkit.git@main
```

### Error: "HTTP Error 404: Not Found"

**Cause:** GitHub release v0.2.0 hasn't been created yet

**Solution:** Git LFS fallback will automatically activate. Ensure you have `git` and `git-lfs` installed:
```bash
# Ubuntu/Debian
sudo apt-get install git git-lfs

# macOS
brew install git git-lfs

# Verify installation
git lfs version
```

### Models Won't Download in Docker

**Solution 1:** Use pre-download method (see Docker Method 2 above)

**Solution 2:** Ensure container has internet access:
```bash
docker run --rm your-image python -c "import requests; print(requests.get('https://google.com').status_code)"
# Should print: 200
```

---

## For SoundCurious Integration

### Update Existing Installation

```bash
# In your Docker container or environment:
pip install --upgrade --force-reinstall git+https://github.com/prossm/chirpkit.git@main

# Verify models are present:
python -c "from chirpkit.model_downloader import ModelDownloader; ModelDownloader.list_available_models()"
```

### Recommended Docker Setup

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install ChirpKit
RUN pip install git+https://github.com/prossm/chirpkit.git@main

# Pre-download models during build (requires internet)
RUN python -c "from chirpkit.model_downloader import ModelDownloader; ModelDownloader.download_all_models()"

# Your application code
COPY . /app
WORKDIR /app

CMD ["python", "your_app.py"]
```

---

## Verification

After installation, verify everything works:

```python
from chirpkit import InsectClassifier

# Initialize classifier
classifier = InsectClassifier()

# Load models (will auto-download if needed)
success = classifier.load_model()

if success:
    print("✅ ChirpKit installed and ready!")
    print(f"   Species: {classifier.n_classes}")
    print(f"   Model version: v0.2.0")
else:
    print("❌ Model loading failed - check logs above")
```

---

## Support

If you encounter issues not covered here:

1. Check model download status:
   ```bash
   python -c "from chirpkit.model_downloader import ModelDownloader; ModelDownloader.list_available_models()"
   ```

2. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. Report issues at: https://github.com/prossm/chirpkit/issues
