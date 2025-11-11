# Installation Notes

## Updated Dependency Versions (January 2025)

### Security Updates
- ✅ **requests** upgraded to >=2.32.4 (fixes CVE: GHSA-9wx4-h78v-vm56, GHSA-9hjg-9r4m-mvj7)
- ✅ All dependencies updated to latest secure versions

### Key Changes Made

#### 1. TensorFlow Version Constraints
**Problem:** `tensorflow-macos` only goes up to version 2.16.2, but requirements were open-ended.

**Solution:**
```python
# requirements.txt
tensorflow>=2.12.0; sys_platform != "darwin"
tensorflow-macos>=2.12.0,<=2.16.2; sys_platform == "darwin"

# setup.py
'tensorflow-macos': [
    'tensorflow-macos>=2.12.0,<=2.16.2',
    'tensorflow-metal>=1.0.0',
]
```

#### 2. Gradio Version Update
**Problem:** Requirements specified `gradio>=3.0.0` but Gradio 5.x has breaking changes and requires `python-multipart>=0.0.20`.

**Solution:**
```python
gradio>=5.0.0
python-multipart>=0.0.20  # Required for file uploads
```

#### 3. Platform-Specific Dependencies
Added proper platform markers to ensure correct TensorFlow installation:
- macOS: `tensorflow-macos` + `tensorflow-metal`
- Linux/Windows: standard `tensorflow`

### Tested Working Versions (macOS Apple Silicon)
```
tensorflow-macos==2.16.2
tensorflow-metal==1.2.0
torch==2.9.0
gradio==5.49.1
python-multipart==0.0.20
numpy==1.26.4
librosa==0.11.0
scikit-learn==1.6.1
```

## Fresh Install Instructions

### macOS (Apple Silicon/Intel)
```bash
# Create virtual environment
python -m venv chirpkit_env
source chirpkit_env/bin/activate

# Install ChirpKit with all dependencies
pip install -e .[full]

# Or install from requirements.txt
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import gradio; print('Gradio:', gradio.__version__)"
```

### Linux/Windows
```bash
# Create virtual environment
python -m venv chirpkit_env
source chirpkit_env/bin/activate  # Linux
# OR
chirpkit_env\Scripts\activate  # Windows

# Install ChirpKit
pip install -e .[full]

# Verify
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

## Known Issues

### 1. TensorFlow Version Conflicts
**Symptom:** `ModuleNotFoundError: No module named 'tensorflow'` even though pip shows it installed.

**Cause:** Multiple Python environments (conda + venv) or corrupted installation.

**Solution:**
```bash
# Check which Python is active
which python
python --version

# Reinstall TensorFlow
pip uninstall -y tensorflow tensorflow-macos
pip install tensorflow-macos  # macOS
# OR
pip install tensorflow  # Linux/Windows
```

### 2. Gradio File Upload Errors
**Symptom:** `ModuleNotFoundError: No module named 'python_multipart'`

**Solution:**
```bash
pip install python-multipart>=0.0.20
```

### 3. BirdNET-Analyzer Dependency
**Note:** Must be installed from GitHub, not PyPI:
```bash
pip install git+https://github.com/kahst/BirdNET-Analyzer.git
```

## Model Storage Configuration

### Default Location
Models are downloaded to `~/.chirpkit/models/` by default.

### Custom Locations (Docker/Containers)
```bash
# Set environment variable
export CHIRPKIT_MODEL_DIR=/custom/path

# Or use CHIRPKIT_HOME
export CHIRPKIT_HOME=/custom/base
# Models will be in /custom/base/models
```

### Programmatic Configuration
```python
import os
os.environ['CHIRPKIT_MODEL_DIR'] = '/custom/path'
from chirpkit import InsectClassifier
```

## Testing Your Installation

```bash
# Test model download and classification
python simple_ui.py

# Access web UI at http://localhost:7860
# Upload an insect sound recording to test
```

## Docker Installation

```dockerfile
FROM python:3.11-slim

# Set custom model directory
ENV CHIRPKIT_MODEL_DIR=/models/chirpkit

# Create volume
VOLUME /models/chirpkit

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install ChirpKit
COPY . /app
WORKDIR /app
RUN pip install -e .[full]

# Download models on first run
CMD ["python", "simple_ui.py"]
```

## Troubleshooting

### Clear Pip Cache
```bash
pip cache purge
pip install --no-cache-dir -e .[full]
```

### Check Installed Versions
```bash
pip list | grep -E "tensorflow|torch|gradio|chirpkit"
```

### Health Check
```bash
chirpkit-doctor  # If CLI is installed
```
