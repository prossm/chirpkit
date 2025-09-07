# ChirpKit Version Management System

## Overview
ChirpKit now uses a centralized version management system to ensure consistency across all components.

## Implementation

### Central Version File
- **File**: `src/chirpkit/_version.py`
- **Current Version**: `0.1.2`
- **Usage**: Single source of truth for version information

### Updated Components

#### 1. Package Initialization (`src/chirpkit/__init__.py`)
```python
from ._version import __version__
```

#### 2. Setup Configuration (`setup.py`)
- Automatically reads version from `_version.py`
- No hardcoded version numbers

#### 3. CLI Tool (`src/chirpkit/cli.py`)
```python
from ._version import __version__
version=f'chirpkit {__version__}'
```

#### 4. Wikipedia Integration (`src/chirpkit/classifier.py`)
```python
from ._version import __version__
'User-Agent': f'ChirpKit/{__version__} (https://github.com/patrickmetzger/chirpkit; contact@chirpkit.ai) Wikipedia Integration'
```

#### 5. Web UI (`simple_ui.py`)
```python
from src.chirpkit._version import __version__
'User-Agent': f'ChirpKit/{__version__} (...)'
```

#### 6. Model Downloads (`src/chirpkit/models.py`)
```python
from ._version import __version__
REMOTE_MODEL_BASE_URL = f"https://github.com/patrickmetzger/chirpkit/releases/download/v{__version__}/"
```

## Benefits

1. **Single Source of Truth**: Version defined in one place only
2. **Automatic Consistency**: All components use the same version
3. **Easy Updates**: Change version in one file, affects everywhere
4. **Proper Attribution**: Wikipedia API calls include correct version
5. **Accurate Downloads**: Model downloads use correct release tags

## How to Update Version

To update ChirpKit version:

1. Edit `src/chirpkit/_version.py`:
```python
__version__ = "0.2.0"  # New version
```

2. All components automatically use the new version:
   - Package metadata
   - CLI version output  
   - User-Agent headers
   - Download URLs
   - Documentation

## Testing

```bash
# Test version consistency
python -c "from chirpkit import __version__; print(f'Version: {__version__}')"

# Test setup.py version
python setup.py --version

# Test CLI version
python -m chirpkit.cli --version
```

All should return the same version number: `0.1.2`