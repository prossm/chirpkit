"""
ChirpKit Model Downloader

Downloads pre-trained models on first use from GitHub releases.
Models are cached in ~/.chirpkit/models/ for subsequent use.
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
import json
import zipfile
import tempfile
import shutil
import subprocess

from .utils import get_chirpkit_logger

logger = get_chirpkit_logger(__name__)


def get_default_cache_dir():
    """
    Get the default cache directory for ChirpKit models.

    Respects environment variables for custom cache locations:
    - CHIRPKIT_MODEL_DIR: Primary override for model storage location
    - CHIRPKIT_HOME: Secondary override (models stored in {CHIRPKIT_HOME}/models)

    If no environment variables are set, defaults to ~/.chirpkit/models

    Returns:
        Path: Directory where models should be stored

    Examples:
        # Use default location
        >>> get_default_cache_dir()
        PosixPath('/Users/username/.chirpkit/models')

        # With environment variable
        >>> os.environ['CHIRPKIT_MODEL_DIR'] = '/models/chirpkit'
        >>> get_default_cache_dir()
        PosixPath('/models/chirpkit')
    """
    # Environment variable takes precedence
    if env_dir := os.environ.get('CHIRPKIT_MODEL_DIR'):
        return Path(env_dir)

    # Secondary: CHIRPKIT_HOME/models
    if chirpkit_home := os.environ.get('CHIRPKIT_HOME'):
        return Path(chirpkit_home) / 'models'

    # Fallback to home directory
    return Path.home() / '.chirpkit' / 'models'


class ModelDownloader:
    """Download ChirpKit models on first use"""

    # Model download URLs (GitHub Releases)
    # Note: Release version should match package version (v0.2.0)
    MODELS = {
        'chirpkit-ensemble': {
            'url': 'https://github.com/prossm/chirpkit/releases/download/v0.2.0/chirpkit-ensemble.zip',
            'size_mb': 19,
            'description': 'ChirpKit ensemble model v6.0 - 7-model ensemble (79.7% accuracy)',
            'files': [
                'ensemble_model_1.pth',
                'ensemble_model_2.pth',
                'ensemble_model_3.pth',
                'ensemble_model_4.pth',
                'ensemble_model_5.pth',
                'ensemble_model_6.pth',
                'ensemble_model_7.pth',
                'ensemble_info.json',
                'label_encoder.joblib'
            ]
        },
        'birdnet': {
            'url': 'https://github.com/prossm/chirpkit/releases/download/v0.2.0/birdnet-models.zip',
            'size_mb': 25,
            'description': 'BirdNET v2.4 embedding extractor',
            'files': [
                'BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite',
                'BirdNET_GLOBAL_6K_V2.4_Labels.txt'
            ]
        }
    }

    @staticmethod
    def get_models_dir(cache_dir=None):
        """
        Get the directory where models should be stored.

        Args:
            cache_dir: Optional custom cache directory path. If not provided,
                      uses get_default_cache_dir() which respects environment variables.

        Returns:
            Path: Directory where models are stored
        """
        # Use explicit cache_dir if provided
        if cache_dir is not None:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            return cache_path

        # Try to use package directory first (for development)
        package_dir = Path(__file__).parent.parent.parent.parent / "models"
        if package_dir.exists() and (package_dir / "trained").exists():
            # Development mode - models are in repo
            return package_dir

        # Production mode - use environment-aware cache directory
        home_dir = get_default_cache_dir()
        home_dir.mkdir(parents=True, exist_ok=True)
        return home_dir

    @staticmethod
    def check_model_exists(model_name, cache_dir=None):
        """
        Check if a model is already downloaded.

        Args:
            model_name: Name of the model to check
            cache_dir: Optional custom cache directory

        Returns:
            bool: True if model exists, False otherwise
        """
        model_dir = ModelDownloader.get_models_dir(cache_dir)

        if model_name == 'chirpkit-ensemble':
            ensemble_dir = model_dir / "trained" / "chirpkit-ensemble"
            ensemble_info = ensemble_dir / "ensemble_info.json"
            return ensemble_info.exists()
        elif model_name == 'birdnet':
            birdnet_dir = model_dir / "birdnet"
            tflite_model = birdnet_dir / "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
            return tflite_model.exists()

        return False

    @staticmethod
    def get_model_path(model_name, cache_dir=None):
        """
        Get the path to a model directory.

        Args:
            model_name: Name of the model
            cache_dir: Optional custom cache directory

        Returns:
            Path: Path to the model directory

        Raises:
            ValueError: If model name is unknown
        """
        model_dir = ModelDownloader.get_models_dir(cache_dir)

        if model_name == 'chirpkit-ensemble':
            return model_dir / "trained" / "chirpkit-ensemble"
        elif model_name == 'birdnet':
            return model_dir / "birdnet"

        raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def _download_from_github_release(model_name, model_info, model_path):
        """Download model from GitHub release (primary method)"""
        logger.info(f"📥 Downloading {model_info['description']}...")
        logger.info(f"   Size: ~{model_info['size_mb']}MB (one-time download)")
        logger.info(f"   Source: GitHub Release")
        logger.info(f"   URL: {model_info['url']}")

        # Create temp file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
            tmp_path = tmp.name

        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            print(f"\r   Progress: {percent:.1f}%", end='', flush=True)

        urllib.request.urlretrieve(model_info['url'], tmp_path, show_progress)
        print()  # New line after progress

        # Extract
        print(f"📦 Extracting to {model_path}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(model_path.parent)

        # Clean up temp file
        os.unlink(tmp_path)

        # Verify all expected files are present
        missing_files = []
        for file_name in model_info['files']:
            if not (model_path / file_name).exists():
                missing_files.append(file_name)

        if missing_files:
            raise FileNotFoundError(f"Missing files after extraction: {missing_files}")

        print(f"✅ {model_name} downloaded successfully!")
        return model_path

    @staticmethod
    def _download_via_git_lfs(model_name, model_path):
        """Fallback: Clone repository with LFS and copy models"""
        print(f"📥 Fallback: Cloning ChirpKit repository with Git LFS...")
        print(f"   This may take a few minutes...")

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Clone repository
                print("   Cloning repository...")
                subprocess.run(
                    ['git', 'clone', '--depth', '1', 'https://github.com/prossm/chirpkit.git', tmpdir],
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Pull LFS files
                print("   Pulling LFS files...")
                subprocess.run(
                    ['git', 'lfs', 'pull'],
                    cwd=tmpdir,
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Copy models to cache
                if model_name == 'birdnet':
                    src_path = Path(tmpdir) / 'models' / 'birdnet'
                    dst_path = model_path
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                elif model_name == 'chirpkit-ensemble':
                    src_path = Path(tmpdir) / 'models' / 'trained' / 'chirpkit-ensemble'
                    dst_path = model_path
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

                print(f"✅ {model_name} downloaded via Git LFS!")
                return model_path

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Git command failed: {e.stderr}")
            except Exception as e:
                raise RuntimeError(f"Git LFS download failed: {e}")

    @staticmethod
    def download_model(model_name, force=False, cache_dir=None):
        """
        Download a specific model if not present.
        Tries multiple download methods with automatic fallback:
        1. GitHub Release (fast, recommended)
        2. Git LFS clone (slower fallback)

        Args:
            model_name: Name of model ('chirpkit-ensemble' or 'birdnet')
            force: Force re-download even if model exists
            cache_dir: Optional custom directory for model storage.
                      If not provided, uses environment variables or default location.

        Returns:
            Path to downloaded model directory

        Environment Variables:
            CHIRPKIT_MODEL_DIR: Override model storage location
            CHIRPKIT_HOME: Alternative override ({CHIRPKIT_HOME}/models)
        """
        if model_name not in ModelDownloader.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(ModelDownloader.MODELS.keys())}")

        model_info = ModelDownloader.MODELS[model_name]
        model_path = ModelDownloader.get_model_path(model_name, cache_dir)

        # Check if already downloaded
        if not force and ModelDownloader.check_model_exists(model_name, cache_dir):
            print(f"✅ {model_name} already downloaded at {model_path}")
            return model_path

        # Try download methods in order of preference
        download_methods = [
            {
                'name': 'GitHub Release',
                'func': lambda: ModelDownloader._download_from_github_release(model_name, model_info, model_path)
            },
            {
                'name': 'Git LFS',
                'func': lambda: ModelDownloader._download_via_git_lfs(model_name, model_path)
            }
        ]

        last_error = None
        for method in download_methods:
            try:
                return method['func']()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    logger.warning(f"{method['name']} failed: Model not found (404)")
                    print(f"⚠️  {method['name']} failed: Model not found at URL")
                    last_error = e
                else:
                    logger.warning(f"{method['name']} failed: HTTP {e.code}")
                    print(f"⚠️  {method['name']} failed: HTTP Error {e.code}")
                    last_error = e
            except Exception as e:
                logger.warning(f"{method['name']} failed: {e}")
                print(f"⚠️  {method['name']} failed: {e}")
                last_error = e

            # If this wasn't the last method, try next one
            if method != download_methods[-1]:
                print(f"   Trying next download method...")

        # All methods failed
        print(f"\n❌ All download methods failed for {model_name}")
        print(f"\n💡 Manual installation instructions:")
        print(f"   1. Install Git LFS: https://git-lfs.github.com/")
        print(f"   2. Clone repository:")
        print(f"      git clone https://github.com/prossm/chirpkit.git")
        print(f"      cd chirpkit && git lfs pull")
        print(f"   3. Install in development mode:")
        print(f"      pip install -e .")

        if last_error:
            raise last_error
        else:
            raise RuntimeError(f"Failed to download {model_name}")

    @staticmethod
    def download_model_legacy(model_name, force=False):
        """
        Legacy download method (kept for reference).
        Use download_model() instead which has automatic fallbacks.
        """
        if model_name not in ModelDownloader.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(ModelDownloader.MODELS.keys())}")

        model_info = ModelDownloader.MODELS[model_name]
        model_path = ModelDownloader.get_model_path(model_name)

        # Check if already downloaded
        if not force and ModelDownloader.check_model_exists(model_name):
            print(f"✅ {model_name} already downloaded at {model_path}")
            return model_path

        print(f"📥 Downloading {model_info['description']}...")
        print(f"   Size: ~{model_info['size_mb']}MB (one-time download)")
        print(f"   URL: {model_info['url']}")

        try:
            # Create temp file for download
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                tmp_path = tmp.name

            # Download with progress
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
                print(f"\r   Progress: {percent:.1f}%", end='', flush=True)

            urllib.request.urlretrieve(model_info['url'], tmp_path, show_progress)
            print()  # New line after progress

            # Extract
            print(f"📦 Extracting to {model_path}...")
            model_path.parent.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(model_path.parent)

            # Clean up temp file
            os.unlink(tmp_path)

            # Verify all expected files are present
            missing_files = []
            for file_name in model_info['files']:
                if not (model_path / file_name).exists():
                    missing_files.append(file_name)

            if missing_files:
                print(f"⚠️  Warning: Some files missing: {missing_files}")

            print(f"✅ {model_name} downloaded successfully!")
            return model_path

        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"\n❌ Model not found at {model_info['url']}")
                print(f"   The model may not be released yet.")
                print(f"\n💡 For now, please use models from the git repository:")
                print(f"   git clone https://github.com/prossm/chirpkit.git")
                print(f"   cd chirpkit")
                print(f"   git lfs pull  # Download model files")
            else:
                print(f"\n❌ HTTP Error {e.code}: {e.reason}")
            raise

        except Exception as e:
            print(f"\n❌ Failed to download {model_name}: {e}")
            print(f"\n💡 Manual installation:")
            print(f"   1. Download from: {model_info['url']}")
            print(f"   2. Extract to: {model_path}")
            print(f"   Or clone the full repository with git lfs")
            raise

    @staticmethod
    def download_all_models(force=False, cache_dir=None):
        """
        Download all required models.

        Args:
            force: Re-download even if models exist
            cache_dir: Custom directory for model storage (default: uses environment variables or ~/.chirpkit/models)

        Returns:
            bool: True if all models downloaded successfully, False otherwise

        Environment Variables:
            CHIRPKIT_MODEL_DIR: Override model storage location
            CHIRPKIT_HOME: Alternative override ({CHIRPKIT_HOME}/models)

        Examples:
            # Download to default location
            ModelDownloader.download_all_models()

            # Download to custom location
            ModelDownloader.download_all_models(cache_dir='/models/chirpkit')

            # Use environment variable
            os.environ['CHIRPKIT_MODEL_DIR'] = '/models/chirpkit'
            ModelDownloader.download_all_models()
        """
        for model_name in ModelDownloader.MODELS:
            try:
                ModelDownloader.download_model(model_name, force=force, cache_dir=cache_dir)
            except Exception as e:
                print(f"Failed to download {model_name}: {e}")
                return False
        return True

    @staticmethod
    def list_available_models():
        """List all available models and their download status"""
        print("📋 ChirpKit Models:")
        print("=" * 60)

        for model_name, info in ModelDownloader.MODELS.items():
            is_downloaded = ModelDownloader.check_model_exists(model_name)
            status = "✅ Downloaded" if is_downloaded else "❌ Not downloaded"
            path = ModelDownloader.get_model_path(model_name)

            print(f"\n{model_name}:")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size_mb']}MB")
            print(f"  Status: {status}")
            print(f"  Path: {path}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    # CLI usage
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            ModelDownloader.list_available_models()
        elif command == "download":
            if len(sys.argv) > 2:
                model_name = sys.argv[2]
                ModelDownloader.download_model(model_name)
            else:
                ModelDownloader.download_all_models()
        elif command == "check":
            ModelDownloader.list_available_models()
        else:
            print("Usage: python model_downloader.py [list|download|check] [model_name]")
    else:
        print("ChirpKit Model Downloader")
        print("=" * 60)
        ModelDownloader.list_available_models()
