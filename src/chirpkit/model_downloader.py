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


class ModelDownloader:
    """Download ChirpKit models on first use"""

    # Model download URLs (GitHub Releases)
    # Note: v6.0 refers to the model version, not the package version
    MODELS = {
        'chirpkit-ensemble': {
            'url': 'https://github.com/prossm/chirpkit/releases/download/v6.0/chirpkit-ensemble.zip',
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
            'url': 'https://github.com/prossm/chirpkit/releases/download/v6.0/birdnet-models.zip',
            'size_mb': 25,
            'description': 'BirdNET v2.4 embedding extractor',
            'files': [
                'BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite',
                'BirdNET_GLOBAL_6K_V2.4_Labels.txt'
            ]
        }
    }

    @staticmethod
    def get_models_dir():
        """Get the directory where models should be stored"""
        # Try to use package directory first (for development)
        package_dir = Path(__file__).parent.parent.parent.parent / "models"
        if package_dir.exists() and (package_dir / "trained").exists():
            # Development mode - models are in repo
            return package_dir

        # Production mode - use user's home directory
        home_dir = Path.home() / ".chirpkit" / "models"
        home_dir.mkdir(parents=True, exist_ok=True)
        return home_dir

    @staticmethod
    def check_model_exists(model_name):
        """Check if a model is already downloaded"""
        model_dir = ModelDownloader.get_models_dir()

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
    def get_model_path(model_name):
        """Get the path to a model directory"""
        model_dir = ModelDownloader.get_models_dir()

        if model_name == 'chirpkit-ensemble':
            return model_dir / "trained" / "chirpkit-ensemble"
        elif model_name == 'birdnet':
            return model_dir / "birdnet"

        raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def download_model(model_name, force=False):
        """
        Download a specific model if not present.

        Args:
            model_name: Name of model ('chirpkit-ensemble' or 'birdnet')
            force: Force re-download even if model exists

        Returns:
            Path to downloaded model directory
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
    def download_all_models(force=False):
        """Download all required models"""
        for model_name in ModelDownloader.MODELS:
            try:
                ModelDownloader.download_model(model_name, force=force)
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
