"""
BirdNET Embedding Extraction for Insect Audio Classification

This module extracts feature embeddings from insect audio using the pre-trained
BirdNET model. These embeddings capture learned acoustic features from millions
of bird/animal sounds, which should transfer well to insect sounds.

Architecture:
    Input Audio (any sample rate)
    → Resample to 48kHz
    → 3-second chunks
    → BirdNET Feature Extractor (frozen)
    → 1024-dim embeddings
    → Insect Classifier Head (trainable)

Expected Performance Boost:
    - Current baseline: 37% on 255 species
    - With transfer learning: 45-55% (estimated 8-18% improvement)
    - BirdNET learned general audio features from millions of samples
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import joblib

# Add BirdNET to path
BIRDNET_PATH = Path(__file__).parent.parent.parent / "BirdNET-Analyzer"
sys.path.insert(0, str(BIRDNET_PATH))

import birdnet_analyzer.config as birdnet_cfg
from birdnet_analyzer import model as birdnet_model
from birdnet_analyzer import audio as birdnet_audio


class BirdNETEmbeddingExtractor:
    """Extract 1024-dimensional embeddings from audio using BirdNET"""

    def __init__(self):
        """Initialize BirdNET model for embedding extraction"""
        print("🔧 Initializing BirdNET embedding extractor...")

        # Configure BirdNET for embedding extraction
        birdnet_cfg.MODEL_PATH = birdnet_cfg.BIRDNET_MODEL_PATH
        birdnet_cfg.LABELS_FILE = birdnet_cfg.BIRDNET_LABELS_FILE
        birdnet_cfg.SAMPLE_RATE = birdnet_cfg.BIRDNET_SAMPLE_RATE  # 48kHz
        birdnet_cfg.SIG_LENGTH = birdnet_cfg.BIRDNET_SIG_LENGTH    # 3.0 seconds
        birdnet_cfg.SIG_OVERLAP = 0.0  # No overlap for training data
        birdnet_cfg.SIG_MINLEN = 1.0   # Minimum 1 second
        birdnet_cfg.BATCH_SIZE = 32    # Process multiple chunks at once

        # Set bandpass filter for insect frequencies
        # Insects typically produce sounds between 1-15kHz
        birdnet_cfg.BANDPASS_FMIN = 1000  # 1kHz
        birdnet_cfg.BANDPASS_FMAX = 15000  # 15kHz

        # Load model for embedding extraction (not classification)
        print(f"📦 Loading BirdNET model from: {birdnet_cfg.MODEL_PATH}")
        birdnet_model.load_model(class_output=False)  # False = get embeddings

        print("✅ BirdNET embedding extractor ready!")
        print(f"   Sample rate: {birdnet_cfg.SAMPLE_RATE}Hz")
        print(f"   Chunk length: {birdnet_cfg.SIG_LENGTH}s")
        print(f"   Embedding dim: 1024")
        print(f"   Bandpass filter: {birdnet_cfg.BANDPASS_FMIN}-{birdnet_cfg.BANDPASS_FMAX}Hz")

    def extract_embeddings_from_audio(self, audio_path, aggregate='mean'):
        """
        Extract embeddings from an audio file.

        Args:
            audio_path: Path to audio file
            aggregate: How to combine multiple 3-second chunks
                      'mean' - average all embeddings
                      'max' - max pool across embeddings
                      'concat' - concatenate first N embeddings
                      'all' - return all embeddings (for data augmentation)

        Returns:
            embeddings: numpy array of shape (1024,) or (n_chunks, 1024)
        """
        try:
            # Load audio and resample to 48kHz
            sig, rate = birdnet_audio.open_audio_file(
                audio_path,
                sample_rate=birdnet_cfg.SAMPLE_RATE,
                fmin=birdnet_cfg.BANDPASS_FMIN,
                fmax=birdnet_cfg.BANDPASS_FMAX
            )

            # Split into 3-second chunks
            chunks = self._split_signal(sig, rate)

            if len(chunks) == 0:
                # Audio too short, pad to 3 seconds
                chunks = [self._pad_signal(sig, rate)]

            # Extract embeddings for all chunks
            embeddings = birdnet_model.embeddings(chunks)  # Shape: (n_chunks, 1024)

            # Aggregate embeddings
            if aggregate == 'mean':
                return np.mean(embeddings, axis=0)  # (1024,)
            elif aggregate == 'max':
                return np.max(embeddings, axis=0)  # (1024,)
            elif aggregate == 'first':
                return embeddings[0]  # (1024,)
            elif aggregate == 'all':
                return embeddings  # (n_chunks, 1024)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregate}")

        except Exception as e:
            print(f"❌ Error extracting embeddings from {audio_path}: {e}")
            # Return zero embedding on error
            return np.zeros(1024, dtype=np.float32)

    def _split_signal(self, sig, rate):
        """Split signal into 3-second non-overlapping chunks"""
        chunk_samples = int(birdnet_cfg.SIG_LENGTH * rate)
        chunks = []

        for i in range(0, len(sig), chunk_samples):
            chunk = sig[i:i + chunk_samples]

            # Pad last chunk if too short but >= MIN_LEN
            if len(chunk) >= int(birdnet_cfg.SIG_MINLEN * rate):
                if len(chunk) < chunk_samples:
                    chunk = self._pad_signal(chunk, rate)
                chunks.append(chunk)

        return chunks

    def _pad_signal(self, sig, rate):
        """Pad signal to 3 seconds with zeros"""
        target_samples = int(birdnet_cfg.SIG_LENGTH * rate)
        if len(sig) < target_samples:
            padding = target_samples - len(sig)
            sig = np.pad(sig, (0, padding), mode='constant')
        return sig


class InsectEmbeddingDataset(Dataset):
    """Dataset that returns BirdNET embeddings instead of raw spectrograms"""

    def __init__(self, X_data, y_labels, extractor, aggregate='mean', cache_dir=None):
        """
        Args:
            X_data: numpy array of audio file paths or audio data
            y_labels: numpy array of integer labels
            extractor: BirdNETEmbeddingExtractor instance
            aggregate: How to aggregate multiple chunks per audio
            cache_dir: Optional directory to cache embeddings
        """
        self.X_data = X_data
        self.y_labels = y_labels
        self.extractor = extractor
        self.aggregate = aggregate
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        """Return (embedding, label) pair"""
        audio_data = self.X_data[idx]
        label = self.y_labels[idx]

        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"emb_{idx}.npy"
            if cache_file.exists():
                embedding = np.load(cache_file)
                return torch.from_numpy(embedding), torch.tensor(label, dtype=torch.long)

        # Extract embedding (audio_data could be path or numpy array)
        if isinstance(audio_data, str) or isinstance(audio_data, Path):
            embedding = self.extractor.extract_embeddings_from_audio(
                audio_data, aggregate=self.aggregate
            )
        else:
            # If X_data is already spectrograms, we need to convert or handle differently
            # For now, assume it's a file path
            embedding = np.zeros(1024, dtype=np.float32)

        # Cache for future use
        if self.cache_dir:
            np.save(cache_file, embedding)

        return torch.from_numpy(embedding), torch.tensor(label, dtype=torch.long)


def extract_embeddings_for_dataset(data_dir, output_dir, split='train', extractor=None):
    """
    Extract BirdNET embeddings for entire dataset.

    Args:
        data_dir: Directory containing .npy files (X_train.npy, y_train.npy, etc.)
        output_dir: Directory to save embeddings
        split: 'train' or 'val'
        extractor: BirdNETEmbeddingExtractor (will create if None)

    Returns:
        Paths to saved embedding files
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor if not provided
    if extractor is None:
        extractor = BirdNETEmbeddingExtractor()

    # Load data
    print(f"\n📂 Loading {split} data from {data_dir}")
    X_path = data_dir / f"X_{split}.npy"
    y_path = data_dir / f"y_{split}.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Data files not found in {data_dir}")

    X_data = np.load(X_path, mmap_mode='r')  # Memory-map for large files
    y_data = np.load(y_path)

    print(f"✅ Loaded {len(X_data)} {split} samples")
    print(f"   X shape: {X_data.shape}")
    print(f"   y shape: {y_data.shape}")

    # Extract embeddings
    print(f"\n🔍 Extracting BirdNET embeddings...")
    embeddings = []

    # Since X_data is spectrograms, we need to load original audio files
    # This requires modifying the data loading or storing audio paths
    # For now, we'll show a warning
    print("⚠️  Warning: X_data contains spectrograms, not audio paths")
    print("   You need to modify data loading to include audio file paths")
    print("   Or reconstruct audio from spectrograms (not recommended)")

    # Save placeholder for now
    # TODO: Update data pipeline to save audio paths alongside spectrograms
    embeddings_path = output_dir / f"X_{split}_embeddings.npy"
    labels_path = output_dir / f"y_{split}.npy"

    print(f"\n💡 Next steps:")
    print(f"1. Modify data loading to track audio file paths")
    print(f"2. Run embedding extraction on original audio files")
    print(f"3. Save embeddings to: {embeddings_path}")

    return embeddings_path, labels_path


def create_audio_path_dataset(data_splits_dir, dataset_name='combined'):
    """
    Create a dataset mapping that includes audio file paths.
    This is needed for embedding extraction since we need original audio.

    Args:
        data_splits_dir: Path to data/splits directory
        dataset_name: Name of dataset (e.g., 'combined')

    Returns:
        Dictionary with audio paths and labels
    """
    splits_dir = Path(data_splits_dir) / dataset_name

    # Load existing data
    X_train = np.load(splits_dir / "X_train.npy")
    y_train = np.load(splits_dir / "y_train.npy")
    X_val = np.load(splits_dir / "X_val.npy")
    y_val = np.load(splits_dir / "y_val.npy")

    print(f"📊 Loaded existing splits:")
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}")

    # TODO: Map spectrograms back to audio file paths
    # This requires storing metadata during preprocessing
    # For now, return structure

    dataset_info = {
        'train': {
            'spectrograms': X_train,
            'labels': y_train,
            'audio_paths': None  # TODO: Add audio path tracking
        },
        'val': {
            'spectrograms': X_val,
            'labels': y_val,
            'audio_paths': None  # TODO: Add audio path tracking
        }
    }

    return dataset_info


if __name__ == "__main__":
    # Demo usage
    print("🦗 BirdNET Transfer Learning for Insect Classification")
    print("=" * 80)

    # Initialize extractor
    extractor = BirdNETEmbeddingExtractor()

    # Example: Extract embeddings from a single audio file
    # test_audio = "data/raw/combined/InsectSound1000/Acheta_domesticus/sample.wav"
    # if Path(test_audio).exists():
    #     embedding = extractor.extract_embeddings_from_audio(test_audio)
    #     print(f"\n✅ Extracted embedding shape: {embedding.shape}")
    #     print(f"   Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")

    print("\n📋 Next Steps:")
    print("1. Modify preprocessing pipeline to save audio file paths")
    print("2. Extract embeddings for all training/validation audio")
    print("3. Train lightweight classifier on frozen embeddings")
    print("4. (Optional) Fine-tune BirdNET backbone layers")
