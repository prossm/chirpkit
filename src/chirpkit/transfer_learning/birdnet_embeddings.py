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
import librosa
import tensorflow as tf

# Try to import from BirdNET-Analyzer if available (for training)
# Otherwise use standalone TFLite implementation (for inference)
USING_BIRDNET_ANALYZER = False

# Try local BirdNET-Analyzer directory first (development/GitHub install)
try:
    BIRDNET_PATH = Path(__file__).parent.parent.parent.parent / "BirdNET-Analyzer"
    if BIRDNET_PATH.exists():
        sys.path.insert(0, str(BIRDNET_PATH))
        import birdnet_analyzer.config as birdnet_cfg
        from birdnet_analyzer import model as birdnet_model
        from birdnet_analyzer import audio as birdnet_audio
        USING_BIRDNET_ANALYZER = True
        print("✅ Using BirdNET-Analyzer (GitHub version with model files)")
except ImportError:
    pass

if not USING_BIRDNET_ANALYZER:
    print("ℹ️  Using standalone TFLite implementation (inference only)")


class BirdNETEmbeddingExtractor:
    """Extract 1024-dimensional embeddings from audio using BirdNET"""

    # BirdNET constants
    SAMPLE_RATE = 48000  # BirdNET uses 48kHz
    SIG_LENGTH = 3.0     # 3-second chunks
    SIG_MINLEN = 1.0     # Minimum 1 second
    EMBEDDING_DIM = 1024 # Output embedding size
    BANDPASS_FMIN = 1000   # 1kHz (insect frequencies)
    BANDPASS_FMAX = 15000  # 15kHz

    def __init__(self, model_path=None):
        """Initialize BirdNET model for embedding extraction"""
        print("🔧 Initializing BirdNET embedding extractor...")

        self.using_analyzer = USING_BIRDNET_ANALYZER

        if USING_BIRDNET_ANALYZER:
            # Use full BirdNET-Analyzer (for training)
            self._init_with_analyzer()
        else:
            # Use standalone TFLite (for inference)
            self._init_standalone(model_path)

        print("✅ BirdNET embedding extractor ready!")
        print(f"   Sample rate: {self.SAMPLE_RATE}Hz")
        print(f"   Chunk length: {self.SIG_LENGTH}s")
        print(f"   Embedding dim: {self.EMBEDDING_DIM}")
        print(f"   Bandpass filter: {self.BANDPASS_FMIN}-{self.BANDPASS_FMAX}Hz")

    def _init_with_analyzer(self):
        """Initialize using full BirdNET-Analyzer"""
        # PyPI package already has MODEL_PATH and LABELS_FILE set to correct defaults
        # Only override if BIRDNET_MODEL_PATH exists (GitHub version)
        if hasattr(birdnet_cfg, 'BIRDNET_MODEL_PATH'):
            birdnet_cfg.MODEL_PATH = birdnet_cfg.BIRDNET_MODEL_PATH
        if hasattr(birdnet_cfg, 'BIRDNET_LABELS_FILE'):
            birdnet_cfg.LABELS_FILE = birdnet_cfg.BIRDNET_LABELS_FILE

        birdnet_cfg.SAMPLE_RATE = self.SAMPLE_RATE
        birdnet_cfg.SIG_LENGTH = self.SIG_LENGTH
        birdnet_cfg.SIG_OVERLAP = 0.0
        birdnet_cfg.SIG_MINLEN = self.SIG_MINLEN
        birdnet_cfg.BATCH_SIZE = 32
        birdnet_cfg.BANDPASS_FMIN = self.BANDPASS_FMIN
        birdnet_cfg.BANDPASS_FMAX = self.BANDPASS_FMAX

        print(f"📦 Loading BirdNET model from: {birdnet_cfg.MODEL_PATH}")
        birdnet_model.load_model(class_output=False)  # False = get embeddings

    def _init_standalone(self, model_path=None):
        """Initialize using standalone TFLite"""
        if model_path is None:
            # Default to models/birdnet/ directory (go up to project root)
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "birdnet" / "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"

        self.model_path = Path(model_path)

        # Auto-download if not present
        if not self.model_path.exists():
            print(f"⚠️  BirdNET model not found at {self.model_path}")
            print(f"   Attempting auto-download...")

            try:
                # Import here to avoid circular dependency
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from chirpkit.model_downloader import ModelDownloader
                birdnet_dir = ModelDownloader.download_model('birdnet')
                self.model_path = birdnet_dir / "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
            except Exception as e:
                print(f"❌ Auto-download failed: {e}")
                print(f"\n💡 Please install models manually:")
                print(f"   git clone https://github.com/prossm/chirpkit.git")
                print(f"   cd chirpkit && git lfs pull")
                raise FileNotFoundError(
                    f"BirdNET model not found at {self.model_path}\n"
                    f"Expected location: models/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
                )

        print(f"📦 Loading TFLite model from: {self.model_path}")

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Debug: print output shapes to find embeddings
        print(f"📊 TFLite model outputs:")
        for i, output in enumerate(self.output_details):
            print(f"   Output {i}: shape={output['shape']}, dtype={output['dtype']}")

        # Find embedding output (should be 1024-dim)
        self.embedding_output_idx = None
        for i, output in enumerate(self.output_details):
            if 1024 in output['shape']:
                self.embedding_output_idx = i
                print(f"✅ Found embedding output at index {i}")
                break

        if self.embedding_output_idx is None:
            print(f"⚠️  Could not find 1024-dim embedding output, using output 0")

    def extract_embeddings_from_audio(self, audio_path, aggregate='mean'):
        """
        Extract embeddings from an audio file.

        Args:
            audio_path: Path to audio file
            aggregate: How to combine multiple 3-second chunks
                      'mean' - average all embeddings
                      'max' - max pool across embeddings
                      'first' - use only first chunk
                      'all' - return all embeddings (for data augmentation)

        Returns:
            embeddings: numpy array of shape (1024,) or (n_chunks, 1024)
        """
        try:
            if self.using_analyzer:
                return self._extract_with_analyzer(audio_path, aggregate)
            else:
                return self._extract_standalone(audio_path, aggregate)

        except Exception as e:
            print(f"❌ Error extracting embeddings from {audio_path}: {e}")
            # Return zero embedding on error
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

    def _extract_with_analyzer(self, audio_path, aggregate):
        """Extract embeddings using full BirdNET-Analyzer"""
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
        return self._aggregate_embeddings(embeddings, aggregate)

    def _extract_standalone(self, audio_path, aggregate):
        """Extract embeddings using standalone TFLite"""
        # Load audio with librosa
        audio_data, sr = librosa.load(
            audio_path,
            sr=self.SAMPLE_RATE,
            mono=True
        )

        # Split into 3-second chunks
        chunks = self._split_signal(audio_data, sr)

        if len(chunks) == 0:
            # Audio too short, pad to 3 seconds
            chunks = [self._pad_signal(audio_data, sr)]

        # Extract embeddings for all chunks
        embeddings = []
        for chunk in chunks:
            embedding = self._extract_chunk_embedding_tflite(chunk)
            embeddings.append(embedding)

        embeddings = np.array(embeddings)  # Shape: (n_chunks, 1024)

        # Aggregate embeddings
        return self._aggregate_embeddings(embeddings, aggregate)

    def _extract_chunk_embedding_tflite(self, audio_chunk):
        """Extract embedding from a single chunk using TFLite"""
        # Ensure correct length
        expected_samples = int(self.SIG_LENGTH * self.SAMPLE_RATE)
        if len(audio_chunk) != expected_samples:
            audio_chunk = self._pad_signal(audio_chunk, self.SAMPLE_RATE)

        # Prepare input for TFLite model
        input_data = audio_chunk.astype(np.float32).reshape(1, -1)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get embedding output (use detected embedding index or fallback to 0)
        output_idx = self.embedding_output_idx if self.embedding_output_idx is not None else 0
        embedding = self.interpreter.get_tensor(self.output_details[output_idx]['index'])

        return embedding.flatten()

    def _aggregate_embeddings(self, embeddings, aggregate):
        """Aggregate multiple embeddings into one"""
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

    def _split_signal(self, sig, rate):
        """Split signal into 3-second non-overlapping chunks"""
        chunk_samples = int(self.SIG_LENGTH * rate)
        min_samples = int(self.SIG_MINLEN * rate)
        chunks = []

        for i in range(0, len(sig), chunk_samples):
            chunk = sig[i:i + chunk_samples]

            # Pad last chunk if too short but >= MIN_LEN
            if len(chunk) >= min_samples:
                if len(chunk) < chunk_samples:
                    chunk = self._pad_signal(chunk, rate)
                chunks.append(chunk)

        return chunks

    def _pad_signal(self, sig, rate):
        """Pad signal to 3 seconds with zeros"""
        target_samples = int(self.SIG_LENGTH * rate)
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
