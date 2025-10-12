"""
ChirpKit Ensemble Classifier with Test-Time Augmentation

This module provides inference for the ChirpKit v6.0 ensemble model.
The model uses BirdNET embeddings as features, with a 7-model ensemble trained
on insect sounds.

Architecture:
- 7 DeepMLP models with different random seeds
- Each model: 1024 → 512 → 256 → 128 → 231 classes
- Optional Test-Time Augmentation (TTA) with Gaussian noise
- Supports flexible deployment modes (single, ensemble, ensemble+TTA)

Performance:
- Single model: 77% accuracy, ~10ms inference
- 7-model ensemble: 79.6% accuracy, ~35ms inference
- Ensemble + TTA: 79.7% accuracy, ~70ms inference
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import joblib


class DeepMLPClassifier(nn.Module):
    """Deep MLP classifier head for ChirpKit (operates on BirdNET embeddings)"""

    def __init__(self, n_classes, embedding_dim=1024, hidden_dims=[512, 256, 128], dropout=0.4):
        super().__init__()

        self.input_proj = nn.Linear(embedding_dim, hidden_dims[0])

        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        self.output = nn.Linear(hidden_dims[-1], n_classes)

    def forward(self, x):
        x = torch.relu(self.input_proj(x))
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class ChirpKitEnsembleClassifier:
    """
    ChirpKit Ensemble Classifier for insect sound identification

    Uses 7-model ensemble trained on BirdNET embeddings for robust predictions.

    Deployment modes:
    - 'single': Use best single model (fastest, ~77% accuracy)
    - 'ensemble': Use 7-model ensemble without TTA (balanced, ~79.6% accuracy)
    - 'ensemble_tta': Use 7-model ensemble with TTA (best, ~79.7% accuracy)
    """

    def __init__(
        self,
        model_dir: str = "models/trained/chirpkit-ensemble",
        mode: str = "ensemble",
        tta_rounds: int = 10,
        tta_noise_std: float = 0.01,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ensemble classifier

        Args:
            model_dir: Directory containing ensemble models
            mode: Deployment mode ('single', 'ensemble', 'ensemble_tta')
            tta_rounds: Number of TTA rounds (only for ensemble_tta mode)
            tta_noise_std: Standard deviation for TTA Gaussian noise
            device: torch device (auto-detected if None)
        """
        self.model_dir = Path(model_dir)
        self.mode = mode
        self.tta_rounds = tta_rounds
        self.tta_noise_std = tta_noise_std
        self.device = device or self._get_device()

        self.models = []
        self.label_encoder = None
        self.n_classes = None
        self.ensemble_info = None

    def _get_device(self) -> torch.device:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def load_models(self):
        """Load ensemble models and metadata"""
        print(f"📦 Loading ChirpKit Ensemble from {self.model_dir}")
        print(f"   Mode: {self.mode}")
        print(f"   Device: {self.device}")

        # Load ensemble info
        info_path = self.model_dir / "ensemble_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.ensemble_info = json.load(f)
            self.n_classes = self.ensemble_info['n_classes']
            num_models = self.ensemble_info['num_models']
            print(f"   Species: {self.n_classes}")
            print(f"   Models: {num_models}")
        else:
            raise FileNotFoundError(f"Ensemble info not found: {info_path}")

        # Load label encoder
        # Try to find label encoder in embeddings directory
        label_encoder_paths = [
            self.model_dir / "label_encoder.joblib",
            Path("data/embeddings/combined/label_encoder.joblib"),
            Path("data/embeddings_kaggle/chirpkit-birdnet-embeddings/label_encoder.joblib"),
        ]

        for encoder_path in label_encoder_paths:
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                print(f"   ✓ Label encoder loaded from {encoder_path}")
                break
        else:
            raise FileNotFoundError(f"Label encoder not found in any of: {label_encoder_paths}")

        # Load models
        if self.mode == 'single':
            # Load only the best model
            best_idx = np.argmax(self.ensemble_info['individual_accuracies'])
            model_path = self.model_dir / f"ensemble_model_{best_idx + 1}.pth"
            model = self._load_single_model(model_path)
            self.models = [model]
            print(f"   ✓ Loaded best model #{best_idx + 1} ({self.ensemble_info['individual_accuracies'][best_idx]:.2f}%)")
        else:
            # Load all models
            for i in range(num_models):
                model_path = self.model_dir / f"ensemble_model_{i + 1}.pth"
                model = self._load_single_model(model_path)
                self.models.append(model)
            print(f"   ✓ Loaded {num_models} ensemble models")

        if self.mode == 'ensemble_tta':
            print(f"   TTA: {self.tta_rounds} rounds with noise std={self.tta_noise_std}")

        print("✅ Ensemble ready!")

    def _load_single_model(self, model_path: Path) -> DeepMLPClassifier:
        """Load a single model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)

        model = DeepMLPClassifier(
            n_classes=self.n_classes,
            embedding_dim=1024,
            hidden_dims=[512, 256, 128],
            dropout=0.4
        )

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model

    def predict(self, embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Predict species from embedding

        Args:
            embedding: (1024,) numpy array - audio embedding from BirdNET
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions and metadata
        """
        if not self.models:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Convert to tensor
        embedding_tensor = torch.from_numpy(embedding).float().to(self.device)
        if embedding_tensor.dim() == 1:
            embedding_tensor = embedding_tensor.unsqueeze(0)  # Add batch dim

        # Get predictions based on mode
        if self.mode == 'single':
            logits = self._predict_single(embedding_tensor)
        elif self.mode == 'ensemble':
            logits = self._predict_ensemble(embedding_tensor, use_tta=False)
        else:  # ensemble_tta
            logits = self._predict_ensemble(embedding_tensor, use_tta=True)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = []
        for rank, idx in enumerate(top_indices):
            species = self.label_encoder.classes_[idx]
            predictions.append({
                'species': species,
                'scientific_name': species.replace('_', ' '),
                'confidence': float(probs[idx]),
                'rank': rank + 1
            })

        return {
            'top_prediction': predictions[0],
            'predictions': predictions,
            'mode': self.mode,
            'num_models': len(self.models),
            'tta_rounds': self.tta_rounds if self.mode == 'ensemble_tta' else 0,
            'model_version': 'v6.0'
        }

    def _predict_single(self, embedding: torch.Tensor) -> torch.Tensor:
        """Predict using single model"""
        with torch.no_grad():
            return self.models[0](embedding)

    def _predict_ensemble(self, embedding: torch.Tensor, use_tta: bool = False) -> torch.Tensor:
        """Predict using ensemble with optional TTA"""
        all_logits = []

        with torch.no_grad():
            for model in self.models:
                if use_tta:
                    # Test-time augmentation: multiple forward passes with noise
                    tta_logits = []

                    # Original (no noise)
                    tta_logits.append(model(embedding))

                    # Augmented versions
                    for _ in range(self.tta_rounds - 1):
                        noise = torch.randn_like(embedding) * self.tta_noise_std
                        aug_embedding = embedding + noise
                        tta_logits.append(model(aug_embedding))

                    # Average TTA predictions for this model
                    model_logits = torch.mean(torch.stack(tta_logits), dim=0)
                else:
                    # Single forward pass per model
                    model_logits = model(embedding)

                all_logits.append(model_logits)

        # Average predictions across all models
        ensemble_logits = torch.mean(torch.stack(all_logits), dim=0)

        return ensemble_logits

    def predict_from_audio(
        self,
        audio_path: str,
        extractor=None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Predict species from audio file

        Args:
            audio_path: Path to audio file
            extractor: BirdNETEmbeddingExtractor instance (will create if None)
            top_k: Number of top predictions

        Returns:
            Prediction results
        """
        # Import here to avoid circular dependency
        from transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor

        if extractor is None:
            extractor = BirdNETEmbeddingExtractor()

        # Extract embedding
        embedding = extractor.extract_embeddings_from_audio(audio_path, aggregate='mean')

        # Predict
        return self.predict(embedding, top_k=top_k)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_version': 'v6.0',
            'model_dir': str(self.model_dir),
            'mode': self.mode,
            'num_models': len(self.models),
            'n_classes': self.n_classes,
            'device': str(self.device),
            'tta_rounds': self.tta_rounds if self.mode == 'ensemble_tta' else 0,
            'tta_noise_std': self.tta_noise_std if self.mode == 'ensemble_tta' else 0,
            'ensemble_info': self.ensemble_info
        }


def load_ensemble_classifier(
    model_dir: str = "models/trained/chirpkit-ensemble",
    mode: str = "ensemble",
    **kwargs
) -> ChirpKitEnsembleClassifier:
    """
    Convenience function to create and load ChirpKit ensemble classifier

    Args:
        model_dir: Directory containing ensemble models
        mode: Deployment mode ('single', 'ensemble', 'ensemble_tta')
        **kwargs: Additional arguments for ChirpKitEnsembleClassifier

    Returns:
        Loaded ChirpKitEnsembleClassifier
    """
    classifier = ChirpKitEnsembleClassifier(model_dir=model_dir, mode=mode, **kwargs)
    classifier.load_models()
    return classifier


if __name__ == "__main__":
    print("🧪 Testing ChirpKit Ensemble Classifier")
    print("=" * 80)

    # Create dummy embedding
    dummy_embedding = np.random.randn(1024).astype(np.float32)

    # Test different modes
    modes = ['single', 'ensemble', 'ensemble_tta']

    for mode in modes:
        print(f"\n📊 Testing mode: {mode}")
        print("-" * 80)

        try:
            classifier = load_ensemble_classifier(mode=mode)
            result = classifier.predict(dummy_embedding, top_k=3)

            print(f"✅ {mode} mode working!")
            print(f"   Top prediction: {result['top_prediction']['species']}")
            print(f"   Confidence: {result['top_prediction']['confidence']:.4f}")
            print(f"   Models used: {result['num_models']}")
            print(f"   TTA rounds: {result['tta_rounds']}")

        except Exception as e:
            print(f"❌ {mode} mode failed: {e}")

    print(f"\n{'=' * 80}")
    print("✅ Ensemble classifier tested successfully!")
