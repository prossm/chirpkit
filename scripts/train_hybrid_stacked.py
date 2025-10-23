#!/usr/bin/env python3
"""
Hybrid Stacked Training: CNN-LSTM Feature Extractor + Bayesian Classifier
Ultra-fast Bayesian training using pre-trained CNN-LSTM features
"""
import torch
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from chirpkit.models.hybrid_stacked_model import HybridStackedModel, create_feature_dataset, CompactBayesianClassifier
from torch.utils.data import DataLoader, TensorDataset

# Import existing data loading utilities
sys.path.append(os.path.join(os.path.dirname(__file__)))
from train_unified import NpyDataset, UnifiedTrainer


class HybridStackedTrainer:
    """Ultra-fast trainer for hybrid stacked model"""

    def __init__(self, pretrained_cnn_lstm_path: str, dataset_name='combined', device='auto'):
        self.pretrained_cnn_lstm_path = pretrained_cnn_lstm_path
        self.dataset_name = dataset_name
        self.device = self._setup_device(device)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.label_encoder = None

        print(f"🔗 Hybrid Stacked Training")
        print(f"📊 Dataset: {dataset_name}")
        print(f"🧠 Base model: {pretrained_cnn_lstm_path}")
        print(f"💻 Device: {self.device}")

    def _setup_device(self, device):
        if device == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(device)

    def load_data_and_extract_features(self):
        """Load data and extract features using pre-trained CNN-LSTM"""
        print(f"\n🔄 Loading {self.dataset_name} dataset...")

        # Use existing data loading logic
        unified_trainer = UnifiedTrainer(dataset_name=self.dataset_name)
        train_dataset, val_dataset = unified_trainer.load_data()

        self.label_encoder = unified_trainer.label_encoder
        n_classes = len(self.label_encoder.classes_)

        print(f"✅ Loaded data: {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"🦗 Classes: {n_classes} species")

        # Create data loaders for feature extraction
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

        # Extract features using pre-trained CNN-LSTM
        print(f"\n🔍 Extracting features from audio using pre-trained CNN-LSTM...")

        train_features, train_labels = create_feature_dataset(
            self.pretrained_cnn_lstm_path,
            train_loader,
            device=self.device,
            save_path=f'data/features/{self.dataset_name}_train_features.pt'
        )

        val_features, val_labels = create_feature_dataset(
            self.pretrained_cnn_lstm_path,
            val_loader,
            device=self.device,
            save_path=f'data/features/{self.dataset_name}_val_features.pt'
        )

        # Create feature datasets
        train_feature_dataset = TensorDataset(train_features, train_labels)
        val_feature_dataset = TensorDataset(val_features, val_labels)

        print(f"✅ Feature extraction complete!")
        print(f"   Training features: {train_features.shape}")
        print(f"   Validation features: {val_features.shape}")

        return train_feature_dataset, val_feature_dataset, n_classes, train_features.shape[1]

    def create_hybrid_model(self, n_classes, feature_dim):
        """Create hybrid stacked model"""
        print(f"\n🔗 Creating hybrid stacked model...")

        self.model = HybridStackedModel(
            pretrained_cnn_lstm_path=self.pretrained_cnn_lstm_path,
            n_classes=n_classes,
            feature_dim=feature_dim
        )

        # For training, we only need the Bayesian classifier part
        # since features are pre-extracted
        self.bayesian_classifier = CompactBayesianClassifier(
            feature_dim=feature_dim,
            n_classes=n_classes,
            dropout=0.3
        )

        self.bayesian_classifier.to(self.device)

        print(f"✅ Hybrid model created:")
        print(f"   Feature dim: {feature_dim}")
        print(f"   Classes: {n_classes}")
        print(f"   Model parameters: {sum(p.numel() for p in self.bayesian_classifier.parameters()):,}")

        return self.bayesian_classifier

    def setup_training(self, feature_dataset, val_dataset, batch_size=128, lr=1e-3, weight_decay=1e-4):
        """Setup training for ultra-fast feature-based learning"""

        # Create data loaders for features (can use larger batch sizes!)
        self.train_loader = DataLoader(
            feature_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Optimizer for compact Bayesian classifier
        self.optimizer = torch.optim.AdamW(
            self.bayesian_classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )

        print(f"🚀 Training setup complete:")
        print(f"   Batch size: {batch_size} (larger batches for speed!)")
        print(f"   Learning rate: {lr}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")

    def train_epoch(self):
        """Ultra-fast training epoch on pre-extracted features"""
        self.bayesian_classifier.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for features, labels in progress_bar:
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass (no MC sampling during training for speed)
            logits = self.bayesian_classifier(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * total_correct / total_samples:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def validate_epoch(self):
        """Ultra-fast validation with uncertainty quantification"""
        self.bayesian_classifier.eval()

        total_loss = 0
        all_predictions = []
        all_labels = []
        all_uncertainties = []

        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Bayesian inference with uncertainty
                uncertainty_data = self.bayesian_classifier.forward_with_uncertainty(features, n_samples=20)

                predictions = uncertainty_data['predictions']
                loss = torch.nn.functional.cross_entropy(predictions, labels)
                total_loss += loss.item()

                # Collect for metrics
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_uncertainties.append(uncertainty_data['total_uncertainty'].cpu())

        # Compute metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_uncertainties = torch.cat(all_uncertainties)

        predicted_classes = torch.argmax(all_predictions, dim=1)
        accuracy = (predicted_classes == all_labels).float().mean().item()
        avg_loss = total_loss / len(self.val_loader)
        avg_uncertainty = all_uncertainties.mean().item()

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'avg_uncertainty': avg_uncertainty
        }

    def train(self, epochs=200, patience=20, save_dir="models/hybrid_stacked"):
        """Ultra-fast hybrid training"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_acc = 0
        patience_counter = 0

        print(f"\n🚀 Starting Hybrid Stacked Training")
        print(f"🎯 Goal: Ultra-fast Bayesian learning on CNN-LSTM features")
        print("=" * 80)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train epoch (ultra-fast on features!)
            train_loss, train_acc = self.train_epoch()

            # Validate with uncertainty
            val_metrics = self.validate_epoch()

            # Learning rate scheduling
            self.scheduler.step(val_metrics['accuracy'])

            epoch_time = time.time() - epoch_start

            # Display results
            print(f"\n" + "="*80)
            print(f"⚡ Epoch {epoch}/{epochs} ({epoch_time:.1f}s) - ULTRA FAST!")
            print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"🎯 Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"🧠 Avg Uncertainty: {val_metrics['avg_uncertainty']:.3f}")

            # Model saving
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0

                # Save best model
                model_path = save_dir / "best_hybrid_model.pth"
                torch.save(self.bayesian_classifier.state_dict(), model_path)

                print(f"\n🏆✨ NEW BEST HYBRID MODEL! ✨🏆")
                print(f"🎯 Accuracy: {best_val_acc:.4f}")
                print(f"💾 Saved to: {model_path}")
                print("🎉" + "="*50 + "🎉")
            else:
                patience_counter += 1
                print(f"⏱️  Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\n🛑 Early stopping: No improvement for {patience} epochs")
                break

        print(f"\n✅ Hybrid Training Complete!")
        print(f"🏆 Best Accuracy: {best_val_acc:.4f}")
        print(f"⚡ Ultra-fast training with pre-extracted features!")

        return best_val_acc

    def calibrate_model(self):
        """Temperature scaling calibration"""
        print(f"\n🌡️ Calibrating hybrid model...")
        temperature = self.bayesian_classifier.calibrate_temperature(self.val_loader, self.device)
        print(f"✅ Calibration complete. Temperature: {temperature:.3f}")
        return temperature

    def save_complete_model(self, save_path, model_name=None):
        """Save complete hybrid model for inference"""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_name = f"hybrid_stacked_{len(self.label_encoder.classes_)}species_{timestamp}"

        # Create complete hybrid model for inference
        complete_model = HybridStackedModel(
            self.pretrained_cnn_lstm_path,
            len(self.label_encoder.classes_),
            feature_dim=self.bayesian_classifier.feature_dim
        )

        # Copy trained Bayesian classifier weights
        complete_model.bayesian_classifier.load_state_dict(self.bayesian_classifier.state_dict())

        # Save complete model
        complete_model.save_model(save_path)

        # Save label encoder
        import joblib
        encoder_path = Path(save_path).parent / f"{model_name}_label_encoder.joblib"
        joblib.dump(self.label_encoder, encoder_path)

        print(f"💾 Complete hybrid model saved:")
        print(f"   Model: {save_path}")
        print(f"   Encoder: {encoder_path}")

        return save_path


def main():
    parser = argparse.ArgumentParser(description='Hybrid Stacked Training: CNN-LSTM + Bayesian')

    # Dataset arguments
    parser.add_argument('--dataset', choices=['insectsound1000', 'insectset459', 'sina', 'xenocanto', 'combined'],
                       default='combined', help='Dataset to train on')

    # Model arguments
    parser.add_argument('--pretrained-cnn-lstm', type=str,
                       default='models/trained/insect_classifier_609species.pth',
                       help='Path to pre-trained CNN-LSTM model')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (can be large for features)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')

    # Output arguments
    parser.add_argument('--save-dir', type=str, default='models/hybrid_stacked',
                       help='Directory to save models')
    parser.add_argument('--no-calibration', action='store_true',
                       help='Skip temperature scaling calibration')

    args = parser.parse_args()

    print("⚡ ChirpKit Hybrid Stacked Training")
    print("🔗 CNN-LSTM Feature Extraction + Ultra-Fast Bayesian Learning")
    print("=" * 80)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🧠 Base CNN-LSTM: {args.pretrained_cnn_lstm}")
    print(f"⚡ Ultra-fast feature-based training!")
    print("=" * 80)

    # Initialize trainer
    trainer = HybridStackedTrainer(
        pretrained_cnn_lstm_path=args.pretrained_cnn_lstm,
        dataset_name=args.dataset
    )

    # Load data and extract features
    train_dataset, val_dataset, n_classes, feature_dim = trainer.load_data_and_extract_features()

    # Create hybrid model
    model = trainer.create_hybrid_model(n_classes, feature_dim)

    # Setup training
    trainer.setup_training(train_dataset, val_dataset, args.batch_size, args.lr, args.weight_decay)

    # Train model (ultra-fast!)
    best_accuracy = trainer.train(args.epochs, args.patience, args.save_dir)

    # Calibrate model
    if not args.no_calibration:
        trainer.calibrate_model()

    # Save complete model
    save_path = Path(args.save_dir) / "complete_hybrid_model.pth"
    trainer.save_complete_model(save_path)

    print(f"\n🎉 Hybrid Stacked Training Complete!")
    print(f"🏆 Final Accuracy: {best_accuracy:.4f}")
    print(f"⚡ Ultra-fast training achieved with feature pre-extraction!")
    print(f"🔗 Best of both worlds: CNN-LSTM features + Bayesian uncertainty!")


if __name__ == "__main__":
    main()