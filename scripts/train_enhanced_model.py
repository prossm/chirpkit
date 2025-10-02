#!/usr/bin/env python3
"""
Advanced training script for 85%+ accuracy target
Combines enhanced architecture, ensemble methods, and sophisticated training strategies
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import joblib
from torch.utils.tensorboard import SummaryWriter

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from models.enhanced_cnn_lstm import EnhancedCNNLSTMClassifier, EnhancedLoss
from models.ensemble_model import InsectEnsemble, AdaptiveEnsemble
from data.augmentation import InsectAudioAugmenter, AugmentedDataset

# Import existing utilities
sys.path.append(os.path.join(os.path.dirname(__file__)))
from train_unified import NpyDataset, UnifiedTrainer

class AdvancedTrainer:
    """Advanced trainer targeting 85%+ accuracy"""

    def __init__(self, model_type='enhanced', dataset_name='combined', device='auto'):
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.device = self._setup_device(device)
        self.splits_dir = Path(f'data/splits/{dataset_name}')

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.label_encoder = None

        # Advanced training parameters
        self.gradient_accumulation_steps = 4
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0
        self.use_swa = True  # Stochastic Weight Averaging

    def _setup_device(self, device):
        """Setup device with preference order"""
        if device == 'auto':
            # Force CPU for stability - MPS can cause bus errors with complex models
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def load_data(self):
        """Load data with enhanced augmentation"""
        print(f"🔄 Loading {self.dataset_name} dataset for advanced training...")

        # Use existing data loading with progress tracking
        print("📂 Initializing data loader...")
        unified_trainer = UnifiedTrainer(dataset_name=self.dataset_name)

        print("📊 Loading dataset splits (this may take a moment for large datasets)...")
        train_dataset_base, val_dataset = unified_trainer.load_data()

        self.label_encoder = unified_trainer.label_encoder
        n_classes = len(self.label_encoder.classes_)

        # Enhanced augmentation for training
        print("🎭 Setting up advanced audio augmentation...")
        augmenter = InsectAudioAugmenter(sr=16000)
        train_dataset = AugmentedDataset(
            train_dataset_base,
            augmenter,
            augmentation_prob=0.7  # Higher augmentation for better generalization
        )

        print(f"✅ Loaded data: {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"🦗 Classes: {n_classes} species")

        return train_dataset, val_dataset, n_classes

    def create_model(self, n_classes, model_configs=None):
        """Create enhanced model based on type"""
        print(f"🧠 Creating {self.model_type} model for {n_classes} classes...")

        if self.model_type == 'enhanced':
            self.model = EnhancedCNNLSTMClassifier(
                n_classes=n_classes,
                lstm_hidden=512,  # Larger for more capacity
                dropout=0.3
            )
            self.criterion = EnhancedLoss(n_classes, alpha=0.3, label_smoothing=0.1)

        elif self.model_type == 'ensemble':
            if model_configs is None:
                # Default ensemble configuration
                model_configs = [
                    {
                        'type': 'enhanced_cnn_lstm',
                        'n_classes': n_classes,
                        'lstm_hidden': 512,
                        'dropout': 0.3,
                        'feature_dim': 512,
                        'weights_path': 'models/enhanced_advanced/enhanced_best.pth'
                    },
                    {
                        'type': 'bayesian_cnn_lstm',
                        'n_classes': n_classes,
                        'dropout': 0.3,
                        'feature_dim': 256,
                        'weights_path': 'models/bayesian_advanced/bayesian_insect_classifier_609species_best.bayesian'
                    },
                    {
                        'type': 'simple_cnn_lstm',
                        'n_classes': n_classes,
                        'feature_dim': 256,
                        'weights_path': 'models/trained/cnn_lstm_best.pth'
                    }
                ]

            self.model = InsectEnsemble(model_configs, ensemble_method='weighted_average')
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        elif self.model_type == 'adaptive_ensemble':
            if model_configs is None:
                model_configs = [
                    {'type': 'enhanced_cnn_lstm', 'n_classes': n_classes, 'lstm_hidden': 512},
                    {'type': 'bayesian_cnn_lstm', 'n_classes': n_classes},
                    {'type': 'simple_cnn_lstm', 'n_classes': n_classes}
                ]

            self.model = AdaptiveEnsemble(model_configs, n_classes)
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = self.model.to(self.device)
        print(f"✅ Model created and moved to {self.device}")

        return self.model

    def setup_training(self, train_dataset, val_dataset, config):
        """Setup optimizers and schedulers"""
        print("🔧 Setting up data loaders...")
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        print("⚙️ Configuring optimizers and schedulers...")

        # Optimizer with different learning rates for different parts
        if self.model_type == 'enhanced':
            # Different learning rates for CNN and LSTM parts
            cnn_params = []
            lstm_params = []
            classifier_params = []

            for name, param in self.model.named_parameters():
                if 'feature_extractor' in name or 'cnn_layers' in name:
                    cnn_params.append(param)
                elif 'lstm' in name or 'attention' in name:
                    lstm_params.append(param)
                else:
                    classifier_params.append(param)

            self.optimizer = optim.AdamW([
                {'params': cnn_params, 'lr': config['lr'] * 0.1},  # Lower LR for CNN
                {'params': lstm_params, 'lr': config['lr']},       # Standard LR for LSTM
                {'params': classifier_params, 'lr': config['lr'] * 2}  # Higher LR for classifier
            ], weight_decay=config['weight_decay'])

        else:
            # Standard optimizer for ensemble models
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )

        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=config['lr'] * 0.01
        )

        # Stochastic Weight Averaging (optional)
        if self.use_swa:
            self.swa_model = optim.swa_utils.AveragedModel(self.model)
            self.swa_scheduler = optim.swa_utils.SWALR(self.optimizer, swa_lr=config['lr'] * 0.1)

        print("✅ Training setup complete")

    def mixup_data(self, x, y, alpha=1.0):
        """Mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def train_epoch(self, epoch):
        """Train one epoch with advanced techniques"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if batch_idx == 0 and epoch == 1:
                print(f"🔄 Processing first batch (shape: {data.shape}) - this may take several minutes on CPU...")
            elif batch_idx % 50 == 0:
                print(f"📊 Batch {batch_idx}/{len(self.train_loader)}")
            data, target = data.to(self.device), target.to(self.device)

            # Apply mixup with probability
            if np.random.random() < 0.5 and self.model_type == 'enhanced':
                data, target_a, target_b, lam = self.mixup_data(data, target, self.mixup_alpha)
                mixup = True
            else:
                mixup = False

            # Forward pass
            if self.model_type == 'enhanced':
                outputs = self.model(data)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    outputs, aux_outputs = outputs
                else:
                    aux_outputs = None
            else:
                outputs = self.model(data)
                aux_outputs = None

            # Calculate loss
            if mixup and self.model_type == 'enhanced':
                if aux_outputs is not None:
                    loss_a, _, _ = self.criterion(outputs, aux_outputs, target_a)
                    loss_b, _, _ = self.criterion(outputs, aux_outputs, target_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = lam * self.criterion(outputs, target_a) + (1 - lam) * self.criterion(outputs, target_b)
            else:
                if aux_outputs is not None and isinstance(self.criterion, EnhancedLoss):
                    loss, main_loss, aux_loss = self.criterion(outputs, aux_outputs, target)
                else:
                    loss = self.criterion(outputs, target)

            # Backward pass with gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps

            # Calculate accuracy
            if not mixup:
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(self.train_loader)

        return avg_loss, accuracy

    def validate(self):
        """Validation with test-time augmentation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Test-time augmentation (optional)
                outputs = self.model(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                if isinstance(self.criterion, EnhancedLoss):
                    loss = self.criterion(outputs, outputs, target)[0]
                else:
                    loss = self.criterion(outputs, target)

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        accuracy = correct / total
        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, accuracy, all_predictions, all_targets

    def train(self, config):
        """Main training loop"""
        writer = SummaryWriter(f"runs/{self.model_type}_{self.dataset_name}_advanced")

        best_acc = 0
        patience_counter = 0
        start_epoch = 1

        # Load checkpoint if exists
        checkpoint_path = f"models/checkpoints/{self.model_type}_latest.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            print(f"🔄 Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}")

        print(f"\n🚀 Starting Advanced Training - Target: 85%+ Accuracy")
        print(f"🎯 Model: {self.model_type}")
        print(f"📊 Dataset: {self.dataset_name}")
        print("=" * 80)

        for epoch in range(start_epoch, config['epochs'] + 1):
            # Training
            if epoch == start_epoch:
                print("🏁 Starting first epoch (model architecture will be finalized)...")
            train_loss, train_acc = self.train_epoch(epoch)

            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate()

            # Learning rate scheduling
            if hasattr(self, 'swa_scheduler') and epoch > config['swa_start']:
                self.swa_scheduler.step()
                self.swa_model.update_parameters(self.model)
            else:
                self.scheduler.step()

            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)

            print(f"\nEpoch {epoch}/{config['epochs']}")
            print(f"📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"📚 Learning Rate: {current_lr:.2e}")

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0

                # Save model
                save_dir = Path(f"models/{self.model_type}_advanced")
                save_dir.mkdir(parents=True, exist_ok=True)

                model_path = save_dir / f"{self.model_type}_best.pth"
                torch.save(self.model.state_dict(), model_path)

                # Save ensemble configs if applicable
                if hasattr(self.model, 'model_weights'):
                    weights_path = save_dir / "ensemble_weights.pt"
                    torch.save(self.model.model_weights.data, weights_path)

                print(f"🎉 New best model! Accuracy: {val_acc:.4f}")

                if val_acc >= 0.85:
                    print("🎯 TARGET ACHIEVED! 85%+ accuracy reached!")

            else:
                patience_counter += 1

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
                'best_acc': best_acc,
                'patience_counter': patience_counter
            }
            torch.save(checkpoint, checkpoint_path)

            print(f"🏆 Best accuracy so far: {best_acc:.4f}")
            print(f"⏳ Patience: {patience_counter}/{config['patience']}")

            # Early stopping
            if patience_counter >= config['patience']:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break

        # Final SWA if used
        if hasattr(self, 'swa_model') and epoch > config['swa_start']:
            print("🔄 Finalizing Stochastic Weight Averaging...")
            try:
                optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)

                # Save SWA model
                save_dir = Path(f"models/{self.model_type}_advanced")
                swa_path = save_dir / f"{self.model_type}_swa.pth"
                torch.save(self.swa_model.state_dict(), swa_path)
                print(f"✅ SWA model saved to {swa_path}")
            except Exception as e:
                print(f"⚠️ SWA finalization failed: {e}")

        writer.close()
        print(f"\n🎉 Training complete! Best accuracy: {best_acc:.4f}")
        return best_acc

def main():
    parser = argparse.ArgumentParser(description='Advanced Training for 85%+ Accuracy')

    parser.add_argument('--model-type', choices=['enhanced', 'ensemble', 'adaptive_ensemble'],
                       default='enhanced', help='Model architecture to use')
    parser.add_argument('--dataset', choices=['insectsound1000', 'insectset459', 'sina', 'xenocanto', 'combined'],
                       default='combined', help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=500, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (smaller for larger models)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--swa-start', type=int, default=150, help='Start SWA at this epoch')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')

    args = parser.parse_args()

    print("🎯 ChirpKit Advanced Training - Target: 85%+ Accuracy")
    print("=" * 80)

    # Initialize trainer
    print("🔧 Initializing advanced trainer...")
    trainer = AdvancedTrainer(
        model_type=args.model_type,
        dataset_name=args.dataset,
        device=args.device
    )

    # Load data (this is often the longest step)
    print("📚 Loading and preparing datasets (this may take 30-60 seconds for large datasets)...")
    train_dataset, val_dataset, n_classes = trainer.load_data()

    # Create model
    print("🏗️ Building model architecture...")
    model = trainer.create_model(n_classes)

    # Training configuration
    config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'patience': args.patience,
        'swa_start': args.swa_start
    }

    # Setup training
    print("📋 Finalizing training configuration...")
    trainer.setup_training(train_dataset, val_dataset, config)

    # Train
    print("🎓 Starting training process...")
    final_accuracy = trainer.train(config)

    if final_accuracy >= 0.85:
        print("\n🎉 SUCCESS! 85%+ accuracy target achieved!")
    else:
        print(f"\n📈 Progress made: {final_accuracy:.4f} accuracy")
        print("💡 Consider ensemble methods or architectural improvements")

if __name__ == "__main__":
    main()