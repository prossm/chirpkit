#!/usr/bin/env python3
"""
HEAVILY REGULARIZED training for limited data (40 samples/species)
Based on train_enhanced_model.py but optimized for preventing overfitting
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

from models.enhanced_cnn_lstm_regularized import RegularizedEnhancedCNNLSTMClassifier, EnhancedLoss
from models.ensemble_model import InsectEnsemble, AdaptiveEnsemble
from data.augmentation import InsectAudioAugmenter, AugmentedDataset

# Import existing utilities
sys.path.append(os.path.join(os.path.dirname(__file__)))
from train_unified import NpyDataset, UnifiedTrainer

class StronglyRegularizedTrainer:
    """Trainer with AGGRESSIVE regularization for limited data"""

    def __init__(self, dataset_name='combined', device='auto'):
        self.dataset_name = dataset_name
        self.device = self._setup_device(device)
        self.splits_dir = Path(f'data/splits/{dataset_name}')

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.label_encoder = None

        # AGGRESSIVE regularization parameters
        self.gradient_accumulation_steps = 4
        self.mixup_alpha = 0.3  # Increased from 0.2
        self.use_swa = True  # Stochastic Weight Averaging
        self.dropout = 0.5  # STRONG dropout

    def _setup_device(self, device):
        """Setup device with preference order"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def load_data(self):
        """Load data with VERY AGGRESSIVE augmentation"""
        print(f"🔄 Loading {self.dataset_name} dataset for strongly regularized training...")

        # Use existing data loading
        print("📂 Initializing data loader...")
        unified_trainer = UnifiedTrainer(dataset_name=self.dataset_name)

        print("📊 Loading dataset splits...")
        train_dataset_base, val_dataset = unified_trainer.load_data()

        self.label_encoder = unified_trainer.label_encoder
        n_classes = len(self.label_encoder.classes_)

        # VERY AGGRESSIVE augmentation
        print("🎭 Setting up AGGRESSIVE audio augmentation (0.8 probability)...")
        augmenter = InsectAudioAugmenter(sr=16000)
        train_dataset = AugmentedDataset(
            train_dataset_base,
            augmenter,
            augmentation_prob=0.8  # Increased from 0.7 - augment 80% of samples!
        )

        print(f"✅ Loaded data: {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"🦗 Classes: {n_classes} species")
        print(f"📊 Samples per species: ~{len(train_dataset)/n_classes:.1f}")

        if len(train_dataset)/n_classes < 50:
            print("⚠️  WARNING: Very limited data - aggressive regularization enabled!")

        return train_dataset, val_dataset, n_classes

    def create_model(self, n_classes):
        """Create heavily regularized enhanced model"""
        print(f"🧠 Creating STRONGLY REGULARIZED model for {n_classes} classes...")
        print(f"   Dropout rate: {self.dropout} (50% - very aggressive!)")
        print(f"   LSTM layers: 3 (enables internal dropout)")
        print(f"   Label smoothing: 0.15 (strong)")

        self.model = RegularizedEnhancedCNNLSTMClassifier(
            n_classes=n_classes,
            lstm_hidden=256,  # Reduced from 512 - smaller capacity
            dropout=self.dropout
        )

        # Enhanced loss with STRONGER label smoothing
        self.criterion = EnhancedLoss(
            n_classes,
            alpha=0.4,  # Stronger auxiliary loss
            label_smoothing=0.15  # Stronger label smoothing
        )

        self.model = self.model.to(self.device)
        print(f"✅ Model created and moved to {self.device}")
        print(f"🎯 Target: Reduce overfitting gap from 50% to <20%")

        return self.model

    def setup_training(self, train_dataset, val_dataset, config):
        """Setup optimizers and schedulers with HIGHER weight decay"""
        print("🔧 Setting up data loaders...")
        # Smaller batch size for better generalization
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

        print("⚙️ Configuring optimizer with STRONG weight decay...")

        # Different learning rates for different parts
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

        # INCREASED weight decay for stronger regularization
        self.optimizer = optim.AdamW([
            {'params': cnn_params, 'lr': config['lr'] * 0.1, 'weight_decay': config['weight_decay'] * 2},  # Extra WD for CNN
            {'params': lstm_params, 'lr': config['lr'], 'weight_decay': config['weight_decay']},
            {'params': classifier_params, 'lr': config['lr'] * 2, 'weight_decay': config['weight_decay'] * 1.5}  # Extra WD for classifier
        ])

        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=15,  # Shorter restart cycle (was 20)
            T_mult=2,
            eta_min=config['lr'] * 0.01
        )

        # Stochastic Weight Averaging
        if self.use_swa:
            self.swa_model = optim.swa_utils.AveragedModel(self.model)
            self.swa_scheduler = optim.swa_utils.SWALR(self.optimizer, swa_lr=config['lr'] * 0.1)

        print("✅ Training setup complete")
        print(f"   Weight decay: {config['weight_decay']} (STRONG)")
        print(f"   Mixup alpha: {self.mixup_alpha}")
        print(f"   Gradient accumulation: {self.gradient_accumulation_steps}")

    def mixup_data(self, x, y, alpha=1.0):
        """Mixup augmentation with stronger mixing"""
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
        """Train one epoch with aggressive regularization"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if batch_idx == 0 and epoch == 1:
                print(f"🔄 Processing first batch (shape: {data.shape})...")
            elif batch_idx % 50 == 0:
                print(f"📊 Batch {batch_idx}/{len(self.train_loader)}")

            data, target = data.to(self.device), target.to(self.device)

            # Apply mixup with HIGHER probability for more regularization
            if np.random.random() < 0.6:  # Increased from 0.5 to 0.6
                data, target_a, target_b, lam = self.mixup_data(data, target, self.mixup_alpha)
                mixup = True
            else:
                mixup = False

            # Forward pass
            outputs = self.model(data)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                outputs, aux_outputs = outputs
            else:
                aux_outputs = None

            # Calculate loss
            if mixup:
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

            # Add L2 regularization on species-specific attention
            if hasattr(self.model, 'get_l2_regularization_loss'):
                l2_loss = self.model.get_l2_regularization_loss()
                loss = loss + l2_loss

            # Backward pass with gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # STRONGER gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Reduced from 1.0
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
        """Validation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

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
        """Main training loop with overfitting monitoring"""
        writer = SummaryWriter(f"runs/regularized_enhanced_{self.dataset_name}")

        best_acc = 0
        best_overfitting_gap = float('inf')
        patience_counter = 0
        start_epoch = 1

        # Load checkpoint if exists
        checkpoint_path = f"models/checkpoints/regularized_enhanced_latest.pth"
        if os.path.exists(checkpoint_path) and not config.get('new_model', False):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_acc = checkpoint['best_acc']
                print(f"🔄 Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}")
                print("🔄 Starting fresh training...")

        print(f"\n🚀 Starting STRONGLY REGULARIZED Training")
        print(f"📊 Dataset: {self.dataset_name}")
        print(f"🎯 Goal: Minimize overfitting gap (train_acc - val_acc)")
        print("=" * 80)

        for epoch in range(start_epoch, config['epochs'] + 1):
            # Training
            if epoch == start_epoch:
                print("🏁 Starting first epoch...")
            train_loss, train_acc = self.train_epoch(epoch)

            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate()

            # Calculate overfitting gap
            overfitting_gap = train_acc - val_acc

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
            writer.add_scalar('Overfitting_Gap', overfitting_gap, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)

            print(f"\nEpoch {epoch}/{config['epochs']}")
            print(f"📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"⚠️  Overfitting Gap: {overfitting_gap:.4f} ({overfitting_gap*100:.1f}%)")
            print(f"📚 Learning Rate: {current_lr:.2e}")

            # Save best model (prioritize validation accuracy, but track overfitting)
            if val_acc > best_acc:
                best_acc = val_acc
                best_overfitting_gap = overfitting_gap
                patience_counter = 0

                # Save model
                save_dir = Path(f"models/regularized_enhanced")
                save_dir.mkdir(parents=True, exist_ok=True)

                model_path = save_dir / f"best_model.pth"
                torch.save(self.model.state_dict(), model_path)

                print(f"🎉 New best model! Val Acc: {val_acc:.4f}, Gap: {overfitting_gap:.4f}")

                if overfitting_gap < 0.15:
                    print("🎯 EXCELLENT! Overfitting gap < 15%!")
                elif overfitting_gap < 0.25:
                    print("✅ Good! Overfitting gap < 25%")
                else:
                    print("⚠️  Still overfitting significantly")

            else:
                patience_counter += 1

            # Save checkpoint
            Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
                'best_acc': best_acc,
                'patience_counter': patience_counter
            }
            torch.save(checkpoint, checkpoint_path)

            print(f"🏆 Best val accuracy: {best_acc:.4f} (gap: {best_overfitting_gap:.4f})")
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
                save_dir = Path(f"models/regularized_enhanced")
                swa_path = save_dir / f"swa_model.pth"
                torch.save(self.swa_model.state_dict(), swa_path)
                print(f"✅ SWA model saved to {swa_path}")
            except Exception as e:
                print(f"⚠️ SWA finalization failed: {e}")

        writer.close()
        print(f"\n🎉 Training complete!")
        print(f"🏆 Best validation accuracy: {best_acc:.4f}")
        print(f"📉 Best overfitting gap: {best_overfitting_gap:.4f} ({best_overfitting_gap*100:.1f}%)")
        return best_acc

def main():
    parser = argparse.ArgumentParser(description='STRONGLY Regularized Training for Limited Data')

    parser.add_argument('--dataset', choices=['insectsound1000', 'insectset459', 'sina', 'xenocanto', 'combined'],
                       default='combined', help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=500, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=2e-4, help='Weight decay (INCREASED)')
    parser.add_argument('--swa-start', type=int, default=150, help='Start SWA at this epoch')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--new-model', action='store_true', help='Start fresh (ignore checkpoint)')

    args = parser.parse_args()

    print("🎯 ChirpKit STRONGLY REGULARIZED Training")
    print("🛡️  Optimized for limited data (40 samples/species)")
    print("=" * 80)
    print("📋 Regularization techniques:")
    print("   ✅ 50% dropout throughout")
    print("   ✅ 3-layer LSTM (enables internal dropout)")
    print("   ✅ Stronger label smoothing (0.15)")
    print("   ✅ Aggressive data augmentation (0.8 probability)")
    print("   ✅ MixUp (0.3 alpha, 60% probability)")
    print("   ✅ Higher weight decay (2e-4)")
    print("   ✅ Gradient clipping (0.5)")
    print("   ✅ Stochastic Weight Averaging")
    print("=" * 80)

    # Initialize trainer
    print("🔧 Initializing trainer...")
    trainer = StronglyRegularizedTrainer(
        dataset_name=args.dataset,
        device=args.device
    )

    # Load data
    print("📚 Loading and preparing datasets...")
    train_dataset, val_dataset, n_classes = trainer.load_data()

    # Create model
    print("🏗️ Building heavily regularized model...")
    model = trainer.create_model(n_classes)

    # Training configuration
    config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'patience': args.patience,
        'swa_start': args.swa_start,
        'new_model': args.new_model
    }

    # Setup training
    print("📋 Finalizing training configuration...")
    trainer.setup_training(train_dataset, val_dataset, config)

    # Train
    print("🎓 Starting training process...")
    final_accuracy = trainer.train(config)

    print(f"\n📈 Final validation accuracy: {final_accuracy:.4f}")
    if final_accuracy >= 0.50:
        print("🎉 SUCCESS! Reached 50%+ with better generalization!")
    else:
        print("📊 Progress made - overfitting should be significantly reduced")

if __name__ == "__main__":
    main()
