#!/usr/bin/env python3
"""
Unified training script for both InsectSound1000 and InsectSet459 datasets
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Label encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Augmentation
from data.augmentation import InsectAudioAugmenter, AugmentedDataset

# Custom Dataset for on-the-fly loading
class NpyDataset(Dataset):
    def __init__(self, features_path, labels_path, label_encoder=None):
        self.features = np.load(features_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')
        self.label_encoder = label_encoder
        if self.label_encoder is not None:
            self.labels = self.label_encoder.transform(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

class UnifiedTrainer:
    """Unified training for multiple datasets"""
    
    def __init__(self, dataset_name='insectsound1000', model_name=None):
        self.dataset_name = dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data paths based on dataset
        if dataset_name in ['insectsound1000', 'insectset459']:
            # Use splits from unified preprocessor
            self.splits_dir = Path(f'data/splits/{dataset_name}')
            if not self.splits_dir.exists():
                self.splits_dir = Path('data/splits')  # Fallback to original location
        elif dataset_name == 'combined':
            # Combined dataset mode - will load both datasets
            self.splits_dir = Path('data/splits/combined')
            self.splits_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: insectsound1000, insectset459, combined")
        
        # Model save paths - unified location with descriptive naming
        self.models_dir = Path('models/trained')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = Path('models/checkpoints')
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Model naming will be determined after loading data (based on species count)
        self.model_name = model_name
        
        # TensorBoard
        self.log_dir = Path('runs') / f'unified_model_experiment'
        
    def load_data(self):
        """Load datasets"""
        if self.dataset_name == 'combined':
            return self._load_combined_data()
        else:
            return self._load_single_dataset()
    
    def _load_single_dataset(self):
        """Load a single dataset"""
        print(f"📁 Loading data from: {self.splits_dir}")
        
        # Check which split files exist
        train_features = self.splits_dir / 'X_train.npy'
        train_labels = self.splits_dir / 'y_train.npy'
        val_features = self.splits_dir / 'X_val.npy'
        val_labels = self.splits_dir / 'y_val.npy'
        
        if not all(f.exists() for f in [train_features, train_labels, val_features, val_labels]):
            raise FileNotFoundError(f"Split files not found in {self.splits_dir}. Run preprocessing first.")
        
        # Fit label encoder on training labels
        train_labels_raw = np.load(train_labels)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(train_labels_raw)
        
        return self._create_datasets_from_splits(train_features, train_labels, val_features, val_labels)
    
    def _load_combined_data(self):
        """Load and combine both datasets"""
        print(f"🔄 Loading combined datasets...")
        
        # Check if combined splits already exist
        combined_train_features = self.splits_dir / 'X_train.npy'
        if combined_train_features.exists():
            print("📁 Found existing combined splits")
            return self._load_single_dataset()
        
        # Load both individual datasets
        datasets_to_combine = []
        all_train_features = []
        all_train_labels = []
        all_val_features = []
        all_val_labels = []
        
        for dataset in ['insectsound1000', 'insectset459']:
            splits_dir = Path(f'data/splits/{dataset}')
            if not splits_dir.exists():
                print(f"⚠️ Skipping {dataset} - splits not found at {splits_dir}")
                continue
                
            print(f"📂 Loading {dataset}...")
            try:
                train_feats = np.load(splits_dir / 'X_train.npy')
                train_labs = np.load(splits_dir / 'y_train.npy')
                val_feats = np.load(splits_dir / 'X_val.npy')
                val_labs = np.load(splits_dir / 'y_val.npy')
                
                all_train_features.append(train_feats)
                all_train_labels.append(train_labs)
                all_val_features.append(val_feats)
                all_val_labels.append(val_labs)
                datasets_to_combine.append(dataset)
                
                print(f"✅ {dataset}: {len(train_feats)} train, {len(val_feats)} val")
            except Exception as e:
                print(f"❌ Error loading {dataset}: {e}")
                continue
        
        if not datasets_to_combine:
            raise FileNotFoundError("No valid datasets found. Run preprocessing first.")
        
        # Combine arrays
        combined_train_features = np.concatenate(all_train_features, axis=0)
        combined_train_labels = np.concatenate(all_train_labels, axis=0)
        combined_val_features = np.concatenate(all_val_features, axis=0)
        combined_val_labels = np.concatenate(all_val_labels, axis=0)
        
        print(f"🔗 Combined: {len(combined_train_features)} train, {len(combined_val_features)} val")
        
        # Fit label encoder on combined training labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(combined_train_labels)
        
        # Save combined splits for future use
        np.save(self.splits_dir / 'X_train.npy', combined_train_features)
        np.save(self.splits_dir / 'y_train.npy', combined_train_labels)
        np.save(self.splits_dir / 'X_val.npy', combined_val_features)
        np.save(self.splits_dir / 'y_val.npy', combined_val_labels)
        
        return self._create_datasets_from_arrays(
            combined_train_features, combined_train_labels,
            combined_val_features, combined_val_labels
        )
    
    def _create_datasets_from_splits(self, train_features_path, train_labels_path, val_features_path, val_labels_path):
        """Create datasets from file paths"""
        # Create datasets using file paths
        train_dataset_base = NpyDataset(train_features_path, train_labels_path, self.label_encoder)
        val_dataset = NpyDataset(val_features_path, val_labels_path, self.label_encoder)
        
        return self._finalize_datasets(train_dataset_base, val_dataset)
    
    def _create_datasets_from_arrays(self, train_features, train_labels, val_features, val_labels):
        """Create datasets from numpy arrays"""
        # For arrays, we need to create temporary datasets differently
        # Convert to the format expected by NpyDataset by encoding labels first
        encoded_train_labels = self.label_encoder.transform(train_labels)
        encoded_val_labels = self.label_encoder.transform(val_labels)
        
        class ArrayDataset(Dataset):
            def __init__(self, features, labels):
                self.features = features
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                x = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)
                y = torch.tensor(self.labels[idx], dtype=torch.long)
                return x, y
        
        train_dataset_base = ArrayDataset(train_features, encoded_train_labels)
        val_dataset = ArrayDataset(val_features, encoded_val_labels)
        
        return self._finalize_datasets(train_dataset_base, val_dataset)
    
    def _finalize_datasets(self, train_dataset_base, val_dataset):
        """Common dataset finalization logic"""
        # Determine model naming based on species count
        n_species = len(self.label_encoder.classes_)
        if not self.model_name:
            self.model_name = f"insect_classifier_{n_species}species"
        
        print(f"🏷️ Model name: {self.model_name} ({n_species} species)")
        
        # Save label encoder
        import joblib
        joblib.dump(self.label_encoder, self.models_dir / f'{self.model_name}_label_encoder.joblib')
        
        # Optional: Add augmentation
        # augmenter = InsectAudioAugmenter()
        # train_dataset = AugmentedDataset(train_dataset_base, augmenter, augmentation_prob=0.3)
        train_dataset = train_dataset_base  # No augmentation for now
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32)
        
        print(f"✅ Loaded {len(train_dataset)} train, {len(val_dataset)} val samples")
        print(f"🦗 {len(self.label_encoder.classes_)} unique species:")
        for i, species in enumerate(self.label_encoder.classes_[:10]):  # Show first 10
            print(f"  {i}: {species}")
        if len(self.label_encoder.classes_) > 10:
            print(f"  ... and {len(self.label_encoder.classes_) - 10} more")
        
        return train_dataset_base, val_dataset
    
    def create_model(self):
        """Create and initialize model"""
        n_classes = len(self.label_encoder.classes_)
        self.model = SimpleCNNLSTMInsectClassifier(n_classes=n_classes)
        
        # Better weight initialization
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.model.apply(init_weights)
        self.model = self.model.to(self.device)
        
        print(f"🤖 Model created for {n_classes} classes on {self.device}")
        
        return self.model
    
    def setup_training(self, lr=3e-3, weight_decay=1e-4):
        """Setup optimizer, criterion, scheduler"""
        # Simple approach - standard CrossEntropyLoss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        
        print(f"⚙️ Training setup: lr={lr}, weight_decay={weight_decay}")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1, precision, recall, all_predictions, all_targets
    
    def train(self, max_epochs=100, patience=15, resume=True):
        """Main training loop"""
        print(f"🚀 Starting training: {max_epochs} max epochs, patience={patience}")
        
        # Setup TensorBoard
        writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Resume training
        start_epoch = 1
        best_val_acc = 0.0
        patience_counter = 0
        checkpoint_path = self.checkpoints_dir / f'{self.model_name}_checkpoint.pth'
        
        if resume and checkpoint_path.exists():
            print("📂 Found checkpoint, resuming training...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            patience_counter = checkpoint.get('patience_counter', 0)
            print(f"✅ Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
        
        # Training loop
        for epoch in range(start_epoch, max_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{max_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall, predictions, targets = self.validate()
            
            # Learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Print metrics
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
            print(f"Precision={val_precision:.4f}, Recall={val_recall:.4f}")
            print(f"LR: {current_lr:.2e}")
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('F1/val', val_f1, epoch)
            writer.add_scalar('Precision/val', val_precision, epoch)
            writer.add_scalar('Recall/val', val_recall, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Early stopping check
            if val_acc > best_val_acc + 1e-4:
                patience_counter = 0
                print(f"🎉 New best model! Val accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Step scheduler
            self.scheduler.step()
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'patience_counter': patience_counter,
                'timestamp': datetime.now().isoformat(),
                'dataset': self.dataset_name
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if epoch == start_epoch or val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.models_dir / f'{self.model_name}.pth')
                
                # Save training info
                training_info = {
                    'dataset': self.dataset_name,
                    'model_name': self.model_name,
                    'last_epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'best_epoch': epoch,
                    'n_classes': len(self.label_encoder.classes_),
                    'species_list': self.label_encoder.classes_.tolist(),
                    'last_updated': datetime.now().isoformat()
                }
                with open(self.models_dir / f'{self.model_name}_info.json', 'w') as f:
                    json.dump(training_info, f, indent=2)
            
            print(f"Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered! No improvement for {patience} epochs.")
                break
        
        writer.close()
        print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.4f}")
        print(f"💾 Model saved to: {self.models_dir}")
        
        return best_val_acc

def main():
    parser = argparse.ArgumentParser(description='Train insect classifier on unified datasets')
    parser.add_argument('--dataset', 
                       choices=['insectsound1000', 'insectset459', 'combined'], 
                       default='combined',
                       help='Dataset to train on (use "combined" for both datasets)')
    parser.add_argument('--model-name', help='Custom model name (optional)')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--no-resume', action='store_true', help='Don\'t resume from checkpoint')
    
    args = parser.parse_args()
    
    print(f"🦗 Unified Insect Classifier Training")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create trainer
    trainer = UnifiedTrainer(
        dataset_name=args.dataset,
        model_name=args.model_name
    )
    
    # Load data and create model
    trainer.load_data()
    trainer.create_model()
    trainer.setup_training(lr=args.lr, weight_decay=args.weight_decay)
    
    # Train
    best_acc = trainer.train(
        max_epochs=args.epochs,
        patience=args.patience,
        resume=not args.no_resume
    )
    
    print(f"\n🎯 Final best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()