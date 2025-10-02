#!/usr/bin/env python3
"""
Unified training script for multiple insect sound datasets
Supports: InsectSound1000, InsectSet459, SINA, Xeno-canto, and combined mode (609 species)
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
import time

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)  # Insert at beginning to prioritize local modules

from models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Label encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score

# Augmentation - use full path to avoid conflicts
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
    
    def __init__(self, dataset_name='insectsound1000', model_name=None, batch_size=64, gradient_accumulation_steps=1, force_cpu=False):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.reset_optimizer = False  # Flag for resetting optimizer on resume

        # Device selection: use CPU by default for stability (MPS causes bus errors)
        if force_cpu:
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Note: MPS disabled due to bus errors in complex training loops
        
        # Data paths based on dataset
        if dataset_name in ['insectsound1000', 'insectset459', 'sina', 'xenocanto']:
            # Use splits from unified preprocessor
            self.splits_dir = Path(f'data/splits/{dataset_name}')
            if not self.splits_dir.exists():
                self.splits_dir = Path('data/splits')  # Fallback to original location
        elif dataset_name == 'combined':
            # Combined dataset mode - will load all available datasets
            self.splits_dir = Path('data/splits/combined')
            self.splits_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: insectsound1000, insectset459, sina, xenocanto, combined")
        
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
        
        # Fit label encoder on ALL labels (train + val) to avoid unseen label errors
        train_labels_raw = np.load(train_labels)
        val_labels_raw = np.load(val_labels)
        all_labels = np.concatenate([train_labels_raw, val_labels_raw])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_labels)
        print(f"🏷️  Label encoder fitted on {len(self.label_encoder.classes_)} unique species")
        
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
        
        for dataset in ['insectsound1000', 'insectset459', 'sina', 'xenocanto']:
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

        # Filter out species with too few samples (< 10 in training set)
        from collections import Counter
        train_species_counts = Counter(combined_train_labels)
        min_samples_per_species = 10

        species_to_keep = {species for species, count in train_species_counts.items()
                          if count >= min_samples_per_species}

        if len(species_to_keep) < len(train_species_counts):
            filtered_species = len(train_species_counts) - len(species_to_keep)
            print(f"🗑️  Filtering out {filtered_species} species with < {min_samples_per_species} training samples")

            # Filter training data
            train_mask = np.array([label in species_to_keep for label in combined_train_labels])
            combined_train_features = combined_train_features[train_mask]
            combined_train_labels = combined_train_labels[train_mask]

            # Filter validation data
            val_mask = np.array([label in species_to_keep for label in combined_val_labels])
            combined_val_features = combined_val_features[val_mask]
            combined_val_labels = combined_val_labels[val_mask]

            print(f"✅ After filtering: {len(combined_train_features)} train, {len(combined_val_features)} val")

        # Fit label encoder on ALL labels (train + val) to avoid unseen label errors
        all_labels = np.concatenate([combined_train_labels, combined_val_labels])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_labels)
        print(f"🏷️  Label encoder fitted on {len(self.label_encoder.classes_)} unique species")

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
        
        # Add augmentation to improve generalization
        augmenter = InsectAudioAugmenter()
        train_dataset = AugmentedDataset(train_dataset_base, augmenter, augmentation_prob=0.5)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Store dataset reference for manual shuffling
        self.train_dataset = train_dataset
        
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
    
    def setup_training(self, lr=1e-4, weight_decay=1e-4, diversity_weight=0.1, use_class_weights=True, label_smoothing=0.1,
                       use_cosine_schedule=True, warmup_epochs=10, max_epochs=2000):
        """Setup optimizer, criterion, scheduler"""
        # Use AdamW optimizer for better performance
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.base_lr = lr  # Store base learning rate for warmup
        self.use_cosine_schedule = use_cosine_schedule
        self.warmup_epochs = warmup_epochs

        # Compute class weights for imbalanced dataset
        if use_class_weights and hasattr(self, 'train_dataset'):
            from collections import Counter
            import torch

            # Get all training labels
            all_labels = []
            for i in range(len(self.train_dataset.dataset) if hasattr(self.train_dataset, 'dataset') else len(self.train_dataset)):
                try:
                    _, label = self.train_dataset.dataset[i] if hasattr(self.train_dataset, 'dataset') else self.train_dataset[i]
                    all_labels.append(label.item())
                except:
                    pass

            # Calculate class weights (inverse frequency)
            label_counts = Counter(all_labels)
            n_samples = len(all_labels)
            n_classes = len(self.label_encoder.classes_)

            # Weight = n_samples / (n_classes * count_for_class)
            class_weights = torch.zeros(n_classes)
            for class_idx, count in label_counts.items():
                class_weights[class_idx] = n_samples / (n_classes * count)

            # Normalize weights to prevent extreme values
            class_weights = class_weights / class_weights.mean()
            class_weights = class_weights.to(self.device)

            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            print(f"⚖️  Using class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}, mean={class_weights.mean():.2f}")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        print(f"✨ Label smoothing: {label_smoothing} (prevents overconfidence)")
        self.diversity_weight = diversity_weight

        # Learning rate scheduler setup
        n_classes = len(self.label_encoder.classes_)

        if use_cosine_schedule:
            # Cosine Annealing with Warmup (modern approach)
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max_epochs // 4,  # First restart after 1/4 of training
                T_mult=2,  # Double the period after each restart
                eta_min=1e-7  # Minimum learning rate
            )
            scheduler_name = f"CosineAnnealing with {warmup_epochs}-epoch warmup"
        else:
            # ReduceLROnPlateau (fallback)
            scheduler_patience = max(10, min(20, n_classes // 30))
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=scheduler_patience,
                min_lr=1e-7,
                verbose=True
            )
            scheduler_name = f"ReduceLROnPlateau (patience={scheduler_patience})"

        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        print(f"⚙️ Training setup: AdamW optimizer, lr={lr}, weight_decay={weight_decay}")
        print(f"⚙️ Batch size: {self.batch_size} x {self.gradient_accumulation_steps} accumulation = {effective_batch_size} effective")
        print(f"⚙️ LR Scheduler: {scheduler_name}")
        print(f"🎲 Attention diversity weight: {diversity_weight}")
        if use_class_weights:
            print(f"⚖️  Class-weighted loss: ENABLED")
    
    def shuffle_data_each_epoch(self):
        """Recreate data loader with new shuffle for each epoch"""
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True
        )
        
    def train_epoch(self, epoch):
        """Train for one epoch with attention diversity loss"""
        self.model.train()

        # Apply warmup if using cosine schedule
        if self.use_cosine_schedule and epoch <= self.warmup_epochs:
            # Linear warmup: gradually increase LR from 0 to base_lr
            warmup_factor = epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * warmup_factor
            if epoch == 1:
                print(f"🔥 Warmup: LR = {self.base_lr * warmup_factor:.2e}")

        total_loss = 0
        total_diversity_loss = 0
        total_correct = 0
        total_samples = 0

        total_batches = len(self.train_loader)

        for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            # Real-time progress bar
            progress_pct = (batch_idx + 1) / total_batches * 100
            bar_length = 30
            filled_length = int(bar_length * progress_pct // 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"    📊 Epoch {epoch} [{bar}] {batch_idx + 1}/{total_batches} ({progress_pct:.1f}%)", end='\r', flush=True)

            # Forward pass with attention weights
            outputs, attention_weights = self.model(X_batch, return_attention=True)

            # Classification loss
            classification_loss = self.criterion(outputs, y_batch)

            # Attention diversity loss (encourage exploration)
            diversity_loss = self.model.compute_attention_diversity_loss(attention_weights)

            # Combined loss (normalize by accumulation steps)
            total_batch_loss = (classification_loss + self.diversity_weight * diversity_loss) / self.gradient_accumulation_steps

            total_batch_loss.backward()

            # Only update weights after accumulating gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += classification_loss.item()
            total_diversity_loss += diversity_loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

        # Clear progress line and move to next line
        print()  # Move to next line after progress tracking

        avg_loss = total_loss / len(self.train_loader)
        avg_diversity_loss = total_diversity_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy, avg_diversity_loss
    
    def validate(self):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        total_val_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(self.val_loader):
                # Validation progress bar
                progress_pct = (batch_idx + 1) / total_val_batches * 100
                bar_length = 30
                filled_length = int(bar_length * progress_pct // 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"    🔍 Validating [{bar}] {batch_idx + 1}/{total_val_batches} ({progress_pct:.1f}%)", end='\r', flush=True)
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        # Clear validation progress line
        print()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1, precision, recall, all_predictions, all_targets
    
    def train(self, max_epochs=2000, patience=15, resume=True):
        """Main training loop"""
        print(f"🚀 Starting training: {max_epochs} max epochs, patience={patience}")
        
        # Setup TensorBoard
        writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Resume training
        start_epoch = 1
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        checkpoint_path = self.checkpoints_dir / f'{self.model_name}_checkpoint.pth'
        
        if resume and checkpoint_path.exists():
            print("📂 Found checkpoint, resuming training...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Check if we should reset optimizer/scheduler (for learning rate changes)
            reset_optimizer = getattr(self, 'reset_optimizer', False)
            if not reset_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✅ Resumed optimizer and scheduler states")
            else:
                print(f"🔄 Reset optimizer and scheduler with new learning rate")
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            # Reset patience when resuming with new training setup
            patience_counter = 0  # Fresh start for patience
            print(f"✅ Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
            print(f"🔄 Reset patience counter for new training cycle")
        
        # Training loop
        training_start_time = time.time()
        for epoch in range(start_epoch, max_epochs + 1):
            # Ensure fresh shuffling each epoch
            self.shuffle_data_each_epoch()
            epoch_start_time = time.time()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{max_epochs} - Started at {current_time}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc, diversity_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall, predictions, targets = self.validate()
            
            # Learning rate (handle different scheduler types)
            try:
                current_lr = self.optimizer.param_groups[0]['lr']
            except:
                current_lr = self.scheduler.get_last_lr()[0]
            
            # Calculate epoch timing
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            total_elapsed = epoch_end_time - training_start_time
            
            # Estimate remaining time
            epochs_completed = epoch - start_epoch + 1
            avg_epoch_time = total_elapsed / epochs_completed
            epochs_remaining = max_epochs - epoch
            estimated_remaining = avg_epoch_time * epochs_remaining
            
            # Print metrics with timing
            completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"🎲 Attention Diversity Loss: {diversity_loss:.4f}")
            print(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
            print(f"Precision={val_precision:.4f}, Recall={val_recall:.4f}")
            print(f"LR: {current_lr:.2e}")
            print(f"⏱️  Epoch Duration: {epoch_duration:.1f}s | Completed at: {completion_time}")
            print(f"📊 Avg Epoch Time: {avg_epoch_time:.1f}s | Est. Remaining: {estimated_remaining/3600:.1f}h")
            print(f"🏆 Best Model: Epoch {best_epoch}, Val Acc {best_val_acc:.4f}")
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('F1/val', val_f1, epoch)
            writer.add_scalar('Precision/val', val_precision, epoch)
            writer.add_scalar('Recall/val', val_recall, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            writer.add_scalar('Attention_Diversity_Loss', diversity_loss, epoch)
            
            # Early stopping check
            if val_acc > best_val_acc + 1e-4:
                patience_counter = 0
                print(f"🎉 New best model! Val accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Step scheduler (different calls for different scheduler types)
            if self.use_cosine_schedule:
                self.scheduler.step()  # CosineAnnealing doesn't need metric
            else:
                self.scheduler.step(val_acc)  # ReduceLROnPlateau needs validation accuracy
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'patience_counter': patience_counter,
                'timestamp': datetime.now().isoformat(),
                'dataset': self.dataset_name
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model only when we improve
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.models_dir / f'{self.model_name}.pth')
                print(f"💾 Saved new best model with accuracy: {best_val_acc:.4f}")
                
                # Save training info only for the best model
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
                       choices=['insectsound1000', 'insectset459', 'sina', 'xenocanto', 'combined'], 
                       default='combined',
                       help='Dataset to train on (use "combined" for all datasets)')
    parser.add_argument('--model-name', help='Custom model name (optional)')
    parser.add_argument('--epochs', type=int, default=2000, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (50 for complex datasets)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (optimized for batch size 16)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (8-16 recommended for 609 classes)')
    parser.add_argument('--gradient-accumulation', type=int, default=4, help='Gradient accumulation steps (effective batch = batch-size * this)')
    parser.add_argument('--diversity-weight', type=float, default=0.02, help='Attention diversity loss weight (lower = more specialization)')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing factor (0.0-0.2, prevents overconfidence)')
    parser.add_argument('--use-cosine-schedule', action='store_true', help='Use cosine annealing schedule instead of ReduceLROnPlateau')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Number of warmup epochs (for cosine schedule)')
    parser.add_argument('--no-class-weights', action='store_true', help='Disable class weighting for imbalanced data')
    parser.add_argument('--no-resume', action='store_true', help='Don\'t resume from checkpoint')
    parser.add_argument('--reset-optimizer', action='store_true', help='Reset optimizer and scheduler while keeping model weights')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU training (disable GPU acceleration)')
    
    args = parser.parse_args()
    
    print(f"🦗 Unified Insect Classifier Training")
    print(f"Dataset: {args.dataset}")
    # Device info
    if args.force_cpu:
        device_name = "CPU (forced)"
    elif torch.backends.mps.is_available():
        device_name = "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        device_name = "CUDA"
    else:
        device_name = "CPU"
    print(f"Device: {device_name}")
    
    # Create trainer
    trainer = UnifiedTrainer(
        dataset_name=args.dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        force_cpu=args.force_cpu
    )
    
    # Load data and create model
    trainer.load_data()
    trainer.create_model()
    trainer.setup_training(
        lr=args.lr,
        weight_decay=args.weight_decay,
        diversity_weight=args.diversity_weight,
        use_class_weights=not args.no_class_weights,
        label_smoothing=args.label_smoothing,
        use_cosine_schedule=args.use_cosine_schedule,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs
    )
    
    # Set reset_optimizer flag if requested
    if args.reset_optimizer:
        trainer.reset_optimizer = True
    
    # Train
    best_acc = trainer.train(
        max_epochs=args.epochs,
        patience=args.patience,
        resume=not args.no_resume
    )
    
    print(f"\n🎯 Final best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()