import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from pathlib import Path
from datetime import datetime
import json
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

try:
    # Try relative imports first (when used as package)
    from ..models.bayesian_cnn_lstm import BayesianInsectClassifier
    from ..data.augmentation import InsectAudioAugmenter, AugmentedDataset
    from ..utils.bayesian_utils import BayesianModelManager
except ImportError:
    # Fall back to absolute imports (when used from scripts)
    from chirpkit.models.bayesian_cnn_lstm import BayesianInsectClassifier
    from chirpkit.data.augmentation import InsectAudioAugmenter, AugmentedDataset
    from chirpkit.utils.bayesian_utils import BayesianModelManager

class ExploratoryDataset(Dataset):
    """Dataset that provides diverse, challenging examples for robust learning"""
    
    def __init__(self, base_dataset, exploration_strategies=None, exploration_prob=0.3):
        self.base_dataset = base_dataset
        self.exploration_prob = exploration_prob
        self.strategies = exploration_strategies or [
            'mixup',
            'cutmix', 
            'adversarial_noise',
            'temporal_shuffle',
            'frequency_dropout',
            'cross_species_blend'
        ]
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Temporarily disable exploration to fix tensor size issues
        # TODO: Debug and fix exploration strategies later
        return self.base_dataset[idx]
    
    def _apply_exploration(self, x, y, idx):
        """Apply random exploratory transformation"""
        strategy = random.choice(self.strategies)
        
        if strategy == 'mixup':
            return self._mixup(x, y, idx)
        elif strategy == 'cutmix':
            return self._cutmix(x, y, idx)
        elif strategy == 'adversarial_noise':
            return self._adversarial_noise(x, y)
        elif strategy == 'temporal_shuffle':
            return self._temporal_shuffle(x, y)
        elif strategy == 'frequency_dropout':
            return self._frequency_dropout(x, y)
        elif strategy == 'cross_species_blend':
            return self._cross_species_blend(x, y, idx)
        
        return x, y
    
    def _mixup(self, x, y, idx, alpha=0.2):
        """Mix two samples with random weight"""
        other_idx = random.randint(0, len(self.base_dataset) - 1)
        other_x, other_y = self.base_dataset[other_idx]
        
        lam = np.random.beta(alpha, alpha)
        mixed_x = lam * x + (1 - lam) * other_x
        
        # Create soft label
        if isinstance(y, torch.Tensor) and y.dim() == 0:
            # Convert to one-hot for mixing
            n_classes = self.base_dataset.label_encoder.classes_.shape[0] if hasattr(self.base_dataset, 'label_encoder') else 471
            y_onehot = torch.zeros(n_classes)
            other_y_onehot = torch.zeros(n_classes)
            y_onehot[y] = 1.0
            other_y_onehot[other_y] = 1.0
            mixed_y = lam * y_onehot + (1 - lam) * other_y_onehot
            return mixed_x, mixed_y
        
        return mixed_x, y
    
    def _cutmix(self, x, y, idx, alpha=1.0):
        """Cut and mix patches from different samples"""
        other_idx = random.randint(0, len(self.base_dataset) - 1)
        other_x, other_y = self.base_dataset[other_idx]
        
        lam = np.random.beta(alpha, alpha)
        
        # Get dimensions
        _, freq_bins, time_steps = x.shape
        
        # Random crop area
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(time_steps * cut_ratio)
        cut_h = int(freq_bins * cut_ratio)
        
        cx = np.random.randint(0, time_steps)
        cy = np.random.randint(0, freq_bins)
        
        x1 = max(0, cx - cut_w // 2)
        x2 = min(time_steps, cx + cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(freq_bins, cy + cut_h // 2)
        
        # Apply cutmix
        mixed_x = x.clone()
        mixed_x[:, y1:y2, x1:x2] = other_x[:, y1:y2, x1:x2]
        
        return mixed_x, y
    
    def _adversarial_noise(self, x, y, epsilon=0.01):
        """Add small adversarial noise to challenge the model"""
        noise = torch.randn_like(x) * epsilon
        return x + noise, y
    
    def _temporal_shuffle(self, x, y, shuffle_prob=0.3):
        """Randomly shuffle temporal segments"""
        _, freq_bins, time_steps = x.shape
        
        if random.random() < shuffle_prob:
            # Divide into segments and shuffle
            n_segments = random.randint(3, 8)
            segment_size = time_steps // n_segments
            
            shuffled_x = x.clone()
            indices = list(range(n_segments))
            random.shuffle(indices)
            
            for i, new_idx in enumerate(indices):
                start_old = new_idx * segment_size
                end_old = min((new_idx + 1) * segment_size, time_steps)
                start_new = i * segment_size
                end_new = min((i + 1) * segment_size, time_steps)
                
                shuffled_x[:, :, start_new:end_new] = x[:, :, start_old:end_old]
            
            return shuffled_x, y
        
        return x, y
    
    def _frequency_dropout(self, x, y, dropout_prob=0.2):
        """Randomly zero out frequency bands"""
        _, freq_bins, time_steps = x.shape
        
        mask = torch.ones_like(x)
        n_dropped = int(freq_bins * dropout_prob)
        
        if n_dropped > 0:
            drop_indices = torch.randperm(freq_bins)[:n_dropped]
            mask[:, drop_indices, :] = 0
        
        return x * mask, y
    
    def _cross_species_blend(self, x, y, idx, blend_strength=0.3):
        """Blend with a different species to create challenging examples"""
        # Get different species
        other_indices = [i for i in range(len(self.base_dataset)) 
                        if self.base_dataset[i][1] != y]
        
        if other_indices:
            other_idx = random.choice(other_indices)
            other_x, other_y = self.base_dataset[other_idx]
            
            alpha = random.uniform(0.1, blend_strength)
            blended_x = (1 - alpha) * x + alpha * other_x
            
            return blended_x, y
        
        return x, y

class UncertaintyAwareLoss(nn.Module):
    """Loss function that incorporates uncertainty for more effective learning"""
    
    def __init__(self, base_criterion, uncertainty_weight=0.1, diversity_weight=0.1, exploration_weight=0.05):
        super().__init__()
        self.base_criterion = base_criterion
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.exploration_weight = exploration_weight
        
    def forward(self, predictions, targets, uncertainty_data=None, attention_weights=None):
        """
        Compute comprehensive loss encouraging effective learning
        
        Args:
            predictions: Model predictions [batch, classes]
            targets: True labels [batch] or soft labels [batch, classes]  
            uncertainty_data: Uncertainty information from Bayesian forward pass
            attention_weights: Attention weights for diversity loss
        """
        # Base classification loss
        if targets.dim() > 1:
            # Soft labels (from mixup, etc.)
            base_loss = -torch.sum(targets * F.log_softmax(predictions, dim=1)) / targets.size(0)
        else:
            # Hard labels
            base_loss = self.base_criterion(predictions, targets)
        
        total_loss = base_loss
        loss_components = {'classification': base_loss.item()}
        
        # Uncertainty regularization - encourage calibrated uncertainty
        if uncertainty_data is not None:
            uncertainty_loss = self._compute_uncertainty_loss(predictions, uncertainty_data)
            total_loss += self.uncertainty_weight * uncertainty_loss
            loss_components['uncertainty'] = uncertainty_loss.item()
        
        # Attention diversity loss - encourage exploration
        if attention_weights is not None:
            diversity_loss = self._compute_attention_diversity_loss(attention_weights)
            total_loss += self.diversity_weight * diversity_loss
            loss_components['diversity'] = diversity_loss.item()
        
        # Exploration bonus - reward confident predictions on hard examples
        exploration_loss = self._compute_exploration_loss(predictions, targets)
        total_loss += self.exploration_weight * exploration_loss
        loss_components['exploration'] = exploration_loss.item()
        
        return total_loss, loss_components
    
    def _compute_uncertainty_loss(self, predictions, uncertainty_data):
        """Encourage well-calibrated uncertainty"""
        probs = F.softmax(predictions, dim=1)
        max_probs, predicted_classes = torch.max(probs, dim=1)
        
        # High confidence should correlate with low uncertainty
        epistemic_uncertainty = uncertainty_data.get('epistemic_uncertainty', torch.zeros_like(max_probs))
        
        # Penalize high confidence + high uncertainty (overconfident)
        overconfidence_penalty = max_probs * epistemic_uncertainty
        
        return overconfidence_penalty.mean()
    
    def _compute_attention_diversity_loss(self, attention_weights):
        """Encourage attention heads to explore different regions"""
        # Handle simplified attention weights from current model
        if attention_weights.dim() != 4:
            # Current model returns 2D attention, skip diversity loss
            return torch.tensor(0.0, device=attention_weights.device)

        # attention_weights: [batch, heads, seq, seq]
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Average attention patterns across sequence
        head_patterns = attention_weights.mean(dim=-1)  # [batch, heads, seq]
        
        # Compute pairwise similarities between heads
        similarities = []
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                pattern_i = head_patterns[:, i, :]
                pattern_j = head_patterns[:, j, :]
                
                # Cosine similarity
                similarity = F.cosine_similarity(pattern_i, pattern_j, dim=1)
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = torch.stack(similarities).mean()
            return avg_similarity  # Minimize similarity to encourage diversity
        
        return torch.tensor(0.0, device=attention_weights.device)
    
    def _compute_exploration_loss(self, predictions, targets):
        """Reward confident correct predictions (encourages bold exploration)"""
        probs = F.softmax(predictions, dim=1)
        
        if targets.dim() > 1:
            # Soft targets - use expected correctness
            target_probs = torch.sum(targets * probs, dim=1)
        else:
            # Hard targets
            target_probs = probs[torch.arange(probs.size(0)), targets]
        
        # Reward high confidence on correct predictions
        confidence_reward = target_probs * torch.log(target_probs + 1e-8)
        
        return -confidence_reward.mean()  # Minimize negative reward

class AdaptiveLearningScheduler:
    """Dynamic learning scheduler based on uncertainty and performance"""
    
    def __init__(self, optimizer, initial_lr=1e-4, uncertainty_threshold=0.5, 
                 adaptation_factor=0.8, patience=5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.uncertainty_threshold = uncertainty_threshold
        self.adaptation_factor = adaptation_factor
        self.patience = patience
        self.best_val_acc = 0
        self.patience_counter = 0
        self.current_phase = 'exploration'  # or 'exploitation'
        
    def step(self, val_acc, avg_uncertainty):
        """Adapt learning based on performance and uncertainty"""
        # Switch between exploration and exploitation phases
        if avg_uncertainty > self.uncertainty_threshold:
            if self.current_phase != 'exploration':
                print("🔍 Switching to EXPLORATION phase (high uncertainty)")
                self.current_phase = 'exploration'
                self._adjust_lr_for_exploration()
        else:
            if self.current_phase != 'exploitation':
                print("🎯 Switching to EXPLOITATION phase (low uncertainty)")
                self.current_phase = 'exploitation'
                self._adjust_lr_for_exploitation()
        
        # Standard performance-based adjustment
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            self._reduce_lr()
            self.patience_counter = 0
    
    def _adjust_lr_for_exploration(self):
        """Higher learning rate for exploration"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr * 1.5
    
    def _adjust_lr_for_exploitation(self):
        """Lower learning rate for fine-tuning"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr * 0.7
    
    def _reduce_lr(self):
        """Reduce learning rate when performance plateaus"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.adaptation_factor
        print(f"📉 Reduced learning rate to: {self.optimizer.param_groups[0]['lr']:.2e}")

class AdvancedInsectTrainer:
    """Advanced trainer with Bayesian uncertainty, exploration, and adaptive learning"""
    
    def __init__(self, model: BayesianInsectClassifier, device: str = 'auto', log_dir: str = None):
        self.model = model
        self.device = self._setup_device(device)
        self.model.to(self.device)

        # TensorBoard logging - use separate directory for Bayesian experiments
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f'runs/bayesian_experiment_{timestamp}'

        self.writer = SummaryWriter(log_dir)
        self.log_dir = Path(log_dir)
        print(f"📊 TensorBoard logging to: {log_dir}")

        # Display device info
        if self.device.type == 'mps':
            device_name = "MPS (Apple Silicon GPU)"
        elif self.device.type == 'cuda':
            device_name = f"CUDA (GPU {torch.cuda.get_device_name()})"
        else:
            device_name = "CPU"
        print(f"💻 Device: {device_name}")

        # Advanced loss function
        self.criterion = UncertaintyAwareLoss(
            base_criterion=nn.CrossEntropyLoss(),
            uncertainty_weight=0.1,
            diversity_weight=0.1,
            exploration_weight=0.05
        )

        # Metrics tracking
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'uncertainty_metrics': [],
            'exploration_metrics': [],
            'learning_phases': []
        }

        # Adaptive training optimization
        self.adaptive_mc_samples = {'train': 10, 'val': 5}  # Start with moderate samples
        self.recent_improvements = []  # Track recent accuracy improvements
        self.stagnation_epochs = 0  # Count epochs without improvement
        self.speed_boost_active = False
        self.gradient_accumulation_steps = 1  # Dynamic gradient accumulation

        # MPS compatibility fixes
        self.use_mps_fixes = (self.device.type == 'mps')

    def _safe_tensor_operation(self, tensor_op, *args, **kwargs):
        """Safely execute tensor operations with MPS fallbacks"""
        if self.use_mps_fixes:
            try:
                return tensor_op(*args, **kwargs)
            except RuntimeError as e:
                if "MPS" in str(e) or "mps" in str(e):
                    print(f"⚠️  MPS operation failed, falling back to CPU: {str(e)[:100]}...")
                    # Move to CPU, perform operation, then back to MPS
                    cpu_args = [arg.cpu() if torch.is_tensor(arg) else arg for arg in args]
                    result = tensor_op(*cpu_args, **kwargs)
                    return result.to(self.device) if torch.is_tensor(result) else result
                else:
                    raise e
        else:
            return tensor_op(*args, **kwargs)

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device with MPS compatibility checks"""
        if device == 'auto':
            if torch.backends.mps.is_available():
                # MPS has some compatibility issues with certain operations
                # For Bayesian training with MC dropout, CPU might be more stable
                print("⚠️  MPS available but using CPU for Bayesian training stability")
                print("   (MPS has known issues with some Bayesian operations)")
                return torch.device('cpu')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        elif device == 'mps':
            if torch.backends.mps.is_available():
                print("🚨 Warning: MPS may cause bus errors with Bayesian training")
                print("   Consider using --device cpu if you encounter crashes")
                return torch.device('mps')
            else:
                print("⚠️  MPS not available, falling back to CPU")
                return torch.device('cpu')
        return torch.device(device)

    def adapt_training_speed(self, current_accuracy, epoch):
        """Dynamically adjust training parameters for optimal speed/quality balance"""
        # Track recent improvements
        if len(self.recent_improvements) >= 5:
            self.recent_improvements.pop(0)

        if len(self.training_history['val_accuracy']) > 0:
            last_acc = self.training_history['val_accuracy'][-1]
            improvement = current_accuracy - last_acc
            self.recent_improvements.append(improvement)

        # Calculate average improvement over recent epochs
        recent_avg_improvement = sum(self.recent_improvements) / max(len(self.recent_improvements), 1)

        # Adaptive strategies based on training phase
        if recent_avg_improvement > 0.005:  # Strong improvement
            self.stagnation_epochs = 0
            # Increase MC samples for better uncertainty estimates during improvement
            self.adaptive_mc_samples['train'] = min(15, self.adaptive_mc_samples['train'] + 1)
            self.adaptive_mc_samples['val'] = min(8, self.adaptive_mc_samples['val'] + 1)
            status = "🚀 ACCELERATING (strong improvement)"

        elif recent_avg_improvement > 0.001:  # Moderate improvement
            self.stagnation_epochs = 0
            # Maintain current settings
            status = "⚡ STEADY (moderate improvement)"

        else:  # Stagnation or decline
            self.stagnation_epochs += 1

            if self.stagnation_epochs >= 3:  # Speed up if stagnating
                # More aggressive speed optimizations for plateau
                self.adaptive_mc_samples['train'] = max(3, self.adaptive_mc_samples['train'] - 2)
                self.adaptive_mc_samples['val'] = max(2, self.adaptive_mc_samples['val'] - 1)
                self.speed_boost_active = True
                status = f"🏃 SPEED BOOST (stagnant {self.stagnation_epochs} epochs)"

                # Additional plateau-breaking strategies
                if self.stagnation_epochs >= 6:
                    # Very aggressive speed boost for persistent plateaus
                    self.adaptive_mc_samples['train'] = 3
                    self.adaptive_mc_samples['val'] = 2
                    status = f"🚀 TURBO MODE (plateau {self.stagnation_epochs} epochs)"

                    # Learning rate cycling for plateau breaking
                    if self.stagnation_epochs % 4 == 0:
                        # Temporary LR boost every 4 stagnant epochs
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 1.5
                        status += " + LR BOOST"
            else:
                status = f"🐌 PATIENCE ({self.stagnation_epochs}/3 stagnant epochs)"

        return status

    def get_adaptive_validation_subset(self, val_loader, speed_boost_factor=2):
        """Use subset of validation data during speed boost for faster epochs"""
        if not self.speed_boost_active:
            return val_loader

        # Use every N-th batch for faster validation during speed boost
        subset_batches = []
        for i, batch in enumerate(val_loader):
            if i % speed_boost_factor == 0:
                subset_batches.append(batch)
                if len(subset_batches) >= len(val_loader) // speed_boost_factor:
                    break

        return subset_batches

    def setup_training(self, train_dataset, val_dataset, batch_size=32,
                      lr=1e-4, weight_decay=1e-4, exploration_prob=0.3):
        """Setup training with epoch-level playful exploration"""

        # Store base dataset for normal and playful epochs
        self.base_train_dataset = train_dataset
        self.exploration_prob = exploration_prob

        # Create normal data loader (used most of the time)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Advanced optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Adaptive scheduler
        self.scheduler = AdaptiveLearningScheduler(
            self.optimizer,
            initial_lr=lr,
            uncertainty_threshold=0.5
        )

        # Store batch size for playful epoch creation
        self.batch_size = batch_size

        print(f"🚀 Advanced training setup complete:")
        print(f"   📊 Train samples: {len(train_dataset)}")
        print(f"   🎯 Val samples: {len(val_dataset)}")
        print(f"   🎲 Exploration probability: {exploration_prob}")
        print(f"   💻 Device: {self.device}")

    def is_playful_epoch(self, epoch, play_interval=10):
        """Determine if this should be a playful exploration epoch"""
        return epoch % play_interval == 0

    def create_playful_dataloader(self):
        """Create a data loader with safe epoch-level augmentations"""
        from data.augmentation import InsectAudioAugmenter, AugmentedDataset

        # Create augmented dataset with safe transformations
        augmenter = InsectAudioAugmenter(sr=16000)  # Standard sample rate

        augmented_dataset = AugmentedDataset(
            self.base_train_dataset,
            augmenter,
            augmentation_prob=0.7  # More aggressive in playful epochs
        )

        return DataLoader(
            augmented_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

    def train_epoch(self, epoch):
        """Train one epoch with advanced learning strategies"""
        self.model.train()

        total_loss = 0
        loss_components_sum = {}
        uncertainty_metrics = []

        # Determine if this is a playful epoch
        is_playful = self.is_playful_epoch(epoch)

        # Choose appropriate data loader
        if is_playful:
            print(f"🎲 Playful Epoch {epoch}: Adding creative augmentations!")
            try:
                current_loader = self.create_playful_dataloader()
            except Exception as e:
                print(f"⚠️  Playful dataloader failed, using normal: {e}")
                current_loader = self.train_loader
        else:
            current_loader = self.train_loader

        progress_bar = tqdm(current_loader, desc=f"Epoch {epoch}" + (" 🎲" if is_playful else ""))
        
        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Adaptive Bayesian forward pass with uncertainty
            uncertainty_data = self.model.forward_with_uncertainty(
                X_batch, n_samples=self.adaptive_mc_samples['train'], return_attention=True
            )
            
            predictions = uncertainty_data['predictions']
            attention_weights = uncertainty_data.get('attention_weights')
            
            # Advanced loss computation
            loss, loss_components = self.criterion(
                predictions, y_batch, uncertainty_data, attention_weights
            )
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            for key, value in loss_components.items():
                loss_components_sum[key] = loss_components_sum.get(key, 0) + value
            
            # Track uncertainty metrics
            if uncertainty_data:
                batch_uncertainty = uncertainty_data['total_uncertainty'].mean().item()
                uncertainty_metrics.append(batch_uncertainty)
            
            # Update progress bar
            if batch_idx % 20 == 0:
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Uncertainty': f"{batch_uncertainty:.3f}" if uncertainty_metrics else "N/A"
                })
        
        # Epoch summary
        avg_loss = total_loss / len(self.train_loader)
        avg_uncertainty = np.mean(uncertainty_metrics) if uncertainty_metrics else 0
        
        avg_loss_components = {
            key: value / len(self.train_loader) 
            for key, value in loss_components_sum.items()
        }
        
        return avg_loss, avg_uncertainty, avg_loss_components
    
    def validate_epoch(self):
        """Validate with uncertainty analysis"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            val_progress = tqdm(self.val_loader, desc=f"Validation", leave=False)
            for X_batch, y_batch in val_progress:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Adaptive Bayesian inference
                uncertainty_data = self.model.forward_with_uncertainty(
                    X_batch, n_samples=self.adaptive_mc_samples['val']
                )
                
                predictions = uncertainty_data['predictions']
                loss = F.cross_entropy(predictions, y_batch)
                total_loss += loss.item()
                
                # Collect predictions and uncertainties
                all_predictions.append(predictions.cpu())
                all_targets.append(y_batch.cpu())
                all_uncertainties.append(uncertainty_data['total_uncertainty'].cpu())
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_uncertainties = torch.cat(all_uncertainties)
        
        predicted_classes = torch.argmax(all_predictions, dim=1)
        accuracy = (predicted_classes == all_targets).float().mean().item()
        avg_loss = total_loss / len(self.val_loader)
        avg_uncertainty = all_uncertainties.mean().item()
        
        # Uncertainty quality metrics
        correct_mask = (predicted_classes == all_targets)
        correct_uncertainty = all_uncertainties[correct_mask].mean().item() if correct_mask.any() else 0
        incorrect_uncertainty = all_uncertainties[~correct_mask].mean().item() if (~correct_mask).any() else 0
        uncertainty_separation = incorrect_uncertainty - correct_uncertainty
        
        validation_metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'avg_uncertainty': avg_uncertainty,
            'uncertainty_separation': uncertainty_separation,
            'correct_uncertainty': correct_uncertainty,
            'incorrect_uncertainty': incorrect_uncertainty
        }
        
        return validation_metrics
    
    def train(self, epochs=1000, patience=50, save_dir="models/advanced", resume=False, checkpoint_dir="models/bayesian_checkpoints"):
        """Main training loop with advanced learning strategies"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize training state
        start_epoch = 1
        best_val_acc = 0
        patience_counter = 0

        # Resume from checkpoint if requested
        if resume:
            checkpoint_info = self.load_checkpoint(checkpoint_dir)
            if checkpoint_info:
                start_epoch = checkpoint_info['start_epoch']
                best_val_acc = checkpoint_info['best_val_acc']
                # Always reset patience on new training run to allow exploration
                patience_counter = 0
                print(f"🔄 Patience reset for new training run (allowing exploration cycles)")
        
        print(f"🧠 Starting Advanced Bayesian Training")
        print(f"🎯 Target: Build uncertainty-aware, exploratory insect expert")
        print("=" * 70)
        
        for epoch in range(start_epoch, epochs + 1):
            print("\n" + "="*70)
            epoch_start = time.time()

            # Train epoch with exploration
            train_loss, train_uncertainty, loss_components = self.train_epoch(epoch)
            
            # Validate with uncertainty analysis
            print(f"🔍 Starting validation...")
            val_metrics = self.validate_epoch()
            print(f"✅ Validation complete. Accuracy: {val_metrics['accuracy']:.4f}")

            # Adaptive speed optimization
            speed_status = self.adapt_training_speed(val_metrics['accuracy'], epoch)

            # Adaptive learning rate scheduling
            self.scheduler.step(val_metrics['accuracy'], val_metrics['avg_uncertainty'])
            
            # Track history
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['uncertainty_metrics'].append(val_metrics)

            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Uncertainty/Train_Avg', train_uncertainty, epoch)
            self.writer.add_scalar('Uncertainty/Validation_Avg', val_metrics['avg_uncertainty'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Log loss components
            if loss_components:
                self.writer.add_scalar('Loss_Components/Base_Loss', loss_components.get('base_loss', 0), epoch)
                self.writer.add_scalar('Loss_Components/Uncertainty_Loss', loss_components.get('uncertainty_loss', 0), epoch)
                self.writer.add_scalar('Loss_Components/Diversity_Loss', loss_components.get('diversity_loss', 0), epoch)
                self.writer.add_scalar('Loss_Components/Exploration_Loss', loss_components.get('exploration_loss', 0), epoch)

            # Log uncertainty distribution
            if 'uncertainty_std' in val_metrics:
                self.writer.add_scalar('Uncertainty/Validation_Std', val_metrics['uncertainty_std'], epoch)

            # Log playful epoch indicator
            self.writer.add_scalar('Training/Playful_Epoch', 1.0 if self.is_playful_epoch(epoch) else 0.0, epoch)
            self.training_history['learning_phases'].append(self.scheduler.current_phase)
            
            # Print comprehensive metrics
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n🔍 Epoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"📊 Training Loss: {train_loss:.4f} | Uncertainty: {train_uncertainty:.3f}")
            print(f"🎯 Val Accuracy: {val_metrics['accuracy']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"🧠 Uncertainty Separation: {val_metrics['uncertainty_separation']:.3f}")
            print(f"🔬 Learning Phase: {self.scheduler.current_phase.upper()} | LR: {current_lr:.2e}")
            print(f"💡 Loss Components: {loss_components}")
            print(f"🎛️  Training Speed: {speed_status} | MC Samples: {self.adaptive_mc_samples['train']}/{self.adaptive_mc_samples['val']}")

            # Model saving and early stopping
            if val_metrics['accuracy'] > best_val_acc:
                previous_best = best_val_acc
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0

                # Save best model
                best_model_path = save_dir / "best_bayesian_model.pth"
                self.model.save_bayesian_state(best_model_path)

                # Save training history
                with open(save_dir / "training_history.json", 'w') as f:
                    json.dump(self.training_history, f, indent=2, default=str)

                print(f"\n🏆✨ NEW BEST MODEL! ✨🏆")
                print(f"🎯 Accuracy: {best_val_acc:.4f} (Previous: {previous_best:.4f})")
                print(f"💾 Saved to: {best_model_path}")
                print("🎉" + "="*50 + "🎉")
                print(f"⏱️  Patience: {patience_counter}/{patience} 🔥 improving!")
                print(f"🏆 Best accuracy so far: {best_val_acc:.4f}")

                # Save checkpoint immediately when model improves
                self.save_checkpoint(epoch, best_val_acc, patience_counter, checkpoint_dir)
                print(f"💾 Checkpoint saved: {checkpoint_dir}/latest_checkpoint.pth")
            else:
                patience_counter += 1
                print(f"⏱️  Patience: {patience_counter}/{patience} ⏳ waiting... (need {patience - patience_counter} more improvements)")
                print(f"🏆 Best accuracy so far: {best_val_acc:.4f}")

            # Always save current model (regardless of performance)
            current_model_path = save_dir / f"current_epoch_{epoch}_model.pth"
            self.model.save_bayesian_state(current_model_path)

            # Save checkpoint every epoch for maximum safety
            if val_metrics['accuracy'] <= best_val_acc:  # Only if not already saved above
                self.save_checkpoint(epoch, best_val_acc, patience_counter, checkpoint_dir)
                print(f"💾 Checkpoint saved: {checkpoint_dir}/latest_checkpoint.pth")
                print(f"📝 Current model saved: {current_model_path}")
            else:
                print(f"📝 Current model saved: {current_model_path}")

            if patience_counter >= patience:
                print(f"\n🛑 Early stopping: No improvement for {patience} epochs")
                break
        
        print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.4f}")
        print(f"💾 Model saved to: {save_dir}")

        # Close TensorBoard writer
        self.writer.close()
        print(f"📊 TensorBoard logs saved to: {self.log_dir}")

        return self.training_history

    def save_checkpoint(self, epoch, best_val_acc, patience_counter, checkpoint_dir="models/bayesian_checkpoints"):
        """Save training checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'patience_counter': patience_counter,
            'training_history': self.training_history,
        }

        checkpoint_path = checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_dir="models/bayesian_checkpoints"):
        """Load training checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_path = checkpoint_dir / 'latest_checkpoint.pth'

        if not checkpoint_path.exists():
            print(f"⚠️  No checkpoint found at {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        patience_counter = checkpoint['patience_counter']
        self.training_history = checkpoint['training_history']

        print(f"✅ Checkpoint loaded! Resuming from epoch {start_epoch}")
        print(f"🏆 Best accuracy so far: {best_val_acc:.4f}")

        return {
            'start_epoch': start_epoch,
            'best_val_acc': best_val_acc,
            'patience_counter': patience_counter
        }