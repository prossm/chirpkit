#!/usr/bin/env python3
"""
Advanced Bayesian training script for ChirpKit
Implements uncertainty-aware learning, exploration strategies, and adaptive optimization
"""
import torch
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from models.bayesian_cnn_lstm import BayesianInsectClassifier
from training.advanced_trainer import AdvancedInsectTrainer
from utils.bayesian_utils import BayesianModelManager
from data.augmentation import InsectAudioAugmenter, AugmentedDataset

# Import existing data loading utilities
sys.path.append(os.path.join(os.path.dirname(__file__)))
from train_unified import NpyDataset, UnifiedTrainer

class BayesianTrainingPipeline:
    """Complete pipeline for advanced Bayesian insect classifier training"""
    
    def __init__(self, dataset_name='combined', device='auto'):
        self.dataset_name = dataset_name
        self.device = device
        self.splits_dir = Path(f'data/splits/{dataset_name}')
        
        if dataset_name == 'combined':
            self.splits_dir = Path('data/splits/combined')
        
        self.model = None
        self.trainer = None
        self.label_encoder = None
        
    def load_data(self):
        """Load and prepare data for Bayesian training"""
        print(f"🔄 Loading {self.dataset_name} dataset for Bayesian training...")
        
        # Use existing data loading logic
        unified_trainer = UnifiedTrainer(dataset_name=self.dataset_name)
        train_dataset_base, val_dataset = unified_trainer.load_data()
        
        self.label_encoder = unified_trainer.label_encoder
        n_classes = len(self.label_encoder.classes_)
        
        print(f"✅ Loaded data: {len(train_dataset_base)} train, {len(val_dataset)} val")
        print(f"🦗 Classes: {n_classes} species")
        
        return train_dataset_base, val_dataset, n_classes
    
    def create_bayesian_model(self, n_classes, pretrained_path=None):
        """Create or load Bayesian model"""
        print(f"🧠 Creating Bayesian model for {n_classes} classes...")
        
        self.model = BayesianInsectClassifier(
            n_classes=n_classes,
            dropout=0.3,
            uncertainty_method='monte_carlo'
        )
        
        if pretrained_path:
            print(f"📂 Loading pretrained weights from: {pretrained_path}")
            if Path(pretrained_path).suffix == '.pth':
                # Convert standard model to Bayesian
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ Converted standard model to Bayesian")
            else:
                # Load Bayesian model
                self.model.load_bayesian_state(pretrained_path)
                print("✅ Loaded Bayesian model state")
        
        return self.model
    
    def setup_advanced_training(self, train_dataset, val_dataset, config):
        """Setup advanced trainer with exploration and uncertainty"""
        # Create log directory based on dataset name
        log_dir = f'runs/bayesian_{self.dataset_name}_experiment'
        self.trainer = AdvancedInsectTrainer(self.model, device=self.device, log_dir=log_dir)
        
        self.trainer.setup_training(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=config['batch_size'],
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            exploration_prob=config['exploration_prob']
        )
        
        print(f"🎲 Advanced training configured:")
        print(f"   🔍 Exploration probability: {config['exploration_prob']}")
        print(f"   🧠 Uncertainty-aware loss enabled")
        print(f"   👁️ Attention diversity optimization")
        print(f"   🔄 Adaptive learning rate scheduling")
    
    def calibrate_model(self):
        """Calibrate model for better uncertainty estimates"""
        print("🌡️ Calibrating model for uncertainty...")
        temperature = self.model.calibrate_temperature(
            self.trainer.val_loader, 
            device=self.trainer.device
        )
        print(f"✅ Calibration complete. Temperature: {temperature:.3f}")
        return temperature
    
    def train_model(self, config):
        """Run advanced training with all enhancements"""
        print("\n🚀 Starting Advanced Bayesian Training")
        print("🎯 Goal: Build uncertainty-aware, exploratory insect expert")
        print("=" * 80)
        
        history = self.trainer.train(
            epochs=config['epochs'],
            patience=config['patience'],
            save_dir=config['save_dir'],
            resume=config.get('resume', False),
            checkpoint_dir=config.get('checkpoint_dir', 'models/bayesian_checkpoints')
        )
        
        return history
    
    def evaluate_model(self):
        """Comprehensive model evaluation with uncertainty analysis"""
        print("\n🔬 Evaluating model with uncertainty analysis...")
        
        manager = BayesianModelManager(device=self.trainer.device)
        manager.model = self.model
        manager.label_encoder = self.label_encoder
        manager.species_list = self.label_encoder.classes_.tolist()
        
        # Analyze uncertainty patterns
        analysis = manager.analyze_uncertainty_patterns(
            self.trainer.val_loader, 
            max_batches=20
        )
        
        print("\n📊 Uncertainty Analysis Results:")
        print(f"   🎯 Accuracy: {analysis['accuracy']:.4f}")
        print(f"   🔍 Mean Uncertainty: {analysis['mean_uncertainty']:.3f}")
        print(f"   ✅ Correct Predictions Uncertainty: {analysis['correct_predictions_uncertainty']:.3f}")
        print(f"   ❌ Incorrect Predictions Uncertainty: {analysis['incorrect_predictions_uncertainty']:.3f}")
        print(f"   📈 Uncertainty Separation: {analysis['incorrect_predictions_uncertainty'] - analysis['correct_predictions_uncertainty']:.3f}")
        
        return analysis
    
    def save_final_model(self, save_dir, model_name=None):
        """Save final calibrated model"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name is None:
            n_classes = len(self.label_encoder.classes_)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_name = f"bayesian_insect_classifier_{n_classes}species_{timestamp}"
        
        # Save Bayesian model
        model_path = save_dir / f"{model_name}.bayesian"
        self.model.save_bayesian_state(model_path)
        
        # Save label encoder
        import joblib
        encoder_path = save_dir / f"{model_name}_label_encoder.joblib"
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save model info
        info = {
            'model_name': model_name,
            'n_classes': len(self.label_encoder.classes_),
            'species_list': self.label_encoder.classes_.tolist(),
            'dataset': self.dataset_name,
            'uncertainty_method': self.model.uncertainty_method,
            'mc_samples': self.model.n_mc_samples,
            'is_calibrated': self.model.is_calibrated,
            'temperature': self.model.temperature.item() if self.model.is_calibrated else None,
            'training_completed': datetime.now().isoformat()
        }
        
        info_path = save_dir / f"{model_name}_info.json"
        import json
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n💾 Final model saved:")
        print(f"   🧠 Model: {model_path}")
        print(f"   🏷️ Labels: {encoder_path}")
        print(f"   📋 Info: {info_path}")
        
        return {
            'model_path': model_path,
            'encoder_path': encoder_path,
            'info_path': info_path
        }

def main():
    parser = argparse.ArgumentParser(description='Advanced Bayesian Insect Classifier Training')
    
    # Dataset arguments
    parser.add_argument('--dataset', choices=['insectsound1000', 'insectset459', 'sina', 'xenocanto', 'combined'], 
                       default='combined', help='Dataset to train on')
    
    # Model arguments
    parser.add_argument('--pretrained', type=str, help='Path to pretrained model to start from')
    parser.add_argument('--new-model', action='store_true', help='Start fresh training (default is to resume from checkpoint)')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint (default behavior)')  # Keep for backward compatibility
    parser.add_argument('--checkpoint-dir', type=str, default='models/bayesian_checkpoints',
                       help='Directory for saving/loading checkpoints')
    parser.add_argument('--model-name', type=str, help='Custom model name')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=500, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    
    # Advanced learning arguments
    parser.add_argument('--exploration-prob', type=float, default=0.3, 
                       help='Probability of applying exploratory augmentations')
    parser.add_argument('--uncertainty-weight', type=float, default=0.1,
                       help='Weight for uncertainty regularization')
    parser.add_argument('--diversity-weight', type=float, default=0.1,
                       help='Weight for attention diversity loss')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use (auto, cpu, cuda, mps)')

    # Output arguments
    parser.add_argument('--save-dir', type=str, default='models/bayesian_advanced',
                       help='Directory to save models')
    parser.add_argument('--no-calibration', action='store_true',
                       help='Skip uncertainty calibration')
    
    args = parser.parse_args()
    
    print("🧠 ChirpKit Advanced Bayesian Training")
    print("🎯 Building uncertainty-aware, exploratory insect expert")
    print("=" * 80)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎲 Exploration: {args.exploration_prob}")
    print(f"🔍 Uncertainty weight: {args.uncertainty_weight}")
    print(f"👁️ Diversity weight: {args.diversity_weight}")

    # Default behavior is to resume unless --new-model is specified
    should_resume = not args.new_model or args.resume
    print(f"🔄 Resume from checkpoint: {'Yes' if should_resume else 'No (fresh start)'}")
    print("=" * 80)

    # Initialize pipeline
    pipeline = BayesianTrainingPipeline(dataset_name=args.dataset, device=args.device)

    # Load data
    train_dataset, val_dataset, n_classes = pipeline.load_data()

    # Create Bayesian model
    model = pipeline.create_bayesian_model(n_classes, args.pretrained)

    # Training configuration

    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'exploration_prob': args.exploration_prob,
        'epochs': args.epochs,
        'patience': args.patience,
        'save_dir': args.save_dir,
        'resume': should_resume,
        'checkpoint_dir': args.checkpoint_dir
    }
    
    # Setup advanced training
    pipeline.setup_advanced_training(train_dataset, val_dataset, config)
    
    # Update loss weights
    pipeline.trainer.criterion.uncertainty_weight = args.uncertainty_weight
    pipeline.trainer.criterion.diversity_weight = args.diversity_weight
    
    # Train model
    history = pipeline.train_model(config)
    
    # Calibrate for better uncertainty
    if not args.no_calibration:
        pipeline.calibrate_model()
    
    # Evaluate with uncertainty analysis
    analysis = pipeline.evaluate_model()
    
    # Save final model
    saved_paths = pipeline.save_final_model(args.save_dir, args.model_name)
    
    print("\n🎉 Advanced Bayesian Training Complete!")
    print(f"🏆 Final Accuracy: {analysis['accuracy']:.4f}")
    print(f"🔍 Uncertainty Quality: {analysis['incorrect_predictions_uncertainty'] - analysis['correct_predictions_uncertainty']:.3f}")
    print(f"💾 Model saved to: {saved_paths['model_path']}")
    
    print("\n🧠 Your insect expert now has:")
    print("   ✅ Uncertainty self-awareness")
    print("   ✅ Exploratory learning strategies") 
    print("   ✅ Attention diversity optimization")
    print("   ✅ Adaptive learning rate scheduling")
    print("   ✅ Calibrated confidence estimates")
    print("\n🦗 Ready for effective and efficient insect identification!")

if __name__ == "__main__":
    main()