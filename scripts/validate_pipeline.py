"""
Complete model validation and testing pipeline
"""
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_model import NpyDataset
from scripts.evaluate_model import evaluate_model
from src.models.cnn_lstm import CNNLSTMInsectClassifier
from src.data.augmentation import InsectAudioAugmenter, AugmentedDataset
from torch.utils.data import DataLoader
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt

def validate_model_consistency():
    """Validate model consistency across different runs"""
    print("🔄 Testing model consistency...")
    
    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_encoder = joblib.load('models/trained/label_encoder.joblib')
    n_classes = len(label_encoder.classes_)
    
    model = CNNLSTMInsectClassifier(n_classes=n_classes)
    model.load_state_dict(torch.load('models/trained/cnn_lstm_best.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Test on same data multiple times
    val_dataset = NpyDataset('data/splits/X_val.npy', 'data/splits/y_val.npy', label_encoder=label_encoder)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    predictions_list = []
    for run in range(3):
        predictions = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        predictions_list.append(predictions)
    
    # Check consistency
    consistent = all(np.array_equal(predictions_list[0], pred) for pred in predictions_list[1:])
    print(f"✅ Model consistency: {'PASSED' if consistent else 'FAILED'}")
    return consistent

def validate_augmentation_diversity():
    """Validate that augmentation creates diverse samples"""
    print("🎭 Testing augmentation diversity...")
    
    # Create augmented dataset
    label_encoder = joblib.load('models/trained/label_encoder.joblib')
    base_dataset = NpyDataset('data/splits/X_train.npy', 'data/splits/y_train.npy', label_encoder=label_encoder)
    
    augmenter = InsectAudioAugmenter()
    aug_dataset = AugmentedDataset(base_dataset, augmenter, augmentation_prob=1.0)
    
    # Test same sample multiple times
    sample_idx = 0
    original_sample, _ = base_dataset[sample_idx]
    
    augmented_samples = []
    for i in range(5):
        aug_sample, _ = aug_dataset[sample_idx]
        augmented_samples.append(aug_sample.numpy())
    
    # Check if augmentations are different
    diversity_scores = []
    for aug_sample in augmented_samples:
        diff = np.mean(np.abs(original_sample.numpy() - aug_sample))
        diversity_scores.append(diff)
    
    avg_diversity = np.mean(diversity_scores)
    print(f"✅ Augmentation diversity score: {avg_diversity:.6f}")
    print(f"✅ Augmentation test: {'PASSED' if avg_diversity > 1e-6 else 'FAILED'}")
    return avg_diversity > 1e-6

def test_model_robustness():
    """Test model robustness to input variations"""
    print("🛡️ Testing model robustness...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_encoder = joblib.load('models/trained/label_encoder.joblib')
    n_classes = len(label_encoder.classes_)
    
    model = CNNLSTMInsectClassifier(n_classes=n_classes)
    model.load_state_dict(torch.load('models/trained/cnn_lstm_best.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load test sample
    val_dataset = NpyDataset('data/splits/X_val.npy', 'data/splits/y_val.npy', label_encoder=label_encoder)
    test_sample, true_label = val_dataset[0]
    test_sample = test_sample.unsqueeze(0).to(device)
    
    # Get original prediction
    with torch.no_grad():
        original_output = model(test_sample)
        original_pred = torch.max(original_output, 1)[1].item()
    
    # Test with noise
    noise_levels = [0.001, 0.01, 0.1]
    robustness_scores = []
    
    for noise_level in noise_levels:
        consistent_predictions = 0
        total_tests = 10
        
        for _ in range(total_tests):
            noisy_sample = test_sample + torch.randn_like(test_sample) * noise_level
            with torch.no_grad():
                noisy_output = model(noisy_sample)
                noisy_pred = torch.max(noisy_output, 1)[1].item()
            
            if noisy_pred == original_pred:
                consistent_predictions += 1
        
        consistency_rate = consistent_predictions / total_tests
        robustness_scores.append(consistency_rate)
        print(f"  📊 Noise level {noise_level}: {consistency_rate:.2%} consistency")
    
    avg_robustness = np.mean(robustness_scores)
    print(f"✅ Average robustness: {avg_robustness:.2%}")
    return avg_robustness

def validate_training_data_quality():
    """Validate training data quality and distribution"""
    print("📊 Validating training data quality...")
    
    label_encoder = joblib.load('models/trained/label_encoder.joblib')
    
    # Load all datasets
    datasets = {
        'train': NpyDataset('data/splits/X_train.npy', 'data/splits/y_train.npy', label_encoder=label_encoder),
        'val': NpyDataset('data/splits/X_val.npy', 'data/splits/y_val.npy', label_encoder=label_encoder),
        'test': NpyDataset('data/splits/X_test.npy', 'data/splits/y_test.npy', label_encoder=label_encoder)
    }
    
    print("\n📈 Dataset Statistics:")
    for split_name, dataset in datasets.items():
        print(f"  {split_name.upper()}: {len(dataset)} samples")
        
        # Check class distribution
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label.item())
        
        unique, counts = np.unique(labels, return_counts=True)
        min_samples = np.min(counts)
        max_samples = np.max(counts)
        balance_ratio = min_samples / max_samples
        
        print(f"    Classes: {len(unique)}")
        print(f"    Balance ratio: {balance_ratio:.3f}")
        print(f"    Samples per class: {min_samples}-{max_samples}")
    
    return True

def generate_validation_report():
    """Generate comprehensive validation report"""
    print("📋 Generating validation report...")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Run all validation tests
    try:
        results['tests']['model_consistency'] = validate_model_consistency()
        results['tests']['augmentation_diversity'] = validate_augmentation_diversity()
        results['tests']['model_robustness'] = test_model_robustness()
        results['tests']['data_quality'] = validate_training_data_quality()
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        results['error'] = str(e)
    
    # Save results
    os.makedirs('models/validation', exist_ok=True)
    with open('models/validation/validation_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    passed_tests = sum(1 for test_result in results['tests'].values() if test_result is True)
    total_tests = len(results['tests'])
    
    print(f"\n📊 Validation Summary:")
    print(f"✅ Passed: {passed_tests}/{total_tests} tests")
    
    if passed_tests == total_tests:
        print("🎉 All validation tests PASSED!")
    else:
        print("⚠️  Some validation tests FAILED - check the details above")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting comprehensive model validation...")
    print("=" * 60)
    
    validation_results = generate_validation_report()
    
    print("\n🎯 Running final evaluation on test set...")
    try:
        test_results = evaluate_model()
        print(f"Final test accuracy: {test_results['accuracy']:.4f}")
    except Exception as e:
        print(f"❌ Test evaluation failed: {e}")
    
    print("\n✅ Validation pipeline complete!")