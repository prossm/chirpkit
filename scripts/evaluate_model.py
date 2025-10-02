"""
Comprehensive evaluation script for the insect classifier
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
from torch.utils.data import DataLoader, Dataset

# Dataset class
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

def apply_tta_augmentations(x):
    """
    Apply test-time augmentation to audio spectrogram

    Args:
        x: [batch, 1, freq, time] spectrogram tensor

    Returns:
        List of augmented versions including original
    """
    augmentations = [x]  # Original

    # Time shift augmentations (circular shift along time axis)
    time_shifts = [10, -10, 20, -20]  # Shift by frames
    for shift in time_shifts:
        if shift != 0:
            augmentations.append(torch.roll(x, shifts=shift, dims=3))

    # Frequency shift augmentations (circular shift along freq axis)
    freq_shifts = [5, -5]  # Shift by frequency bins
    for shift in freq_shifts:
        if shift != 0:
            augmentations.append(torch.roll(x, shifts=shift, dims=2))

    # Time masking (mask random time segments) - helps with robustness
    time_masked = x.clone()
    time_len = x.shape[3]
    mask_len = time_len // 10  # Mask 10% of time
    mask_start = np.random.randint(0, time_len - mask_len)
    time_masked[:, :, :, mask_start:mask_start+mask_len] = 0
    augmentations.append(time_masked)

    # Frequency masking (mask random freq bands)
    freq_masked = x.clone()
    freq_len = x.shape[2]
    mask_len = freq_len // 10  # Mask 10% of frequencies
    mask_start = np.random.randint(0, freq_len - mask_len)
    freq_masked[:, :, mask_start:mask_start+mask_len, :] = 0
    augmentations.append(freq_masked)

    return augmentations

def predict_with_tta(model, x, device, n_augmentations=9):
    """
    Make prediction with test-time augmentation

    Args:
        model: trained model
        x: [batch, 1, freq, time] input tensor
        device: torch device
        n_augmentations: number of augmentations to use (including original)

    Returns:
        averaged_probabilities: [batch, n_classes] probability distribution
    """
    model.eval()
    augmented_versions = apply_tta_augmentations(x)[:n_augmentations]

    all_predictions = []
    with torch.no_grad():
        for aug_x in augmented_versions:
            aug_x = aug_x.to(device)
            outputs = model(aug_x)
            probabilities = torch.softmax(outputs, dim=1)
            all_predictions.append(probabilities)

    # Average predictions across all augmentations
    averaged_probabilities = torch.stack(all_predictions).mean(dim=0)

    return averaged_probabilities

def evaluate_model(model_path='models/trained/insect_classifier_609species.pth',
                  test_features_path='data/splits/combined/X_val.npy',
                  test_labels_path='data/splits/combined/y_val.npy',
                  label_encoder_path='models/trained/insect_classifier_609species_label_encoder.joblib',
                  use_tta=False,
                  n_augmentations=9):
    """
    Comprehensive model evaluation

    Args:
        model_path: Path to trained model
        test_features_path: Path to test features
        test_labels_path: Path to test labels
        label_encoder_path: Path to label encoder
        use_tta: Use test-time augmentation (slower but more accurate)
        n_augmentations: Number of augmentations for TTA
    """
    
    print("🔍 Loading model and test data...")
    
    # Load label encoder
    label_encoder = joblib.load(label_encoder_path)
    n_classes = len(label_encoder.classes_)
    
    # Load model
    # Setup device with MPS support
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = SimpleCNNLSTMInsectClassifier(n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = NpyDataset(test_features_path, test_labels_path, label_encoder=label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if use_tta:
        print(f"📊 Evaluating on {len(test_dataset)} test samples with TTA ({n_augmentations} augmentations)...")
    else:
        print(f"📊 Evaluating on {len(test_dataset)} test samples...")

    # Get predictions
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if use_tta:
                # Use test-time augmentation for better accuracy
                probabilities = predict_with_tta(model, X_batch, device, n_augmentations)
            else:
                # Standard inference
                outputs = model(X_batch)
                probabilities = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(probabilities, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_probabilities = np.array(all_probabilities)
    species_names = label_encoder.classes_
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted'
    )
    
    print(f"\n📈 Overall Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    # Get unique labels present in test set
    unique_labels = np.unique(all_targets)
    present_species_names = [species_names[i] for i in unique_labels]

    report = classification_report(all_targets, all_predictions,
                                 labels=unique_labels,
                                 target_names=present_species_names,
                                 zero_division=0)
    print(report)
    
    # Save results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': classification_report(
            all_targets, all_predictions,
            labels=unique_labels,
            target_names=present_species_names,
            output_dict=True,
            zero_division=0
        ),
        'species_names': species_names.tolist(),
        'species_in_test': len(unique_labels),
        'total_species': len(species_names),
        'test_samples': len(test_dataset)
    }
    
    os.makedirs('models/evaluation', exist_ok=True)
    with open('models/evaluation/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=unique_labels)

    # Only plot confusion matrix if not too many classes
    if len(unique_labels) <= 50:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=present_species_names, yticklabels=present_species_names)
    else:
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                    xticklabels=present_species_names, yticklabels=present_species_names)
    plt.title('Confusion Matrix - Insect Species Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/evaluation/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved to models/evaluation/confusion_matrix.png")
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # Sort by accuracy to show worst performers
    sorted_indices = np.argsort(per_class_acc)
    sorted_species = [present_species_names[i] for i in sorted_indices]
    sorted_acc = per_class_acc[sorted_indices]

    # Show worst 20 and best 20 performers
    print(f"\n🔴 Worst 20 performing species:")
    for i in range(min(20, len(sorted_species))):
        support = cm.sum(axis=1)[sorted_indices[i]]
        print(f"  {sorted_species[i]:40s}: {sorted_acc[i]:.3f} (n={int(support)})")

    print(f"\n🟢 Best 20 performing species:")
    for i in range(max(0, len(sorted_species)-20), len(sorted_species)):
        support = cm.sum(axis=1)[sorted_indices[i]]
        print(f"  {sorted_species[i]:40s}: {sorted_acc[i]:.3f} (n={int(support)})")

    plt.figure(figsize=(16, 6))
    bars = plt.bar(range(len(present_species_names)), per_class_acc)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Species')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(present_species_names)), present_species_names, rotation=90, ha='right', fontsize=6)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('models/evaluation/per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Per-class accuracy plot saved to models/evaluation/per_class_accuracy.png")
    
    # Skip ROC curves for large number of classes (too slow and cluttered)
    print(f"\n⏭️  Skipping ROC curves (too many classes: {len(unique_labels)})")
    
    print(f"\n✅ Evaluation complete! Results saved to models/evaluation/")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate insect classifier')
    parser.add_argument('--use-tta', action='store_true', help='Use test-time augmentation (slower but more accurate)')
    parser.add_argument('--n-augmentations', type=int, default=9, help='Number of augmentations for TTA')
    parser.add_argument('--model-path', default='models/trained/insect_classifier_609species.pth', help='Path to model')
    parser.add_argument('--features', default='data/splits/combined/X_val.npy', help='Test features path')
    parser.add_argument('--labels', default='data/splits/combined/y_val.npy', help='Test labels path')
    parser.add_argument('--label-encoder', default='models/trained/insect_classifier_609species_label_encoder.joblib', help='Label encoder path')

    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model_path,
        test_features_path=args.features,
        test_labels_path=args.labels,
        label_encoder_path=args.label_encoder,
        use_tta=args.use_tta,
        n_augmentations=args.n_augmentations
    )
    print(f"\n🎯 Final Test Accuracy: {results['accuracy']:.4f}")