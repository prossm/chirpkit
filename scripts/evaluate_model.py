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
from src.models.cnn_lstm import CNNLSTMInsectClassifier
from scripts.train_model import NpyDataset
from torch.utils.data import DataLoader
import os

def evaluate_model(model_path='models/trained/cnn_lstm_best.pth',
                  test_features_path='data/splits/X_test.npy',
                  test_labels_path='data/splits/y_test.npy',
                  label_encoder_path='models/trained/label_encoder.joblib'):
    """Comprehensive model evaluation"""
    
    print("🔍 Loading model and test data...")
    
    # Load label encoder
    label_encoder = joblib.load(label_encoder_path)
    n_classes = len(label_encoder.classes_)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMInsectClassifier(n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = NpyDataset(test_features_path, test_labels_path, label_encoder=label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"📊 Evaluating on {len(test_dataset)} test samples...")
    
    # Get predictions
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
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
    report = classification_report(all_targets, all_predictions, 
                                 target_names=species_names)
    print(report)
    
    # Save results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': classification_report(
            all_targets, all_predictions, target_names=species_names, output_dict=True
        ),
        'species_names': species_names.tolist(),
        'test_samples': len(test_dataset)
    }
    
    os.makedirs('models/evaluation', exist_ok=True)
    with open('models/evaluation/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=species_names, yticklabels=species_names)
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
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(species_names)), per_class_acc)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Species')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(species_names)), species_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('models/evaluation/per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"💾 Per-class accuracy plot saved to models/evaluation/per_class_accuracy.png")
    
    # ROC Curves (for multi-class)
    if n_classes > 2:
        # Binarize the output
        y_test_binarized = label_binarize(all_targets, classes=range(n_classes))
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, color in enumerate(colors):
            if i < len(species_names):  # Safety check
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], all_probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{species_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-Class ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('models/evaluation/roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"💾 ROC curves saved to models/evaluation/roc_curves.png")
    
    print(f"\n✅ Evaluation complete! Results saved to models/evaluation/")
    return results

if __name__ == "__main__":
    results = evaluate_model()
    print(f"\n🎯 Final Test Accuracy: {results['accuracy']:.4f}")