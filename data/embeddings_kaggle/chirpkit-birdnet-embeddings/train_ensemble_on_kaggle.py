#!/usr/bin/env python3
"""
Train Ensemble of BirdNET classifiers on Kaggle GPU

This script trains multiple models with different random seeds and
averages their predictions for better accuracy.

Expected improvement: +2-3% over single model (77% → 79-80%)

Usage on Kaggle:
    !python /kaggle/input/chirpkit-birdnet-embeddings/train_ensemble_on_kaggle.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import joblib
import json
from tqdm.notebook import tqdm

# Dataset class
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# Deep MLP Classifier (same as v1 - our best single model)
class DeepMLPClassifier(nn.Module):
    def __init__(self, n_classes, embedding_dim=1024, hidden_dims=[512, 256, 128], dropout=0.4):
        super().__init__()

        self.input_proj = nn.Linear(embedding_dim, hidden_dims[0])

        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        self.output = nn.Linear(hidden_dims[-1], n_classes)

    def forward(self, x):
        x = torch.relu(self.input_proj(x))
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def train_single_model(train_loader, val_loader, n_classes, device, seed, model_idx):
    """Train a single model with given random seed"""

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Configuration (same as v1, but more epochs and patience)
    EPOCHS = 300
    LR = 1e-3
    DROPOUT = 0.4
    PATIENCE = 50  # More patience to explore longer

    print(f"\n{'='*80}")
    print(f"🔧 Training Model {model_idx + 1} (seed={seed})")
    print(f"{'='*80}")

    # Initialize model
    model = DeepMLPClassifier(n_classes=n_classes, dropout=DROPOUT).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Print progress every epoch
        gap = train_acc - val_acc
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Gap: {gap:.2f}%", end='')

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f" 🎉 New best!")
        else:
            print()  # Just newline
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹️  Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\n✅ Model {model_idx + 1} complete! Best Val Acc: {best_val_acc:.2f}%")

    return model, best_val_acc


def ensemble_predict(models, val_loader, device, use_tta=True, tta_rounds=5):
    """Get ensemble predictions by averaging model outputs

    Args:
        models: List of trained models
        val_loader: Validation data loader
        device: CPU or CUDA device
        use_tta: Whether to use test-time augmentation
        tta_rounds: Number of augmented predictions per sample
    """

    all_predictions = []
    all_labels = []

    # Get predictions from each model
    for model in models:
        model.eval()
        model_predictions = []
        labels_list = []

        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)

                if use_tta:
                    # Test-time augmentation: add small Gaussian noise multiple times
                    tta_predictions = []

                    # Original prediction (no augmentation)
                    outputs = model(embeddings)
                    probs = torch.softmax(outputs, dim=1)
                    tta_predictions.append(probs)

                    # Augmented predictions
                    for _ in range(tta_rounds - 1):
                        # Add small random noise to embeddings (1% std - more aggressive)
                        noise = torch.randn_like(embeddings) * 0.01
                        aug_embeddings = embeddings + noise

                        outputs = model(aug_embeddings)
                        probs = torch.softmax(outputs, dim=1)
                        tta_predictions.append(probs)

                    # Average across TTA rounds
                    avg_tta_probs = torch.mean(torch.stack(tta_predictions), dim=0)
                    model_predictions.append(avg_tta_probs.cpu().numpy())
                else:
                    # No TTA - just regular prediction
                    outputs = model(embeddings)
                    probs = torch.softmax(outputs, dim=1)
                    model_predictions.append(probs.cpu().numpy())

                labels_list.append(labels.numpy())

        all_predictions.append(np.concatenate(model_predictions))
        if len(all_labels) == 0:
            all_labels = np.concatenate(labels_list)

    # Average predictions across models
    avg_predictions = np.mean(all_predictions, axis=0)
    ensemble_preds = np.argmax(avg_predictions, axis=1)

    # Calculate accuracy
    accuracy = 100. * np.sum(ensemble_preds == all_labels) / len(all_labels)

    return accuracy, ensemble_preds, all_labels


def train():
    # Configuration
    NUM_MODELS = 7  # Train 7 models with different seeds
    BATCH_SIZE = 512
    SEEDS = [42, 123, 456, 789, 2024, 3141, 5678]  # Different random seeds

    # Load data
    print("📊 Loading embeddings...")
    data_dir = Path("/kaggle/input/chirpkit-birdnet-embeddings")

    X_train = np.load(data_dir / "X_train_embeddings.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val_embeddings.npy")
    y_val = np.load(data_dir / "y_val.npy")

    # Load metadata
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    n_classes = metadata['n_classes']

    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Classes: {n_classes}")

    # Create datasets
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")

    print(f"\n🎯 Training Ensemble of {NUM_MODELS} models...")
    print(f"   Expected: 77% (single model) → 80-81% (ensemble + TTA)")
    print(f"   TTA: 10 rounds with 1% noise")
    print(f"{'='*80}\n")

    # Train multiple models
    models = []
    individual_accs = []

    for i, seed in enumerate(SEEDS):
        model, best_acc = train_single_model(
            train_loader, val_loader, n_classes, device, seed, i
        )
        models.append(model)
        individual_accs.append(best_acc)

    # Evaluate ensemble
    print(f"\n{'='*80}")
    print(f"🎯 ENSEMBLE EVALUATION")
    print(f"{'='*80}")

    print(f"\n📊 Individual Model Accuracies:")
    for i, acc in enumerate(individual_accs):
        print(f"   Model {i+1}: {acc:.2f}%")

    avg_individual = np.mean(individual_accs)
    print(f"\n📈 Average Individual: {avg_individual:.2f}%")

    # Evaluate without TTA
    print(f"\n🔍 Evaluating ensemble WITHOUT test-time augmentation...")
    ensemble_acc_no_tta, _, _ = ensemble_predict(models, val_loader, device, use_tta=False)
    print(f"🏆 Ensemble Accuracy (no TTA): {ensemble_acc_no_tta:.2f}%")

    # Evaluate with TTA
    print(f"\n🔍 Evaluating ensemble WITH test-time augmentation (10 rounds, 1% noise)...")
    ensemble_acc_tta, _, _ = ensemble_predict(models, val_loader, device, use_tta=True, tta_rounds=10)
    print(f"🏆 Ensemble Accuracy (with TTA): {ensemble_acc_tta:.2f}%")

    print(f"\n✨ TTA Improvement: +{ensemble_acc_tta - ensemble_acc_no_tta:.2f}%")
    print(f"📈 Total Improvement over baseline (37%): +{ensemble_acc_tta - 37:.2f}%")

    # Save all models
    print(f"\n💾 Saving models...")
    for i, model in enumerate(models):
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_acc': individual_accs[i],
            'seed': SEEDS[i],
            'n_classes': n_classes
        }, f'ensemble_model_{i+1}.pth')
        print(f"   ✓ ensemble_model_{i+1}.pth ({individual_accs[i]:.2f}%)")

    # Save ensemble metadata
    ensemble_info = {
        'num_models': NUM_MODELS,
        'individual_accuracies': individual_accs,
        'average_individual': float(avg_individual),
        'ensemble_accuracy_no_tta': float(ensemble_acc_no_tta),
        'ensemble_accuracy_with_tta': float(ensemble_acc_tta),
        'tta_improvement': float(ensemble_acc_tta - ensemble_acc_no_tta),
        'total_improvement': float(ensemble_acc_tta - avg_individual),
        'seeds': SEEDS,
        'n_classes': n_classes,
        'tta_rounds': 10,
        'tta_noise_std': 0.01
    }

    with open('ensemble_info.json', 'w') as f:
        json.dump(ensemble_info, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ ENSEMBLE TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"🏆 Final Ensemble Accuracy (with TTA): {ensemble_acc_tta:.2f}%")
    print(f"🎯 Target: 80%")
    if ensemble_acc_tta >= 80:
        print(f"🎉 TARGET ACHIEVED! 🎉")
    else:
        print(f"📊 Gap to target: {80 - ensemble_acc_tta:.2f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    train()
