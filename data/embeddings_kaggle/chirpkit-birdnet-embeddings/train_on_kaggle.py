#!/usr/bin/env python3
"""
Train BirdNET classifier on Kaggle GPU

This script trains on pre-extracted embeddings for fast GPU training.

Usage on Kaggle:
    1. Add this dataset as input
    2. Create new notebook
    3. Copy this script
    4. Run!
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


# Deep MLP Classifier
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


def train():
    # Configuration
    EPOCHS = 200
    BATCH_SIZE = 512  # Large batch for GPU
    LR = 1e-3
    DROPOUT = 0.4
    PATIENCE = 20

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

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")

    model = DeepMLPClassifier(n_classes=n_classes, dropout=DROPOUT).to(device)
    print(f"🏗️  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n🚀 Starting training...")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
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

        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        scheduler.step(val_acc)

        print(f"Epoch {epoch:3d}/{EPOCHS} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Gap: {train_acc-val_acc:.2f}%")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, 'best_model.pth')

            print(f"🎉 New best! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹️  Early stopping")
                break

    print(f"\n✅ Training complete!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📈 Improvement over baseline (37%): +{best_val_acc - 37:.2f}%")


if __name__ == "__main__":
    train()
