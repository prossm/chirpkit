import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
# TensorBoard
from torch.utils.tensorboard import SummaryWriter
import os
import json
from datetime import datetime
# Label encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.augmentation import InsectAudioAugmenter, AugmentedDataset


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

# Prepare label encoder and datasets
train_features_path = 'data/splits/X_train.npy'
train_labels_path = 'data/splits/y_train.npy'
val_features_path = 'data/splits/X_val.npy'
val_labels_path = 'data/splits/y_val.npy'

# Fit encoder on train labels, save for later use
train_labels_raw = np.load(train_labels_path, mmap_mode='r')
label_encoder = LabelEncoder()
label_encoder.fit(train_labels_raw)
import joblib
os.makedirs('models/trained', exist_ok=True)
joblib.dump(label_encoder, 'models/trained/label_encoder.joblib')

train_dataset_base = NpyDataset(train_features_path, train_labels_path, label_encoder=label_encoder)
val_dataset = NpyDataset(val_features_path, val_labels_path, label_encoder=label_encoder)

# Disable augmentation initially to focus on architecture
# augmenter = InsectAudioAugmenter()
# train_dataset = AugmentedDataset(train_dataset_base, augmenter, augmentation_prob=0.3)
train_dataset = train_dataset_base  # No augmentation for now

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
print("DataLoaders created. Starting training loop...")

# Create checkpoint directory
checkpoint_dir = 'models/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# TensorBoard log dir
log_dir = os.path.join('runs', 'cnn_lstm_experiment')
writer = SummaryWriter(log_dir=log_dir)

# Convert to torch tensors


# Model
classes = label_encoder.classes_
n_classes = len(classes)
model = SimpleCNNLSTMInsectClassifier(n_classes=n_classes)

# Better weight initialization for complex architecture
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

model.apply(init_weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Calculate class weights to handle imbalance
train_labels_for_weights = np.load(train_labels_path, mmap_mode='r')
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels_for_weights),
    y=train_labels_for_weights
)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# Calculate class weights but don't use them yet
print(f"📊 Class weights calculated (but NOT using them):")
for i, (cls, weight) in enumerate(zip(label_encoder.classes_, class_weights)):
    print(f"  {cls}: {weight:.3f}")

print(f"🧪 TESTING: Using standard CrossEntropyLoss (no class weights)")

# Simpler approach - higher LR scaled for batch size 64, no class weights
optimizer = optim.Adam(model.parameters(), lr=4.24e-3, weight_decay=1e-4)  
criterion = nn.CrossEntropyLoss()  # No class weights

# Simple learning rate scheduler that works
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

import signal
import sys

# Graceful interrupt handling
def signal_handler(sig, frame):
    print(f"\n🛑 Training interrupted! Saving checkpoint...")
    global model, optimizer, scheduler, epoch, best_val_acc, patience_counter
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'patience_counter': patience_counter,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Emergency checkpoint saved to {checkpoint_path}")
        print("🔄 You can resume training by running this script again")
    except Exception as e:
        print(f"❌ Failed to save emergency checkpoint: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Resume training functionality
start_epoch = 1
best_val_acc = 0.0
checkpoint_path = 'models/checkpoints/latest_checkpoint.pth'

# Early stopping variables
patience = 15
patience_counter = 0
min_delta = 1e-4

# Check if we can resume from checkpoint
if os.path.exists(checkpoint_path):
    print("Found checkpoint, resuming training...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    patience_counter = checkpoint.get('patience_counter', 0)
    print(f"Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
    print(f"Patience counter: {patience_counter}/{patience}")
elif os.path.exists('models/trained/cnn_lstm_best.pth'):
    print("🆕 Starting fresh with enhanced architecture (ignoring old incompatible model)")
    print("💡 Tip: Delete 'models/trained/cnn_lstm_best.pth' to skip this message")
else:
    print("Starting fresh training...")

# Training loop - now with more epochs and resume capability
max_epochs = 100
for epoch in range(start_epoch, max_epochs + 1):
    print(f"Starting epoch {epoch}...")
    model.train()
    total_loss = 0
    total_train = 0
    correct_train = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += y_batch.size(0)
        correct_train += (predicted == y_batch).sum().item()
    avg_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train
    print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    print(f"Epoch {epoch} - Train Accuracy: {train_acc:.4f}")
    print(f"💾 Logging to TensorBoard: epoch {epoch}")
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    # Validation with detailed metrics
    model.eval()
    val_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    val_f1 = f1_score(all_targets, all_predictions, average='weighted')
    val_precision = precision_score(all_targets, all_predictions, average='weighted')
    val_recall = recall_score(all_targets, all_predictions, average='weighted')
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Epoch {epoch} - Val F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
    print(f"Learning Rate: {current_lr:.2e}")
    
    # Log all metrics to TensorBoard
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('F1/val', val_f1, epoch)
    writer.add_scalar('Precision/val', val_precision, epoch)
    writer.add_scalar('Recall/val', val_recall, epoch)
    writer.add_scalar('Learning_Rate', current_lr, epoch)
    # Early stopping check
    if val_acc > best_val_acc + min_delta:
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Step the learning rate scheduler
    scheduler.step()
    
    # Save checkpoint after each epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'val_acc': val_acc,
        'train_loss': avg_loss,
        'train_acc': train_acc,
        'patience_counter': patience_counter,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model and training info
    if epoch == start_epoch or val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/trained/cnn_lstm_best.pth')
        
        # Save training information
        training_info = {
            'last_epoch': epoch,
            'best_val_acc': best_val_acc,
            'best_epoch': epoch,
            'total_epochs_trained': epoch,
            'last_updated': datetime.now().isoformat()
        }
        with open('models/trained/training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"🎉 New best model saved! Val accuracy: {val_acc:.4f}")
        
        # Save detailed classification report for best model
        if epoch % 10 == 0 or epoch == max_epochs:  # Every 10 epochs or final epoch
            species_names = label_encoder.classes_
            report = classification_report(all_targets, all_predictions, 
                                         target_names=species_names, output_dict=True)
            
            # Save classification report
            report_path = f'models/trained/classification_report_epoch_{epoch}.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📊 Classification report saved to {report_path}")
    
    print(f"Epoch {epoch} completed. Checkpoint saved. Patience: {patience_counter}/{patience}")
    print("=" * 60)
    
    # Early stopping check
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
        break

print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
print("You can resume training anytime by running this script again.")
writer.close()
