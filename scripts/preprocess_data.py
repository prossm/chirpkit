import os
from pathlib import Path
import numpy as np
import pandas as pd
from data.preprocessing import InsectAudioPreprocessor

RAW_DATA_DIR = Path('data/raw/InsectSound1000')
PROCESSED_DATA_DIR = Path('data/processed')
SPLITS_DIR = Path('data/splits')
METADATA_PATH = Path('data/metadata/metadata.csv')

# Ensure output directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

preprocessor = InsectAudioPreprocessor()

# Load metadata (assumes a CSV with columns: filepath,species)
metadata = pd.read_csv(METADATA_PATH)

features = []
labels = []

for idx, row in metadata.iterrows():
    audio_path = Path(row['filepath'])
    label = row['species']
    try:
        feats = preprocessor.load_and_preprocess(audio_path)
        features.append(feats['spectrogram'])
        labels.append(label)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1} files...")

# Convert to numpy arrays and save
np.save(PROCESSED_DATA_DIR / 'features.npy', np.array(features))
np.save(PROCESSED_DATA_DIR / 'labels.npy', np.array(labels))

# Optionally, create train/val/test splits
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

np.save(SPLITS_DIR / 'X_train.npy', np.array(X_train))
np.save(SPLITS_DIR / 'y_train.npy', np.array(y_train))
np.save(SPLITS_DIR / 'X_val.npy', np.array(X_val))
np.save(SPLITS_DIR / 'y_val.npy', np.array(y_val))
np.save(SPLITS_DIR / 'X_test.npy', np.array(X_test))
np.save(SPLITS_DIR / 'y_test.npy', np.array(y_test))
print('Preprocessing complete. Features and splits saved.')
