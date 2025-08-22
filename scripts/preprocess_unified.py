#!/usr/bin/env python3
"""
Unified preprocessing script for both InsectSound1000 and InsectSet459 datasets
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data.preprocessing import InsectAudioPreprocessor

class UnifiedDatasetProcessor:
    """Handle preprocessing for multiple insect audio datasets"""
    
    def __init__(self, base_data_dir='data/raw'):
        self.base_data_dir = Path(base_data_dir)
        self.preprocessor = InsectAudioPreprocessor()
        
        # Dataset configurations
        self.datasets = {
            'insectsound1000': {
                'data_dir': self.base_data_dir / 'insectsound1000',
                'metadata_file': 'metadata.csv',
                'audio_dir': 'versions/1/InsectSound1000',
                'format': 'insectsound1000'
            },
            'insectset459': {
                'data_dir': self.base_data_dir / 'insectset459',
                'metadata_file': 'InsectSet459_Train_Val_Annotation.csv',
                'audio_dir': 'Train',  # For training data
                'validation_dir': 'Validation',  # For validation data  
                'format': 'insectset459'
            }
        }
    
    def load_metadata(self, dataset_name):
        """Load metadata for specified dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")
        
        config = self.datasets[dataset_name]
        metadata_path = config['data_dir'] / config['metadata_file']
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        df = pd.read_csv(metadata_path)
        
        if config['format'] == 'insectsound1000':
            # InsectSound1000 format: filepath, species
            return self._process_insectsound1000_metadata(df, config)
        elif config['format'] == 'insectset459':
            # InsectSet459 format: different structure
            return self._process_insectset459_metadata(df, config)
    
    def _process_insectsound1000_metadata(self, df, config):
        """Process InsectSound1000 metadata format"""
        print(f"📊 Processing InsectSound1000 metadata: {len(df)} samples")
        
        processed_data = []
        for _, row in df.iterrows():
            # The filepath in metadata.csv is already relative to project root
            processed_data.append({
                'filepath': Path(row['filepath']),  # Use filepath as-is
                'species': row['species'],
                'split': 'train',  # Will be split later
                'dataset': 'insectsound1000'
            })
        
        return pd.DataFrame(processed_data)
    
    def _process_insectset459_metadata(self, df, config):
        """Process InsectSet459 metadata format"""
        print(f"📊 Processing InsectSet459 metadata: {len(df)} samples")
        
        # Map splits based on directory structure
        train_base = config['data_dir'] / config['audio_dir']
        val_base = config['data_dir'] / config['validation_dir'] if 'validation_dir' in config else None
        
        processed_data = []
        for _, row in df.iterrows():
            # InsectSet459 CSV format: file_name, species_name, subset
            filename = row.get('file_name', '')
            species = row.get('species_name', '')
            split = row.get('subset', 'Train')  # Train or Validation
            
            if not filename or not species:
                continue
                
            # Determine file path based on split
            if split.lower() == 'validation' and val_base:
                filepath = val_base / filename
            else:
                filepath = train_base / filename
            
            processed_data.append({
                'filepath': filepath,
                'species': species,
                'split': split.lower(),
                'dataset': 'insectset459'
            })
        
        return pd.DataFrame(processed_data)
    
    def preprocess_dataset(self, dataset_name, output_prefix='', limit=None):
        """Preprocess a specific dataset"""
        print(f"🔄 Preprocessing dataset: {dataset_name}")
        
        # Load metadata
        metadata_df = self.load_metadata(dataset_name)
        
        if limit:
            print(f"⚠️ Limiting to {limit} samples for testing")
            metadata_df = metadata_df.head(limit)
        
        print(f"📁 Found {len(metadata_df)} samples")
        print(f"🦗 Species distribution:")
        print(metadata_df['species'].value_counts().head(10))
        
        # Process audio files
        features = []
        labels = []
        valid_files = []
        
        for idx, row in metadata_df.iterrows():
            audio_path = Path(row['filepath'])
            species = row['species']
            
            try:
                # Check if file exists
                if not audio_path.exists():
                    print(f"⚠️ File not found: {audio_path}")
                    continue
                
                # Load and preprocess audio
                feats = self.preprocessor.load_and_preprocess(audio_path)
                features.append(feats['spectrogram'])
                labels.append(species)
                valid_files.append(str(audio_path))
                
                if (len(features)) % 100 == 0:
                    print(f"✅ Processed {len(features)} files...")
                    
            except Exception as e:
                print(f"❌ Error processing {audio_path}: {e}")
                continue
        
        print(f"✅ Successfully processed {len(features)} out of {len(metadata_df)} files")
        
        if not features:
            print("❌ No valid features extracted!")
            return None
        
        # Convert to numpy arrays
        features_array = np.array(features)
        labels_array = np.array(labels)
        
        # Create output directory
        output_dir = Path('data/processed') / (f"{dataset_name}_{output_prefix}" if output_prefix else dataset_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features and labels
        np.save(output_dir / 'features.npy', features_array)
        np.save(output_dir / 'labels.npy', labels_array)
        
        # Save file list for reference
        pd.DataFrame({'filepath': valid_files, 'species': labels}).to_csv(
            output_dir / 'processed_files.csv', index=False
        )
        
        print(f"💾 Saved features: {features_array.shape}")
        print(f"📁 Output directory: {output_dir}")
        
        return {
            'features': features_array,
            'labels': labels_array,
            'output_dir': output_dir,
            'dataset_name': dataset_name
        }
    
    def create_splits(self, features, labels, output_dir, test_size=0.2, val_size=0.1):
        """Create train/validation/test splits"""
        print(f"🔄 Creating data splits...")
        
        # Create splits directory
        splits_dir = output_dir.parent.parent / 'splits' / output_dir.name
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, stratify=labels, random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for the reduced dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
        )
        
        # Save splits
        np.save(splits_dir / 'X_train.npy', X_train)
        np.save(splits_dir / 'y_train.npy', y_train)
        np.save(splits_dir / 'X_val.npy', X_val)
        np.save(splits_dir / 'y_val.npy', y_val)
        np.save(splits_dir / 'X_test.npy', X_test)
        np.save(splits_dir / 'y_test.npy', y_test)
        
        print(f"✅ Splits saved to: {splits_dir}")
        print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Show species distribution
        unique_species = np.unique(labels)
        print(f"🦗 {len(unique_species)} unique species")
        
        return splits_dir

def main():
    parser = argparse.ArgumentParser(description='Preprocess insect audio datasets')
    parser.add_argument('--dataset', 
                       choices=['insectsound1000', 'insectset459', 'both'], 
                       default='both',
                       help='Dataset to preprocess')
    parser.add_argument('--limit', type=int, help='Limit number of samples for testing')
    parser.add_argument('--no-splits', action='store_true', help='Skip creating train/val/test splits')
    parser.add_argument('--output-prefix', default='', help='Prefix for output directory')
    
    args = parser.parse_args()
    
    processor = UnifiedDatasetProcessor()
    
    if args.dataset == 'both':
        datasets = ['insectsound1000', 'insectset459']
    else:
        datasets = [args.dataset]
    
    for dataset_name in datasets:
        try:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")
            
            result = processor.preprocess_dataset(
                dataset_name, 
                output_prefix=args.output_prefix,
                limit=args.limit
            )
            
            if result and not args.no_splits:
                processor.create_splits(
                    result['features'], 
                    result['labels'], 
                    result['output_dir']
                )
                
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            continue
    
    print(f"\n✅ Preprocessing complete!")

if __name__ == "__main__":
    main()