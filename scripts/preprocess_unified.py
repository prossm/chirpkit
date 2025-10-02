#!/usr/bin/env python3
"""
Unified preprocessing script for InsectSound1000, InsectSet459, and SINA datasets
Creates globally balanced insect classifier with European, global, and North American species
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# Add src to path (use insert to prioritize local modules)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data.preprocessing import InsectAudioPreprocessor

class UnifiedDatasetProcessor:
    """Handle preprocessing for multiple insect audio datasets"""

    def __init__(self, base_data_dir='data/raw', use_enhanced=True):
        self.base_data_dir = Path(base_data_dir)
        # Use enhanced features for 80% target accuracy
        # 256 mel bins, 22050 Hz, 40 MFCCs with deltas
        if use_enhanced:
            self.preprocessor = InsectAudioPreprocessor(
                target_sr=22050,      # Higher sample rate for high-freq insect calls
                duration=2.5,
                n_fft=4096,           # 4x better frequency resolution
                hop_length=256,       # Better temporal resolution
                n_mels=256,           # 2x frequency resolution
                n_mfcc=40,            # Richer timbre features
                use_enhanced=True
            )
        else:
            # Fallback to basic features
            self.preprocessor = InsectAudioPreprocessor()
        
        # Dataset configurations
        self.datasets = {
            'insectsound1000': {
                'data_dir': self.base_data_dir / 'insectsound1000',
                'metadata_file': 'metadata.csv',
                'audio_dir': 'versions/1/InsectSound1000',
                'format': 'insectsound1000',
                'region': 'Europe'
            },
            'insectset459': {
                'data_dir': self.base_data_dir / 'insectset459',
                'metadata_file': 'InsectSet459_Train_Val_Annotation.csv',
                'audio_dir': 'Train',  # For training data
                'validation_dir': 'Validation',  # For validation data  
                'format': 'insectset459',
                'region': 'Global'
            },
            'sina': {
                'data_dir': self.base_data_dir / 'sina',
                'audio_dir': 'audio',  # Extracted from ZIP
                'format': 'sina',
                'region': 'North America'
            },
            'xenocanto': {
                'data_dir': self.base_data_dir / 'xenocanto',
                'metadata_file': 'xenocanto_metadata.json',
                'audio_dir': 'audio',
                'format': 'xenocanto',
                'region': 'Global Community'
            }
        }
    
    def load_metadata(self, dataset_name):
        """Load metadata for specified dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")

        config = self.datasets[dataset_name]

        if config['format'] == 'sina':
            # SINA format: use metadata from SINA text files
            return self._process_sina_metadata(config)
        elif config['format'] == 'xenocanto':
            # Xeno-canto format: JSON metadata file
            return self._process_xenocanto_metadata(config)
        else:
            # CSV-based formats
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
        """Process InsectSound1000 metadata format with balanced subsampling"""
        print(f"📊 Processing InsectSound1000 metadata: {len(df)} samples")
        
        # IMPORTANT: Subsample to 1000 files to remove European bias
        # Use stratified sampling to maintain species diversity
        subsample_size = 1000
        
        if len(df) > subsample_size:
            print(f"🎯 Subsampling {len(df)} -> {subsample_size} files for geographic balance")
            
            # Group by species and sample proportionally
            species_counts = df['species'].value_counts()
            sampled_dfs = []
            
            for species, count in species_counts.items():
                species_df = df[df['species'] == species]
                # Calculate proportional sample size
                n_samples = max(1, int((count / len(df)) * subsample_size))
                n_samples = min(n_samples, len(species_df))  # Don't exceed available
                
                sampled = species_df.sample(n=n_samples, random_state=42)
                sampled_dfs.append(sampled)
                print(f"  {species}: {count} -> {n_samples} samples")
            
            df = pd.concat(sampled_dfs, ignore_index=True)
            print(f"✅ Subsampled to {len(df)} files maintaining species diversity")
        
        processed_data = []
        for _, row in df.iterrows():
            # The filepath in metadata.csv is already relative to project root
            processed_data.append({
                'filepath': Path(row['filepath']),  # Use filepath as-is
                'species': row['species'],
                'split': 'train',  # Will be split later
                'dataset': 'insectsound1000',
                'region': 'Europe'
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

            # Normalize species name: convert underscores to spaces
            species = species.replace('_', ' ')

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

        # Filter out species with only 1 sample (can't be split)
        import pandas as pd
        from collections import Counter

        label_counts = Counter(labels)
        single_sample_species = [label for label, count in label_counts.items() if count == 1]

        if single_sample_species:
            print(f"⚠️  Found {len(single_sample_species)} species with only 1 sample")
            print(f"🗑️  Removing single-sample species for proper train/test splitting")

            # Create mask to filter out single-sample species
            mask = [label not in single_sample_species for label in labels]
            features = features[mask]
            labels = labels[mask]

            print(f"📊 After filtering: {len(features)} samples, {len(set(labels))} species")

        # Create splits directory
        splits_dir = output_dir.parent.parent / 'splits' / output_dir.name
        splits_dir.mkdir(parents=True, exist_ok=True)

        # Check if we have enough samples for stratified splitting
        min_class_count = min(Counter(labels).values())
        test_samples = int(len(features) * test_size)
        unique_classes = len(set(labels))

        if test_samples < unique_classes or min_class_count < 2:
            print(f"⚠️  Cannot use stratified splitting:")
            print(f"   Test samples: {test_samples}, Unique classes: {unique_classes}")
            print(f"   Min class count: {min_class_count}")
            print(f"🔄 Using random splitting instead")
            # Use random split for datasets with insufficient samples per class
            X_temp, X_test, y_temp, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=42
            )
        else:
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                features, labels, test_size=test_size, stratify=labels, random_state=42
            )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for the reduced dataset
        min_class_count_temp = min(Counter(y_temp).values())
        if len(X_temp) < 30 or min_class_count_temp < 2:
            print(f"🔄 Using random split for train/val (min class: {min_class_count_temp})")
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42
            )
        else:
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
    
    def _process_sina_metadata(self, config):
        """Process SINA metadata from text files"""
        print("📊 Processing SINA metadata from text files")
        
        # Load SINA media.txt file
        media_file = config['data_dir'] / 'audio' / 'media.txt'
        
        if not media_file.exists():
            raise FileNotFoundError(f"SINA media.txt not found: {media_file}")
        
        import pandas as pd
        # Read tab-separated media.txt
        df = pd.read_csv(media_file, sep='\t')
        
        processed_data = []
        wav_dir = config['data_dir'] / 'wav_files'
        
        for _, row in df.iterrows():
            media_id = row.get('MediaID', f'sina_{len(processed_data)}')
            species = row.get('TaxonID', 'unknown_species')
            
            # Look for corresponding audio file
            safe_species = species.replace(' ', '_').replace('/', '_')
            filename = f"{media_id}_{safe_species}.wav"
            filepath = wav_dir / filename
            
            if filepath.exists():
                processed_data.append({
                    'filepath': filepath,
                    'species': species,
                    'split': 'train',
                    'dataset': 'sina',
                    'region': 'North America'
                })
        
        print(f"✅ Processed {len(processed_data)} SINA recordings")
        return pd.DataFrame(processed_data)
    
    def _process_xenocanto_metadata(self, config):
        """Process Xeno-canto JSON metadata with species name mapping"""
        print("📊 Processing Xeno-canto metadata from JSON")

        metadata_file = config['data_dir'] / config['audio_dir'] / config['metadata_file']

        if not metadata_file.exists():
            raise FileNotFoundError(f"Xeno-canto metadata not found: {metadata_file}")

        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Load species mapping
        mapping_file = Path('data/xenocanto_species_mapping.json')
        species_mapping = {}
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
                # Extract just the species -> scientific_name mapping
                species_mapping = {
                    species: data['scientific_name']
                    for species, data in mapping_data.items()
                    if not data['scientific_name'].startswith('UNMAPPED_')
                }
            print(f"📋 Loaded mapping for {len(species_mapping)} species")
        else:
            print("⚠️  No species mapping file found - using original names")

        processed_data = []
        audio_dir = config['data_dir'] / config['audio_dir']
        mapped_count = 0

        for item in metadata:
            filename = item.get('filename', '')
            original_species = item.get('scientific_name', 'unknown_species')

            if not original_species or original_species == 'unknown_species':
                original_species = item.get('species', 'unknown_species')

            # Apply species mapping if available
            if original_species in species_mapping:
                species = species_mapping[original_species]
                mapped_count += 1
            else:
                species = original_species

            filepath = audio_dir / filename

            if filepath.exists():
                processed_data.append({
                    'filepath': filepath,
                    'species': species,
                    'split': 'train',
                    'dataset': 'xenocanto',
                    'region': 'Global Community',
                    'country': item.get('country', 'Unknown'),
                    'quality': item.get('quality', 'Unknown')
                })

        print(f"✅ Processed {len(processed_data)} Xeno-canto recordings")
        print(f"🔄 Applied mapping to {mapped_count} species names")
        return pd.DataFrame(processed_data)

def main():
    parser = argparse.ArgumentParser(description='Preprocess insect audio datasets')
    parser.add_argument('--dataset', 
                       choices=['insectsound1000', 'insectset459', 'sina', 'xenocanto', 'all'], 
                       default='all',
                       help='Dataset to preprocess')
    parser.add_argument('--limit', type=int, help='Limit number of samples for testing')
    parser.add_argument('--no-splits', action='store_true', help='Skip creating train/val/test splits')
    parser.add_argument('--output-prefix', default='', help='Prefix for output directory')
    
    args = parser.parse_args()
    
    processor = UnifiedDatasetProcessor()
    
    if args.dataset == 'all':
        datasets = ['insectsound1000', 'insectset459', 'sina', 'xenocanto']
    elif args.dataset == 'both':
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