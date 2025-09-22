#!/usr/bin/env python3
"""
Download SINA (Singing Insects of North America) dataset from Zenodo
Provides North American species to balance European bias from InsectSound1000
"""

import os
import requests
import zipfile
from pathlib import Path
import argparse
from tqdm import tqdm

def download_file(url, local_path, description="Downloading"):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as file, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

def extract_zip(zip_path, extract_to):
    """Extract ZIP file with progress"""
    print(f"📦 Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Extraction complete")

def download_sina_dataset(data_dir="data", force_download=False):
    """Download and extract SINA dataset"""
    
    # Create directories
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw" / "sina"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # SINA audio dataset URL from Zenodo
    sina_url = "https://zenodo.org/records/13312646/files/archive.zip?download=1"
    sina_zip_path = raw_dir / "archive.zip"
    sina_extract_dir = raw_dir / "audio"
    
    print("🦗 Downloading SINA (Singing Insects of North America) Dataset")
    print("=" * 60)
    print(f"📍 Source: {sina_url}")
    print(f"💾 Destination: {raw_dir}")
    print(f"🌍 Coverage: North American crickets, katydids, cicadas")
    print(f"🎵 Content: ~6,460 species-specific songs")
    print("=" * 60)
    
    # Download audio files
    if not sina_zip_path.exists() or force_download:
        print("📥 Downloading SINA audio dataset...")
        try:
            download_file(sina_url, sina_zip_path, "SINA Audio")
            print(f"✅ Downloaded: {sina_zip_path}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            print("💡 Note: You may need to download manually from:")
            print(f"   {sina_url}")
            return False
    else:
        print(f"📁 Found existing file: {sina_zip_path}")
    
    # Extract audio files
    if not sina_extract_dir.exists() or force_download:
        if sina_zip_path.exists():
            extract_zip(sina_zip_path, sina_extract_dir)
        else:
            print("❌ ZIP file not found, cannot extract")
            return False
    else:
        print(f"📁 Found existing extraction: {sina_extract_dir}")
    
    # Process metadata and download audio files
    if sina_extract_dir.exists():
        media_file = sina_extract_dir / "media.txt"
        taxa_file = sina_extract_dir / "taxa.txt"
        
        if media_file.exists() and taxa_file.exists():
            print("📋 Found SINA metadata files")
            
            # Download individual audio files from URLs
            audio_dir = raw_dir / "wav_files"
            audio_dir.mkdir(exist_ok=True)
            
            downloaded_count = download_sina_audio_files(media_file, audio_dir, force_download)
            
            print("📊 Dataset summary:")
            print(f"   🎵 Audio files: {downloaded_count} downloaded")
            print(f"   📁 Metadata: {sina_extract_dir}")
            print(f"   📁 Audio files: {audio_dir}")
            print("   🌍 Geographic coverage: North America (balances European bias)")
            print("   🦗 Species types: Crickets, katydids")
            print(f"   📊 Species count: ~200+ North American species")
            return downloaded_count > 0
        else:
            print("❌ Metadata files not found in extraction")
            return False
    else:
        print("❌ Extraction directory not found")
        return False

def download_sina_audio_files(media_file, audio_dir, force_download=False):
    """Download individual WAV files from SINA media.txt URLs"""
    import pandas as pd
    
    print("🎵 Processing SINA audio file URLs...")
    
    # Read media.txt with tab separator
    try:
        df = pd.read_csv(media_file, sep='\t')
        print(f"📊 Found {len(df)} audio records in media.txt")
    except Exception as e:
        print(f"❌ Error reading media.txt: {e}")
        return 0
    
    downloaded = 0
    failed = 0
    
    for idx, row in df.iterrows():
        if idx % 20 == 0:  # Progress update every 20 files
            print(f"📥 Progress: {idx}/{len(df)} processed...")
            
        media_id = row.get('MediaID', f'sina_{idx}')
        species = row.get('TaxonID', 'unknown_species')
        audio_url = row.get('AccessURI', '')
        
        if not audio_url or not audio_url.startswith('http'):
            continue
            
        # Create filename: MediaID_species.wav
        safe_species = species.replace(' ', '_').replace('/', '_')
        filename = f"{media_id}_{safe_species}.wav"
        local_path = audio_dir / filename
        
        if local_path.exists() and not force_download:
            downloaded += 1
            continue
            
        try:
            response = requests.get(audio_url, timeout=30)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            downloaded += 1
            
        except Exception as e:
            failed += 1
            if failed <= 5:  # Only show first few errors
                print(f"❌ Failed to download {audio_url}: {e}")
    
    print(f"✅ Downloaded {downloaded} files, {failed} failed")
    return downloaded

def main():
    parser = argparse.ArgumentParser(description="Download SINA dataset for balanced global coverage")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    
    args = parser.parse_args()
    
    print("🌍 ChirpKit: Building Globally Balanced Insect Classifier")
    print("📈 Adding North American species to counter European bias")
    print()
    
    success = download_sina_dataset(args.data_dir, args.force)
    
    if success:
        print()
        print("🎉 SINA dataset download complete!")
        print()
        print("📋 Next steps:")
        print("1. Run preprocessing: python scripts/preprocess_unified.py --dataset sina")
        print("2. Or process all datasets: python scripts/preprocess_unified.py --dataset all")
        print("3. Train balanced model: python scripts/train_unified.py --dataset combined")
        print()
        print("🌍 Dataset Balance Summary:")
        print("   🇪🇺 Europe: InsectSound1000 (~170k samples)")
        print("   🌎 Global: InsectSet459 (~10k samples)")  
        print("   🇺🇸 North America: SINA (~6.5k samples)")
        print("   ✅ Much better geographic representation!")
    else:
        print("❌ Download failed. Please check network connection or download manually.")
        exit(1)

if __name__ == "__main__":
    main()