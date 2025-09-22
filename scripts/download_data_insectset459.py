#!/usr/bin/env python3
"""
Download the InsectSet459 dataset from Zenodo
"""
import os
import requests
import hashlib
from tqdm import tqdm
import zipfile

def verify_md5(file_path, expected_md5):
    """Verify file MD5 checksum"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_md5

def download_file(url, local_path, expected_md5=None):
    """Download file with progress bar and verification"""
    print(f"📥 Downloading: {os.path.basename(local_path)}")
    print(f"🔗 URL: {url}")
    
    # Check if file already exists and is valid
    if os.path.exists(local_path):
        if expected_md5:
            print("🔍 File exists, verifying checksum...")
            if verify_md5(local_path, expected_md5):
                print("✅ File already downloaded and verified!")
                return True
            else:
                print("❌ Checksum mismatch, re-downloading...")
                os.remove(local_path)
        else:
            print("✅ File already exists (no checksum verification)")
            return True
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f, tqdm(
        desc=os.path.basename(local_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    # Verify checksum if provided
    if expected_md5:
        print("🔍 Verifying checksum...")
        if verify_md5(local_path, expected_md5):
            print("✅ Download verified!")
            return True
        else:
            print("❌ Checksum verification failed!")
            return False
    else:
        print("✅ Download completed!")
        return True

def extract_dataset(zip_path, extract_dir):
    """Extract dataset with progress"""
    print(f"📦 Extracting: {os.path.basename(zip_path)}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        
        with tqdm(total=len(files), desc="Extracting") as pbar:
            for file in files:
                zip_ref.extract(file, extract_dir)
                pbar.update(1)
    
    print(f"✅ Extracted to: {extract_dir}")

def main():
    """Download the InsectSet459 dataset"""
    print("🦗 InsectSet459 Dataset Downloader")
    print("=" * 50)
    
    # Dataset information
    dataset_info = {
        "Train.zip": {
            "url": "https://zenodo.org/records/14056458/files/Train.zip",
            "size": "51.3 GB",
            "md5": "f2ca7e912238df5f8943e7592486c493"
        },
        "Validation.zip": {
            "url": "https://zenodo.org/records/14056458/files/Validation.zip", 
            "size": "16.4 GB",
            "md5": "7b520ff720058ea59646f763c6265fb3"
        },
        "InsectSet459_Train_Val_Annotation.csv": {
            "url": "https://zenodo.org/records/14056458/files/InsectSet459_Train_Val_Annotation.csv",
            "size": "5.5 MB", 
            "md5": "d27f6b95598e49170ca996279135a48f"
        }
    }
    
    # Create data directory
    data_dir = "data/raw/insectset459"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"📁 Download directory: {data_dir}")
    print(f"💾 Total dataset size: ~68 GB")
    print()
    
    # Ask user which files to download
    print("Available files:")
    for i, (filename, info) in enumerate(dataset_info.items(), 1):
        print(f"{i}. {filename} ({info['size']})")
    
    print()
    choice = input("Download options:\n1. All files\n2. Annotation file only (for exploration)\n3. Train + Annotation\n4. Validation + Annotation\nChoice (1-4): ")
    
    files_to_download = []
    if choice == "1":
        files_to_download = list(dataset_info.keys())
    elif choice == "2":
        files_to_download = ["InsectSet459_Train_Val_Annotation.csv"]
    elif choice == "3":
        files_to_download = ["Train.zip", "InsectSet459_Train_Val_Annotation.csv"]
    elif choice == "4":
        files_to_download = ["Validation.zip", "InsectSet459_Train_Val_Annotation.csv"]
    else:
        print("Invalid choice, downloading annotation file only")
        files_to_download = ["InsectSet459_Train_Val_Annotation.csv"]
    
    print(f"\n📥 Will download: {', '.join(files_to_download)}")
    print()
    
    # Download selected files
    downloaded_files = []
    for filename in files_to_download:
        info = dataset_info[filename]
        local_path = os.path.join(data_dir, filename)
        
        try:
            success = download_file(info["url"], local_path, info["md5"])
            if success:
                downloaded_files.append(local_path)
            print()
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            print()
    
    # Ask about extraction
    zip_files = [f for f in downloaded_files if f.endswith('.zip')]
    if zip_files:
        extract = input(f"Extract {len(zip_files)} ZIP file(s)? (y/n): ").lower().startswith('y')
        if extract:
            for zip_path in zip_files:
                try:
                    extract_dir = os.path.join(data_dir, "extracted")
                    os.makedirs(extract_dir, exist_ok=True)
                    extract_dataset(zip_path, extract_dir)
                    print()
                except Exception as e:
                    print(f"❌ Error extracting {zip_path}: {e}")
                    print()
    
    print("🎉 Download process completed!")
    print(f"📁 Files saved to: {data_dir}")
    
    # Show next steps
    if "InsectSet459_Train_Val_Annotation.csv" in [os.path.basename(f) for f in downloaded_files]:
        print("\n📊 Next steps:")
        print("1. Explore the annotation file to understand the 459 species")
        print("2. Plan data preprocessing for the larger dataset")
        print("3. Consider transfer learning from your current 12-species model")

if __name__ == "__main__":
    main()