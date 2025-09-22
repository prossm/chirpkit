#!/usr/bin/env python3
"""
Download Xeno-canto Orthoptera dataset for globally balanced insect classifier
Provides excellent global coverage with high-quality community recordings
"""

import os
import requests
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import urllib.parse

def search_xenocanto_orthoptera(area=None, country=None, quality=None, max_results=None, start_page=1):
    """Search Xeno-canto for Orthoptera recordings"""

    base_url = "https://xeno-canto.org/api/3/recordings"

    # Build query for grasshoppers/Orthoptera
    # Based on API docs, use grp: parameter with proper quoting
    query_parts = ["grp:grasshoppers"]

    if area:
        query_parts.append(f"area:{area}")
    if country:
        query_parts.append(f"cnt:{country}")
    # Handle quality parameter properly - don't pass multiple values at once
    if quality and ',' not in str(quality):
        query_parts.append(f"q:{quality}")
    elif quality:
        # For multiple quality values, we'll need to handle differently
        # For now, just use the first quality rating
        first_quality = str(quality).split(',')[0]
        query_parts.append(f"q:{first_quality}")

    query = "+".join(query_parts)

    print(f"🔍 Searching Xeno-canto: {query}")
    print(f"🔗 Full URL: {base_url}?query={urllib.parse.quote_plus(query)}")
    if start_page > 1:
        print(f"⏭️  Starting from page {start_page} to resume download")

    all_recordings = []
    page = start_page
    
    while True:
        params = {
            'query': query,
            'page': page,
            'key': '1f72bba2148f41a63deb8a32028ef24652f723cb'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            recordings = data.get('recordings', [])
            total_pages = data.get('numPages', 0)
            total_recordings = data.get('numRecordings', 0)
            
            print(f"📊 API Response - Total recordings: {total_recordings}, Pages: {total_pages}")
            
            if not recordings:
                print("🔍 No recordings on this page, stopping search")
                break
                
            all_recordings.extend(recordings)
            
            print(f"📄 Page {page}: Found {len(recordings)} recordings (total: {len(all_recordings)})")
            
            if max_results and len(all_recordings) >= max_results:
                all_recordings = all_recordings[:max_results]
                print(f"🛑 Reached max results limit: {max_results}")
                break
                
            # Check if more pages available using total pages from API response
            if page >= total_pages:
                print(f"📄 Reached last page ({page}/{total_pages})")
                break
                
            page += 1
            time.sleep(0.5)  # Be respectful to the API
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed on page {page}: {e}")
            break
            
    return all_recordings

def download_xenocanto_recordings(recordings, download_dir, force_download=False):
    """Download individual recordings from Xeno-canto"""
    
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading {len(recordings)} recordings to {download_dir}")
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    # Create metadata file
    metadata = []
    
    for recording in tqdm(recordings, desc="Downloading"):
        try:
            xc_id = recording.get('id', 'unknown')
            species = recording.get('en', 'unknown_species').replace(' ', '_')
            subspecies = recording.get('ssp', '')
            country = recording.get('cnt', 'unknown')
            location = recording.get('loc', '')
            quality = recording.get('q', 'unrated')
            length = recording.get('length', '0')
            
            # Create filename: XC{id}_{species}_{country}.mp3
            filename = f"XC{xc_id}_{species}_{country}.mp3"
            local_path = download_dir / filename
            
            # Skip if already exists
            if local_path.exists() and not force_download:
                skipped += 1
                continue
            
            # Get download URL
            file_url = recording.get('file', '')
            if not file_url:
                failed += 1
                continue
                
            # Download the file
            response = requests.get(file_url, timeout=60)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
                
            downloaded += 1
            
            # Add to metadata
            metadata.append({
                'filename': filename,
                'xc_id': xc_id,
                'species': recording.get('en', ''),
                'scientific_name': recording.get('gen', '') + ' ' + recording.get('sp', ''),
                'subspecies': subspecies,
                'country': country,
                'location': location,
                'quality': quality,
                'length': length,
                'recordist': recording.get('rec', ''),
                'date': recording.get('date', ''),
                'type': recording.get('type', ''),
                'url': recording.get('url', ''),
                'file_url': file_url,
                'dataset': 'xenocanto_orthoptera'
            })
            
            # Small delay to be respectful
            time.sleep(0.1)
            
        except Exception as e:
            failed += 1
            if failed <= 5:  # Only show first few errors
                print(f"❌ Failed to download XC{xc_id}: {e}")
    
    # Save metadata
    metadata_file = download_dir / "xenocanto_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Downloaded: {downloaded} files")
    print(f"⏭️  Skipped (already exist): {skipped} files") 
    print(f"❌ Failed: {failed} files")
    print(f"📋 Metadata saved: {metadata_file}")
    
    return downloaded

def download_xenocanto_orthoptera(data_dir="data", area=None, country=None,
                                quality=None, max_results=10000, force_download=False, start_page=1):
    """Main function to download Xeno-canto Orthoptera dataset"""

    # Create directories
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw" / "xenocanto"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("🦗 Downloading Xeno-canto Orthoptera Dataset")
    print("=" * 60)
    print("🌍 Coverage: Global grasshopper and cricket recordings")
    print("🎵 Source: Community-driven recordings from around the world")
    print("📊 Quality: High-quality recordings with metadata")
    print("=" * 60)

    # Search for recordings
    recordings = search_xenocanto_orthoptera(
        area=area,
        country=country,
        quality=quality,
        max_results=max_results,
        start_page=start_page
    )
    
    if not recordings:
        print("❌ No recordings found matching criteria")
        return False
        
    print(f"\n📊 Found {len(recordings)} Orthoptera recordings")
    
    # Get species statistics
    species_set = set()
    countries_set = set()
    for r in recordings:
        if r.get('en'):
            species_set.add(r['en'])
        if r.get('cnt'):
            countries_set.add(r['cnt'])
    
    print(f"📈 Species diversity: {len(species_set)} unique species")
    print(f"🌍 Geographic spread: {len(countries_set)} countries")
    
    # Download recordings
    download_dir = raw_dir / "audio"
    downloaded = download_xenocanto_recordings(recordings, download_dir, force_download)
    
    if downloaded > 0:
        print(f"\n🎉 Xeno-canto Orthoptera download complete!")
        print(f"   🎵 Downloaded: {downloaded} recordings")
        print(f"   🦗 Species: {len(species_set)} unique species")  
        print(f"   🌍 Countries: {len(countries_set)} countries")
        print(f"   📁 Location: {download_dir}")
        return True
    else:
        print("❌ No files downloaded")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Xeno-canto Orthoptera dataset")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument("--area", help="Continental area (africa,america,asia,australia,europe)")
    parser.add_argument("--country", help="Specific country")
    parser.add_argument("--quality", default=None, help="Quality ratings (A,B,C,D,E or combinations)")
    parser.add_argument("--max-results", type=int, default=10000, help="Maximum recordings (default: 10000)")
    parser.add_argument("--start-page", type=int, default=1, help="Start from specific page (for resuming downloads)")
    parser.add_argument("--force", action="store_true", help="Force re-download existing files")

    args = parser.parse_args()

    print("🌍 ChirpKit: Building Globally Balanced Insect Classifier")
    print("📈 Adding global Orthoptera recordings from Xeno-canto community")
    print()

    success = download_xenocanto_orthoptera(
        data_dir=args.data_dir,
        area=args.area,
        country=args.country,
        quality=args.quality,
        max_results=args.max_results,
        force_download=args.force,
        start_page=args.start_page
    )
    
    if success:
        print()
        print("📋 Next steps:")
        print("1. Run preprocessing: python scripts/preprocess_unified.py --dataset xenocanto")
        print("2. Or process all datasets: python scripts/preprocess_unified.py --dataset all")
        print("3. Train balanced model: python scripts/train_unified.py --dataset combined")
        print()
        print("🌍 Updated Dataset Balance:")
        print("   🇪🇺 Europe: InsectSound1000 (~170k samples)")
        print("   🌎 Global: InsectSet459 (~10k samples)")
        print("   🇺🇸 North America: SINA (~265 samples)")
        print("   🌍 Global Community: Xeno-canto (~thousands of samples)")
        print("   ✅ Excellent global representation!")
    else:
        print("❌ Download failed.")
        exit(1)

if __name__ == "__main__":
    main()