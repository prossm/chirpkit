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

def search_xenocanto_orthoptera(area=None, country=None, quality=None, start_page=1, existing_files=0):
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
            
            current_total = existing_files + len(all_recordings)
            print(f"📄 Page {page}: Found {len(recordings)} recordings (total: {current_total}/{total_recordings})")
            
                
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

def download_xenocanto_recordings(recordings, download_dir, force_download=False, existing_files=0, total_available=0):
    """Download individual recordings from Xeno-canto"""

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {len(recordings)} recordings to {download_dir}")

    downloaded = 0
    failed = 0
    skipped = 0

    # Load existing metadata if available
    metadata_file = download_dir / "xenocanto_metadata.json"
    existing_metadata = []
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                existing_metadata = json.load(f)
            print(f"📋 Loaded {len(existing_metadata)} existing metadata entries")
        except Exception as e:
            print(f"⚠️  Could not load existing metadata: {e}")

    metadata = []

    # Create progress bar that starts from existing files count
    progress_bar = tqdm(
        recordings,
        desc="Downloading",
        initial=existing_files,
        total=total_available if total_available > 0 else existing_files + len(recordings)
    )

    for recording in progress_bar:
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

            # Skip files without proper species names (soundscape, identity unknown, etc.)
            if (not species or
                species.lower() in ['soundscape', 'identity unknown', 'unknown_species', ''] or
                'soundscape' in species.lower() or
                ('identity' in species.lower() and 'unknown' in species.lower())):
                print(f"⏭️  Skipping non-species recording: {species}")
                skipped += 1
                continue
            
            # Skip if already exists
            if local_path.exists() and not force_download:
                skipped += 1
                # Don't update progress bar for skipped files since they're already counted in initial
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
            # Update progress bar for newly downloaded file
            progress_bar.update(1)
            
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

    # Close progress bar
    progress_bar.close()

    # Combine existing and new metadata
    all_metadata = existing_metadata + metadata

    # Remove duplicates based on filename
    seen_filenames = set()
    unique_metadata = []
    for item in all_metadata:
        filename = item.get('filename', '')
        if filename and filename not in seen_filenames:
            seen_filenames.add(filename)
            unique_metadata.append(item)

    # Save combined metadata
    with open(metadata_file, 'w') as f:
        json.dump(unique_metadata, f, indent=2)

    print(f"📋 Total metadata entries: {len(unique_metadata)} (was {len(existing_metadata)}, added {len(metadata)})"
    
    print(f"✅ Downloaded: {downloaded} files")
    print(f"⏭️  Skipped (already exist): {skipped} files") 
    print(f"❌ Failed: {failed} files")
    print(f"📋 Metadata saved: {metadata_file}")
    
    return downloaded

def download_xenocanto_orthoptera(data_dir="data", area=None, country=None,
                                quality=None, force_download=False, start_page=1):
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

    # Check existing files to show actual progress
    download_dir = raw_dir / "audio"
    download_dir.mkdir(parents=True, exist_ok=True)
    existing_files = len(list(download_dir.glob("*.mp3")))
    if existing_files > 0:
        print(f"📁 Found {existing_files} existing files on disk")

    # Search for recordings
    recordings = search_xenocanto_orthoptera(
        area=area,
        country=country,
        quality=quality,
        start_page=start_page,
        existing_files=existing_files
    )
    
    if not recordings:
        print("❌ No recordings found matching criteria")
        return False
        
    total_available = existing_files + len(recordings)
    print(f"\n📊 Found {len(recordings)} new Orthoptera recordings")
    print(f"📊 Total metadata available: {total_available} recordings")

    # Get species statistics
    species_set = set()
    countries_set = set()
    for r in recordings:
        if r.get('en'):
            species_set.add(r['en'])
        if r.get('cnt'):
            countries_set.add(r['cnt'])

    print(f"📈 Species diversity: {len(species_set)} unique species (in new batch)")
    print(f"🌍 Geographic spread: {len(countries_set)} countries (in new batch)")
    
    # Download recordings
    downloaded = download_xenocanto_recordings(
        recordings,
        download_dir,
        force_download,
        existing_files=existing_files,
        total_available=total_available
    )
    
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