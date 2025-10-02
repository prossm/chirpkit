#!/usr/bin/env python3
"""
Rebuild complete Xeno-canto metadata by extracting information from filenames
This script creates metadata for files that were downloaded but missing metadata entries
"""

import json
import re
from pathlib import Path
import argparse

def parse_xenocanto_filename(filename):
    """
    Parse Xeno-canto filename to extract metadata
    Only accepts files with actual species names: XC{id}_{species}_{country}.mp3
    Rejects: Soundscape, Identity unknown, empty species, etc.
    """
    # Try main format: XC{id}_{species}_{country}.mp3
    match = re.match(r'XC(\d+)_(.+)_([^_]+)\.mp3$', filename)
    if not match:
        return None

    xc_id, middle_part, country = match.groups()

    # Reject files without proper species names
    if (middle_part == '' or
        middle_part.lower() in ['soundscape', 'identity unknown'] or
        'soundscape' in middle_part.lower() or
        'identity' in middle_part.lower() and 'unknown' in middle_part.lower()):
        return None

    # Only accept files with actual species names
    species = middle_part.replace('_', ' ')

    return {
        'filename': filename,
        'xc_id': xc_id,
        'species': species,
        'scientific_name': species,  # Use species as scientific name for now
        'subspecies': '',
        'country': country,
        'location': '',
        'quality': 'Unknown',
        'length': '0',
        'recordist': '',
        'date': '',
        'type': 'species',
        'url': f'https://xeno-canto.org/{xc_id}',
        'file_url': '',
        'dataset': 'xenocanto_orthoptera'
    }

def rebuild_metadata(audio_dir, force_rebuild=False):
    """Rebuild metadata for all audio files"""

    audio_dir = Path(audio_dir)
    metadata_file = audio_dir / 'xenocanto_metadata.json'

    # Load existing metadata
    existing_metadata = {}
    if metadata_file.exists() and not force_rebuild:
        try:
            with open(metadata_file, 'r') as f:
                existing_data = json.load(f)

            for item in existing_data:
                filename = item.get('filename', '')
                if filename:
                    existing_metadata[filename] = item

            print(f"📋 Loaded {len(existing_metadata)} existing metadata entries")
        except Exception as e:
            print(f"⚠️  Error loading existing metadata: {e}")

    # Find all MP3 files
    mp3_files = list(audio_dir.glob('*.mp3'))
    print(f"🔍 Found {len(mp3_files)} audio files")

    # Process each file
    all_metadata = []
    new_entries = 0

    for mp3_file in mp3_files:
        filename = mp3_file.name

        # Skip if we already have metadata for this file
        if filename in existing_metadata:
            all_metadata.append(existing_metadata[filename])
        else:
            # Parse filename to create metadata
            metadata = parse_xenocanto_filename(filename)
            if metadata:
                all_metadata.append(metadata)
                new_entries += 1
            else:
                print(f"⚠️  Could not parse filename: {filename}")

    # Save rebuilt metadata
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"✅ Rebuilt metadata complete!")
    print(f"📊 Total entries: {len(all_metadata)}")
    print(f"🆕 New entries created: {new_entries}")
    print(f"♻️  Existing entries preserved: {len(existing_metadata)}")
    print(f"📁 Saved to: {metadata_file}")

    # Show species distribution
    species_count = {}
    for item in all_metadata:
        species = item.get('scientific_name', 'unknown')
        species_count[species] = species_count.get(species, 0) + 1

    print(f"🦗 Species found: {len(species_count)} unique species")

    # Show top 10 species
    top_species = sorted(species_count.items(), key=lambda x: x[1], reverse=True)[:10]
    print("📈 Top 10 species:")
    for species, count in top_species:
        print(f"   {species}: {count}")

    return len(all_metadata)

def main():
    parser = argparse.ArgumentParser(description="Rebuild Xeno-canto metadata from filenames")
    parser.add_argument("--audio-dir", default="data/raw/xenocanto/audio", help="Audio directory path")
    parser.add_argument("--force", action="store_true", help="Force complete rebuild (ignore existing metadata)")

    args = parser.parse_args()

    print("🔧 Rebuilding Xeno-canto Metadata")
    print("=" * 50)

    total_entries = rebuild_metadata(args.audio_dir, args.force)

    print("=" * 50)
    print(f"🎉 Complete! {total_entries} metadata entries created.")
    print("📋 Now you can run preprocessing with complete metadata coverage.")

if __name__ == "__main__":
    main()