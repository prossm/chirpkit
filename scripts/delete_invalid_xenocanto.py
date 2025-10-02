#!/usr/bin/env python3
"""
Delete all invalid Xeno-canto audio files (soundscape, identity unknown, empty species, ._ files)
Keeps only files with actual species names to save disk space
"""

import json
import re
from pathlib import Path

def is_valid_species_file(filename):
    """Check if filename represents a valid species recording"""
    # Skip macOS resource fork files (will be handled by dot_clean later)
    if filename.startswith('._'):
        return True  # Don't delete these - dot_clean will handle them

    # Must be MP3
    if not filename.endswith('.mp3'):
        return True  # Don't delete non-MP3 files

    # Try to parse filename
    match = re.match(r'XC(\d+)_(.+)_([^_]+)\.mp3$', filename)
    if not match:
        return False  # Delete malformed filenames

    xc_id, middle_part, country = match.groups()

    # Reject files without proper species names
    if (middle_part == '' or
        middle_part.lower() in ['soundscape', 'identity unknown'] or
        'soundscape' in middle_part.lower() or
        ('identity' in middle_part.lower() and 'unknown' in middle_part.lower())):
        return False  # Delete these invalid species files

    return True  # Keep valid species files

def delete_invalid_files():
    """Delete all invalid Xeno-canto files"""

    audio_dir = Path('data/raw/xenocanto/audio')

    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        return

    # Find all files in the audio directory
    all_files = list(audio_dir.glob('*'))
    print(f"📁 Found {len(all_files)} total files")

    # Separate valid and invalid files
    valid_files = []
    invalid_files = []

    for file_path in all_files:
        if file_path.is_file():
            if is_valid_species_file(file_path.name):
                valid_files.append(file_path)
            else:
                invalid_files.append(file_path)

    print(f"✅ Valid species files: {len(valid_files)}")
    print(f"❌ Invalid files to delete: {len(invalid_files)}")

    if not invalid_files:
        print("🎉 No invalid files found! Directory is already clean.")
        return

    # Calculate space to free
    total_size = 0
    for file_path in invalid_files:
        try:
            total_size += file_path.stat().st_size
        except:
            pass

    size_gb = total_size / (1024**3)
    print(f"💾 Will free up {size_gb:.1f} GB of disk space")

    # Show examples of what will be deleted
    print(f"\n📝 Examples of files to delete:")

    # Group by type for better understanding
    examples = {
        'Soundscape recordings': [],
        'Identity unknown': [],
        'Empty species (__*)': [],
        'Other invalid': []
    }

    for file_path in invalid_files[:20]:  # Show first 20 as examples
        filename = file_path.name
        if 'soundscape' in filename.lower():
            examples['Soundscape recordings'].append(filename)
        elif 'identity' in filename.lower() and 'unknown' in filename.lower():
            examples['Identity unknown'].append(filename)
        elif '__' in filename:
            examples['Empty species (__*)'].append(filename)
        else:
            examples['Other invalid'].append(filename)

    for category, files in examples.items():
        if files:
            print(f"\n   {category}:")
            for filename in files[:3]:  # Show max 3 examples per category
                print(f"     {filename}")
            if len(files) > 3:
                print(f"     ... and {len(files) - 3} more in this category")

    # Confirm deletion
    print(f"\n⚠️  This will permanently delete {len(invalid_files)} files ({size_gb:.1f} GB)")
    response = input("❓ Are you sure you want to delete these files? (type 'DELETE' to confirm): ")

    if response != 'DELETE':
        print("⏭️  Deletion cancelled")
        return

    # Delete files
    deleted_count = 0
    deleted_size = 0

    print(f"\n🗑️  Deleting invalid files...")

    for file_path in invalid_files:
        try:
            file_size = file_path.stat().st_size
            file_path.unlink()
            deleted_count += 1
            deleted_size += file_size

            if deleted_count % 1000 == 0:
                print(f"   Deleted {deleted_count}/{len(invalid_files)} files...")

        except Exception as e:
            print(f"❌ Error deleting {file_path.name}: {e}")

    final_size_gb = deleted_size / (1024**3)

    print(f"\n✅ Deletion complete!")
    print(f"🗑️  Deleted: {deleted_count} files")
    print(f"💾 Space freed: {final_size_gb:.1f} GB")
    print(f"📁 Remaining files: {len(valid_files)}")

    # Verify the cleanup
    remaining_files = list(audio_dir.glob('*.mp3'))
    print(f"🔍 Verification: {len(remaining_files)} MP3 files remain")

if __name__ == "__main__":
    delete_invalid_files()