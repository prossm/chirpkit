#!/usr/bin/env python3
"""
Remove audio files that don't have species names (soundscape, identity unknown, etc.)
to save disk space and avoid confusion
"""

import json
from pathlib import Path

def cleanup_unwanted_audio():
    """Remove audio files that aren't in the metadata"""

    audio_dir = Path('data/raw/xenocanto/audio')
    metadata_file = audio_dir / 'xenocanto_metadata.json'

    # Load metadata to get list of wanted files
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    wanted_files = set(item['filename'] for item in metadata)
    print(f"📋 Found {len(wanted_files)} wanted files in metadata")

    # Find all MP3 files
    all_files = list(audio_dir.glob('*.mp3'))
    print(f"📁 Found {len(all_files)} total audio files")

    # Identify unwanted files
    unwanted = []
    for audio_file in all_files:
        if audio_file.name not in wanted_files and not audio_file.name.startswith('._'):
            unwanted.append(audio_file)

    print(f"🗑️  Found {len(unwanted)} unwanted audio files to delete")

    if not unwanted:
        print("✅ No unwanted files to delete!")
        return

    # Show some examples
    print("📝 Examples of files to delete:")
    for example in unwanted[:5]:
        print(f"   {example.name}")

    if len(unwanted) > 5:
        print(f"   ... and {len(unwanted) - 5} more")

    # Calculate space to free
    total_size = sum(f.stat().st_size for f in unwanted)
    size_gb = total_size / (1024**3)
    print(f"💾 Will free up {size_gb:.1f} GB of disk space")

    # Ask for confirmation
    response = input(f"\n❓ Delete {len(unwanted)} unwanted files? (y/N): ")

    if response.lower() in ['y', 'yes']:
        deleted = 0
        for audio_file in unwanted:
            try:
                audio_file.unlink()
                deleted += 1
            except Exception as e:
                print(f"❌ Error deleting {audio_file.name}: {e}")

        print(f"✅ Deleted {deleted} unwanted files")
        print(f"💾 Freed up {size_gb:.1f} GB of disk space")
    else:
        print("⏭️  Skipped deletion")

if __name__ == "__main__":
    cleanup_unwanted_audio()