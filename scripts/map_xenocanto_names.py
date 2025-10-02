#!/usr/bin/env python3
"""
Map Xeno-canto common names to scientific names for standardization
Creates a comprehensive mapping to identify species overlaps across datasets
"""

import pandas as pd
import requests
import json
import time
from pathlib import Path
import re
from collections import Counter

def load_xenocanto_species():
    """Load all Xeno-canto species and their frequencies from raw metadata"""
    import json

    # Read from raw metadata instead of processed files
    metadata_file = Path('data/raw/xenocanto/audio/xenocanto_metadata.json')
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Extract species names (common names from xenocanto)
    species_list = []
    for item in metadata:
        species = item.get('scientific_name', '')
        if not species:
            species = item.get('species', '')
        if species:
            species_list.append(species)

    species_counts = pd.Series(species_list).value_counts()
    return species_counts

def create_orthoptera_mapping():
    """Create comprehensive mapping of Orthoptera common names to scientific names"""

    # Start with known mappings - focusing on most common species first
    mapping = {
        # Most frequent species (>100 recordings each)
        'Great Green Bush-cricket': 'Tettigonia viridissima',
        'Mediterranean Katydid': 'Tettigonia viridissima',  # Same species
        'Greek Predatory Bush-cricket': 'Saga hellenica',
        'Schmidt\'s Marbled Bush-cricket': 'Barbitistes constrictus',
        'Limnos Bush-cricket': 'Poecilimon zimmeri',
        'La Greca\'s Slender Bush-cricket': 'Tessellana lagrecai',
        'Field Cricket': 'Gryllus campestris',
        'Arabian Sickle Bush-cricke': 'Saga ephippigera',  # Note: typo in original
        'Upland Green Bush-cricket': 'Tettigonia cantans',
        'Black Saw Bush-cricket': 'Barbitistes serricauda',

        # Common house/field crickets
        'House Cricket': 'Acheta domesticus',
        'House cricket': 'Acheta domesticus',
        'Indian House Cricket': 'Acheta domesticus',
        'Wood Cricket': 'Nemobius sylvestris',
        'Two-spotted Tree Cricket': 'Neoxabea bipunctata',
        'Fall Field Cricket': 'Gryllus pennsylvanicus',
        'Spring Field Cricket': 'Gryllus veletis',

        # Bush-crickets and Katydids
        'Phaneroptera nana': 'Phaneroptera nana',  # Already scientific
        'Noble Bright Bush-cricket': 'Poecilimon nobilis',
        'Sickle-bearing Bush-cricket': 'Phaneroptera falcata',
        'Oak Bush-cricket': 'Meconema thalassinum',
        'Large Conehead': 'Ruspolia nitidula',
        'Bog Bush-cricket': 'Metrioptera brachyptera',
        'Heath Bushcricket': 'Gampsocleis glabra',
        'Dark Bush-cricket': 'Pholidoptera griseoaptera',
        'Saw-tailed Bush-cricket': 'Barbitistes serricauda',

        # Grasshoppers
        'Italian Bow-winged Grasshopper': 'Chorthippus italicus',
        'Gravel Grasshopper': 'Chorthippus mollis',
        'Iberian Field Grasshopper': 'Chorthippus vagans',
        'Sicilian Match Grasshopper': 'Chorthippus saulcyi',
        'Common Straw Grasshopper': 'Pseudochorthippus parallelus',
        'Woodland Grasshopper': 'Omocestus rufipes',
        'French Grasshopper': 'Euchorthippus declivus',
        'Pyrenean Grasshopper': 'Chorthippus scalaris',

        # Wart-biters
        'Apulian Wart-biter': 'Decticus albifrons',
        'Gerona Pygmy Wart-biter': 'Decticus verrucivorus',
        'Granada Pygmy Wart-biter': 'Decticus verrucivorus',
        'Lusitanian Pygmy Wart-biter': 'Decticus verrucivorus',

        # Saddle Bush-crickets (Ephippiger genus)
        'Perez\'s Saddle Bush-cricke': 'Ephippiger perezi',  # Note: typo in original
        'Corsican Saddle Bush-cricket': 'Ephippiger corsicus',
        'Spanish Saddle Bush-cricket': 'Ephippiger hispanicus',
        'Swiss Saddle Bush-cricket': 'Ephippiger helveticus',
        'Albanian Saddle Bush-cricket': 'Ephippiger albanicus',
        'Apennine Saddle Bush-cricket': 'Ephippiger apenninius',
        'Catalan Saddle Bush-cricket': 'Ephippiger catalanus',
        'Rambur\'s Saddle Bush-cricket': 'Ephippiger ramburi',

        # Geographic variants of same species
        'Sardinian Mimetic Bush-cricket': 'Tylopsis lilifolia',
        'Lily Bush-cricket': 'Tylopsis lilifolia',
        'Vicentine Saw-tailed Bush-cricket': 'Barbitistes vicetinus',
        'Large Saw-tailed Bush-cricket': 'Barbitistes serricauda',
        'Alpine Saw Bush-cricket': 'Barbitistes alpinus',
        'Southern Saw-tailed Bush-cricket': 'Barbitistes serricauda',
        'Eastern Mountain Bush-cricket': 'Barbitistes obtusus',
        'Hooked Bright Bush-cricket': 'Poecilimon hamatus',
        'Fischer\'s Bush-cricket': 'Barbitistes fischeri',
        'Sepia Bush-cricket': 'Sepiana sepium',
        'Bonfils\' Bush-cricket': 'Barbitistes bonfillsi',
        'Carlotta\'s Saddle Bush-cricket': 'Ephippiger carlottae',

        # Mole crickets
        'Northern Mole Cricket': 'Neocurtilla hexadactyla',
        'Prairie Mole Cricket': 'Gryllotalpa major',
        'European mole cricket': 'Gryllotalpa gryllotalpa',
        'African mole cricket': 'Gryllotalpa africana',
        'Tawny mole cricket': 'Gryllotalpa orientalis',
        'Dark Night Mole Cricket': 'Gryllotalpa orientalis',
        'Mole Cricket': 'Gryllotalpa gryllotalpa',
        'Vineyard Mole-cricket': 'Gryllotalpa vineae',

        # Tree crickets
        'Broad-winged Tree Cricket': 'Oecanthus pellucens',
        'Pine Tree Cricket': 'Oecanthus pini',
        'Striped Tree Cricket': 'Oecanthus nigricornis',
        'Davis\'s Tree Cricket': 'Oecanthus davisi',
        'Black-horned tree cricket': 'Oecanthus nigricornis',
        'Tree Cricket': 'Oecanthus pellucens',  # Generic, map to common European species

        # North American species
        'Northern Bush-katydid': 'Scudderia septentrionalis',
        'Mormon Cricket': 'Anabrus simplex',
        'Broad-winged Bush-katydid': 'Scudderia pistillata',
        'Angular-winged katydid': 'Microcentrum rhombifolium',
        'Broad-winged katydid': 'Microcentrum rhombifolium',

        # Mediterranean/European specialties
        'Betico Bush-cricket': 'Pseudophyllus neriosus',
        'White-lined Bright Bush-cricket': 'Poecilimon linearis',
        'False robust conehead': 'Ruspolia nitidula',
        'Round-tipped Conehead': 'Ruspolia nitidula',
        'Broad-tipped Conehead': 'Ruspolia nitidula',
        'Long-winged Conehead': 'Conocephalus fuscus',
        'Eurasian meadow katydid': 'Pseudochorthippus parallelus',
        'African Meadow Katydid': 'Ornithacris cavroisi',

        # Wart-biters (additional)
        'Wartbiter': 'Decticus verrucivorus',
        'Thyme Pygmy Wart-biter': 'Decticus verrucivorus',

        # Desert/specialized habitats
        'Desert Cricket': 'Gryllomorpha dalmatina',
        'Reed Cricket': 'Pseudogryllus exiguus',
        'Two-spotted Cricket': 'Gryllomorpha dalmatina',
        'Striped Marsh-cricket': 'Pteronemobius heydenii',
        'Japanese burrowing cricket': 'Velarifictorus micado',

        # Asian/Pacific species
        'Pacific Ducetia': 'Ducetia japonica',
        'Tinkling leaf-runner': 'Mecopoda elongata',

        # Keep unknowns as-is for manual review
        'Mystery mystery': 'Mystery mystery',
        'Sonus naturalis': 'Sonus naturalis',
    }

    return mapping

def apply_mapping_rules(species_name, mapping):
    """Apply mapping rules to convert common names to scientific names"""

    # Direct mapping
    if species_name in mapping:
        return mapping[species_name], 'direct_mapping'

    # Check for already scientific names (Genus species pattern)
    scientific_pattern = re.match(r'^[A-Z][a-z]+ [a-z]+$', species_name)
    if scientific_pattern:
        return species_name, 'already_scientific'

    # Pattern-based mapping for common variations
    # Note: Use re.sub() for backreferences to work properly
    patterns = [
        # Bush-cricket variations - keep original format
        (r'(.+) Bush-cricket', lambda m: f"{m.group(1)} Bush-cricket"),  # Standardize format
        (r'(.+) Saddle Bush-cricket', lambda m: 'Ephippiger species'),  # Most saddle bush-crickets are Ephippiger
        (r'(.+) Saw(?:-tailed)? Bush-cricket', lambda m: 'Barbitistes species'),  # Most saw-tailed are Barbitistes
        (r'(.+) Bright Bush-cricket', lambda m: 'Poecilimon species'),  # Most bright bush-crickets are Poecilimon

        # Grasshopper variations - standardize capitalization
        (r'(.+) Grasshopper', lambda m: f"{m.group(1)} grasshopper"),
        (r'(.+) Field Grasshopper', lambda m: 'Chorthippus species'),  # Most field grasshoppers are Chorthippus

        # Cricket variations
        (r'(.+) Field Cricket', lambda m: 'Gryllus species'),  # Most field crickets are Gryllus
        (r'(.+) Tree Cricket', lambda m: 'Oecanthus species'),  # Most tree crickets are Oecanthus
        (r'(.+) Mole Cricket', lambda m: 'Gryllotalpa species'),  # Most mole crickets are Gryllotalpa
    ]

    for pattern, replacement_func in patterns:
        match = re.match(pattern, species_name, re.IGNORECASE)
        if match:
            return replacement_func(match), 'pattern_mapping'

    return f'UNMAPPED_{species_name}', 'unmapped'

def analyze_mapping_coverage():
    """Analyze how well our mapping covers the Xeno-canto species"""

    species_counts = load_xenocanto_species()
    mapping = create_orthoptera_mapping()

    print(f"📊 Xeno-canto Species Mapping Analysis")
    print("=" * 50)
    print(f"Total species: {len(species_counts)}")
    print(f"Total recordings: {species_counts.sum()}")

    mapped_species = 0
    mapped_recordings = 0
    unmapped_species = []

    for species, count in species_counts.items():
        scientific_name, mapping_type = apply_mapping_rules(species, mapping)

        if not scientific_name.startswith('UNMAPPED_'):
            mapped_species += 1
            mapped_recordings += count
        else:
            unmapped_species.append((species, count, mapping_type))

    print(f"\n✅ Successfully mapped:")
    print(f"   Species: {mapped_species}/{len(species_counts)} ({mapped_species/len(species_counts)*100:.1f}%)")
    print(f"   Recordings: {mapped_recordings}/{species_counts.sum()} ({mapped_recordings/species_counts.sum()*100:.1f}%)")

    # Show top unmapped species by recording count
    unmapped_species.sort(key=lambda x: x[1], reverse=True)
    print(f"\n❌ Top unmapped species (by recording count):")
    for species, count, _ in unmapped_species[:15]:
        print(f"   {species}: {count} recordings")

    # Save complete mapping
    complete_mapping = {}
    for species in species_counts.index:
        scientific_name, mapping_type = apply_mapping_rules(species, mapping)
        complete_mapping[species] = {
            'scientific_name': scientific_name,
            'mapping_type': mapping_type,
            'recording_count': int(species_counts[species])  # Convert numpy int64 to int
        }

    output_file = Path('data/xenocanto_species_mapping.json')
    with open(output_file, 'w') as f:
        json.dump(complete_mapping, f, indent=2)

    print(f"\n💾 Complete mapping saved: {output_file}")

    return complete_mapping, unmapped_species

def find_cross_dataset_overlaps():
    """Find overlaps after applying scientific name mapping"""

    # Load mapped Xeno-canto species
    mapping_file = Path('data/xenocanto_species_mapping.json')
    if not mapping_file.exists():
        print("❌ Run analyze_mapping_coverage() first to create mapping file")
        return

    with open(mapping_file) as f:
        xc_mapping = json.load(f)

    # Get scientific names from Xeno-canto (excluding unmapped)
    xc_scientific = set()
    for species_data in xc_mapping.values():
        sci_name = species_data['scientific_name']
        if not sci_name.startswith('UNMAPPED_') and not sci_name.endswith(' species'):
            xc_scientific.add(sci_name)

    # Load other datasets
    insect_df = pd.read_csv('data/raw/insectset459/InsectSet459_Train_Val_Annotation.csv')
    insect_species = set(insect_df['species_name'].str.replace('_', ' ', regex=False))

    sina_file = Path('data/raw/sina/audio/taxa.txt')
    if sina_file.exists():
        with open(sina_file) as f:
            sina_species = set(line.strip() for line in f if line.strip())
    else:
        sina_species = set()

    print(f"🔄 Cross-Dataset Overlaps (After Mapping)")
    print("=" * 50)

    # Find overlaps
    xc_insect_overlap = xc_scientific & insect_species
    xc_sina_overlap = xc_scientific & sina_species
    insect_sina_overlap = insect_species & sina_species

    print(f"Xeno-canto ∩ InsectSet459: {len(xc_insect_overlap)} species")
    for species in sorted(xc_insect_overlap):
        print(f"   {species}")

    print(f"Xeno-canto ∩ SINA: {len(xc_sina_overlap)} species")
    for species in sorted(xc_sina_overlap):
        print(f"   {species}")

    print(f"InsectSet459 ∩ SINA: {len(insect_sina_overlap)} species")
    for species in sorted(insect_sina_overlap):
        print(f"   {species}")

    # Calculate combined statistics
    total_unique = len(xc_scientific | insect_species | sina_species)
    print(f"\n🦗 Total unique species: {total_unique}")

    return {
        'xenocanto': xc_scientific,
        'insectset459': insect_species,
        'sina': sina_species,
        'overlaps': {
            'xc_insect': xc_insect_overlap,
            'xc_sina': xc_sina_overlap,
            'insect_sina': insect_sina_overlap
        }
    }

if __name__ == "__main__":
    print("🗺️  Mapping Xeno-canto Common Names to Scientific Names")
    print("=" * 60)

    # Analyze current mapping coverage
    mapping_data, unmapped = analyze_mapping_coverage()

    print("\n" + "=" * 60)

    # Find cross-dataset overlaps
    overlap_data = find_cross_dataset_overlaps()

    print(f"\n📋 Next steps:")
    print(f"1. Review unmapped species and add to mapping dictionary")
    print(f"2. Use overlapping species for combined training datasets")
    print(f"3. Consider creating hierarchical labels (genus, family level)")