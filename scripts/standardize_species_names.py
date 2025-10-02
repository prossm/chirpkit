#!/usr/bin/env python3
"""
Standardize species names across all datasets to enable unified training
Maps common names to scientific names and identifies cross-dataset overlaps
"""

import pandas as pd
import json
from pathlib import Path
import re

# Common name to scientific name mapping for Orthoptera
ORTHOPTERA_NAME_MAPPING = {
    # Bush-crickets / Katydids
    'Great Green Bush-cricket': 'Tettigonia viridissima',
    'Mediterranean Katydid': 'Tettigonia viridissima',  # Synonym
    'Upland Green Bush-cricket': 'Tettigonia cantans',
    'Phaneroptera nana': 'Phaneroptera nana',  # Already scientific
    'Schmidt\'s Marbled Bush-cricket': 'Barbitistes constrictus',
    'Greek Predatory Bush-cricket': 'Saga hellenica',
    'Limnos Bush-cricket': 'Poecilimon zimmeri',
    'La Greca\'s Slender Bush-cricket': 'Tessellana lagrecai',
    'Arabian Sickle Bush-cricke': 'Saga ephippigera',  # Typo in source
    'Black Saw Bush-cricket': 'Barbitistes serricauda',
    'Noble Bright Bush-cricket': 'Poecilimon nobilis',
    'Sickle-bearing Bush-cricket': 'Phaneroptera falcata',
    'Italian Bow-winged Grasshopper': 'Chorthippus italicus',
    'Oak Bush-cricket': 'Meconema thalassinum',
    'Dark Night Mole Cricket': 'Gryllotalpa orientalis',

    # Field Crickets
    'Field Cricket': 'Gryllus campestris',
    'Fall Field Cricket': 'Gryllus pennsylvanicus',
    'Spring Field Cricket': 'Gryllus veletis',
    'House Cricket': 'Acheta domesticus',
    'House cricket': 'Acheta domesticus',
    'Indian House Cricket': 'Acheta domesticus',
    'Two-spotted Tree Cricket': 'Neoxabea bipunctata',
    'Wood Cricket': 'Nemobius sylvestris',

    # Grasshoppers
    'Gravel Grasshopper': 'Chorthippus mollis',
    'Iberian Field Grasshopper': 'Chorthippus vagans',
    'Sicilian Match Grasshopper': 'Chorthippus saulcyi',
    'Apulian Wart-biter': 'Decticus albifrons',
    'Gerona Pygmy Wart-biter': 'Decticus verrucivorus',
    'Perez\'s Saddle Bush-cricke': 'Ephippiger perezi',  # Typo in source

    # Mole Crickets
    'Northern Mole Cricket': 'Neocurtilla hexadactyla',
    'Prairie Mole Cricket': 'Gryllotalpa major',
    'European mole cricket': 'Gryllotalpa gryllotalpa',
    'African mole cricket': 'Gryllotalpa africana',
    'Tawny mole cricket': 'Gryllotalpa orientalis',

    # Katydids (North American)
    'Northern Bush-katydid': 'Scudderia septentrionalis',
    'Riverine Mountain-cricket': 'Anabrus simplex',

    # Generic/Unknown entries - keep as is for now
    'Mystery mystery': 'Mystery mystery',
    'Sonus naturalis': 'Sonus naturalis',
    'Unknown species': 'Unknown species',

    # Geographic variants - map to base species
    'Sardinian Mimetic Bush-cricket': 'Tylopsis lilifolia',
    'Vicentine Saw-tailed Bush-cricket': 'Barbitistes vicetinus',
    'Large Saw-tailed Bush-cricket': 'Barbitistes serricauda',
    'Alpine Saw Bush-cricket': 'Barbitistes alpinus',
    'Corsican Saddle Bush-cricket': 'Ephippiger corsicus',
    'Eastern Mountain Bush-cricket': 'Barbitistes obtusus',
    'Hooked Bright Bush-cricket': 'Poecilimon hamatus',
    'Fischer\'s Bush-cricket': 'Barbitistes fischeri',
    'Sepia Bush-cricket': 'Sepiana sepium',
    'Lily Bush-cricket': 'Tylopsis lilifolia',
    'Bonfils\' Bush-cricket': 'Barbitistes bonfillsi',
    'Carlotta\'s Saddle Bush-cricket': 'Ephippiger carlottae',
    'Betico Bush-cricket': 'Pseudophyllus neriosus',
    'Albanian Saddle Bush-cricket': 'Ephippiger albanicus',
    'White-lined Bright Bush-cricket': 'Poecilimon linearis',
    'False robust conehead': 'Ruspolia nitidula',
}

def standardize_species_name(name, dataset_source):
    """Standardize a species name to scientific format"""

    # Handle already scientific names (from InsectSet459, SINA)
    if dataset_source in ['insectset459', 'sina']:
        # Convert underscores to spaces for InsectSet459
        scientific_name = name.replace('_', ' ') if dataset_source == 'insectset459' else name

        # Basic validation - should be "Genus species" format
        parts = scientific_name.split()
        if len(parts) >= 2 and parts[0][0].isupper():
            return scientific_name.strip()
        else:
            print(f"⚠️  Unusual scientific name format: {scientific_name}")
            return scientific_name.strip()

    # Handle Xeno-canto common names
    elif dataset_source == 'xenocanto':
        # Direct mapping
        if name in ORTHOPTERA_NAME_MAPPING:
            return ORTHOPTERA_NAME_MAPPING[name]

        # Try to extract scientific name if it exists in the string
        scientific_match = re.search(r'([A-Z][a-z]+ [a-z]+)', name)
        if scientific_match:
            return scientific_match.group(1)

        # For unmapped common names, keep as-is but mark for review
        print(f"⚠️  Unmapped common name: {name}")
        return f"UNMAPPED_{name.replace(' ', '_')}"

    return name

def analyze_species_overlap():
    """Analyze species overlap across all datasets"""

    datasets = {}

    # 1. Load Xeno-canto species (already processed)
    xenocanto_file = Path('data/processed/xenocanto/processed_files.csv')
    if xenocanto_file.exists():
        df = pd.read_csv(xenocanto_file)
        xenocanto_species = df['species'].unique()
        datasets['xenocanto'] = [standardize_species_name(s, 'xenocanto') for s in xenocanto_species]
        print(f"📁 Xeno-canto: {len(xenocanto_species)} species")

    # 2. Load InsectSet459 species
    insectset_file = Path('data/raw/insectset459/InsectSet459_Train_Val_Annotation.csv')
    if insectset_file.exists():
        df = pd.read_csv(insectset_file)
        insectset_species = df['species_name'].unique()
        datasets['insectset459'] = [standardize_species_name(s, 'insectset459') for s in insectset_species]
        print(f"📁 InsectSet459: {len(insectset_species)} species")

    # 3. Load SINA species
    sina_file = Path('data/raw/sina/audio/taxa.txt')
    if sina_file.exists():
        with open(sina_file, 'r') as f:
            sina_species = [line.strip() for line in f if line.strip()]
        datasets['sina'] = [standardize_species_name(s, 'sina') for s in sina_species]
        print(f"📁 SINA: {len(sina_species)} species")

    # Analyze overlaps
    print(f"\n📊 Species Overlap Analysis")
    print("=" * 50)

    all_species = set()
    for dataset_name, species_list in datasets.items():
        all_species.update(species_list)

    print(f"🦗 Total unique species (after standardization): {len(all_species)}")

    # Find overlaps between datasets
    overlaps = {}
    dataset_names = list(datasets.keys())

    for i, dataset1 in enumerate(dataset_names):
        for dataset2 in dataset_names[i+1:]:
            overlap = set(datasets[dataset1]) & set(datasets[dataset2])
            if overlap:
                overlaps[f"{dataset1}_vs_{dataset2}"] = overlap
                print(f"🔄 {dataset1} ∩ {dataset2}: {len(overlap)} species")
                for species in sorted(overlap)[:5]:  # Show first 5
                    print(f"    {species}")
                if len(overlap) > 5:
                    print(f"    ... and {len(overlap) - 5} more")

    # Show unmapped species
    unmapped = [s for s in all_species if s.startswith('UNMAPPED_')]
    if unmapped:
        print(f"\n⚠️  Unmapped species requiring manual review: {len(unmapped)}")
        for species in sorted(unmapped)[:10]:
            print(f"    {species}")

    return datasets, overlaps, all_species

def create_unified_species_mapping():
    """Create a unified species mapping file for consistent training"""

    datasets, overlaps, all_species = analyze_species_overlap()

    # Create unified mapping
    species_mapping = {}
    species_id = 0

    for species in sorted(all_species):
        if not species.startswith('UNMAPPED_'):
            species_mapping[species] = species_id
            species_id += 1

    # Save mapping
    output_file = Path('data/unified_species_mapping.json')
    with open(output_file, 'w') as f:
        json.dump(species_mapping, f, indent=2)

    print(f"\n💾 Unified species mapping saved: {output_file}")
    print(f"🦗 Mapped species: {len(species_mapping)}")

    return species_mapping

if __name__ == "__main__":
    print("🔄 Standardizing Species Names Across Datasets")
    print("=" * 60)

    mapping = create_unified_species_mapping()

    print("\n✅ Species standardization complete!")
    print("📋 Next steps:")
    print("1. Review any unmapped species and add to ORTHOPTERA_NAME_MAPPING")
    print("2. Update preprocessing scripts to use unified species mapping")
    print("3. Re-process datasets with standardized names for unified training")