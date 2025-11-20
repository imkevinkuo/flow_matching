#!/usr/bin/env python3
"""
Generate attribute mapping files for pairs of CelebA attributes.
Excludes pairs where both attributes contain "Hair" or one has "Hair" and the other is "Bald".
"""

import json
import os
from pathlib import Path
from itertools import combinations

# List of attributes to generate pairs from
ATTRIBUTES = [
    "Bald",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Gray_Hair",
    "Male",
    "Smiling",
    "Eyeglasses",
    "Mustache",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Necklace",
    "Wearing_Necktie",
]

def should_exclude_pair(attr1, attr2):
    """
    Check if a pair should be excluded based on the rules:
    - Both attributes contain "Hair"
    - One has "Hair" and the other is "Bald"
    """
    has_hair_1 = "Hair" in attr1 or attr1 == "Bald"
    has_hair_2 = "Hair" in attr2 or attr2 == "Bald"

    return has_hair_1 and has_hair_2

def generate_mapping_file(attr1, attr2, output_dir):
    """
    Generate a mapping file for a pair of attributes.
    Format: {attr1: true, attr2: false} and {attr1: false, attr2: true}
    """
    mapping = {
        "labels": [
            {attr1: True, attr2: False},
            {attr1: False, attr2: True}
        ]
    }

    # Create filename from attributes (lowercase, remove underscores)
    filename = f"{attr1.lower().replace('_', '')}_{attr2.lower().replace('_', '')}.json"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"Generated: {filename}")
    return filepath

def main():
    # Set output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "attribute_mappings"
    output_dir.mkdir(exist_ok=True)

    print(f"Generating attribute mapping files in: {output_dir}")
    print(f"Total attributes: {len(ATTRIBUTES)}")
    print()

    generated_count = 0
    excluded_count = 0

    # Generate all pairs
    for attr1, attr2 in combinations(ATTRIBUTES, 2):
        if should_exclude_pair(attr1, attr2):
            print(f"Excluded: {attr1} <-> {attr2} (Hair/Bald rule)")
            excluded_count += 1
        else:
            generate_mapping_file(attr1, attr2, output_dir)
            generated_count += 1

    print()
    print(f"Summary:")
    print(f"  Generated: {generated_count} files")
    print(f"  Excluded: {excluded_count} pairs")
    print(f"  Total pairs considered: {generated_count + excluded_count}")

if __name__ == "__main__":
    main()
