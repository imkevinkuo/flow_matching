#!/usr/bin/env python3
"""
Generate attribute mapping files for pairs of CelebA attributes.
Generates pairs from the product of ATTRIBUTES1 and ATTRIBUTES2.
"""

import json
import os
from pathlib import Path
from itertools import product

# First set of attributes
ATTRIBUTES1 = [
    "Bald",
    "Blond_Hair",
]

# Second set of attributes
ATTRIBUTES2 = [
    "Smiling",
    "Eyeglasses",
    # "Mustache",
    "Wearing_Lipstick",
    # "Wearing_Earrings",
    # "Wearing_Hat",
    # "Wearing_Necklace",
    # "Wearing_Necktie",
]


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
    print(f"ATTRIBUTES1: {len(ATTRIBUTES1)} attributes")
    print(f"ATTRIBUTES2: {len(ATTRIBUTES2)} attributes")
    print(f"Total pairs: {len(ATTRIBUTES1) * len(ATTRIBUTES2)}")
    print()

    generated_count = 0

    # Generate all pairs from the product of ATTRIBUTES1 and ATTRIBUTES2
    for attr1, attr2 in product(ATTRIBUTES1, ATTRIBUTES2):
        generate_mapping_file(attr1, attr2, output_dir)
        generated_count += 1

    print()
    print(f"Summary:")
    print(f"  Generated: {generated_count} files")

if __name__ == "__main__":
    main()
