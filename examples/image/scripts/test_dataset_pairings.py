#!/usr/bin/env python3
"""
Test script to verify dataset pairings load correctly without running training.
Tests three configurations:
1. cat_faces,celeba
2. ms_cats,celeba
3. celeba with attribute-based labels (blond_glasses)
"""

import os
import sys
import argparse
import torch
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_utils import get_data


def test_dataset_config(config_name, args):
    """Test a single dataset configuration"""
    print("\n" + "="*80)
    print(f"Testing configuration: {config_name}")
    print("="*80)

    # Define transform (same as used in training)
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        # Load training dataset
        print("\n--- Loading TRAINING dataset ---")
        train_dataset, num_classes = get_data(args, transform, train=True)

        print(f"\n✓ Training dataset loaded successfully")
        print(f"  Total examples: {len(train_dataset)}")
        print(f"  Number of classes: {num_classes}")

        # Test first few samples
        print("\n--- Testing sample access ---")
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            if len(sample) == 2:
                image, label = sample
                print(f"  Sample {i}: image shape={image.shape}, label={label}")
            elif len(sample) == 3:
                image, label, caption = sample
                print(f"  Sample {i}: image shape={image.shape}, label={label}, caption length={len(caption)}")
            else:
                raise Exception(f"Unexpected sample format: {len(sample)} elements")

        # Load validation dataset
        print("\n--- Loading VALIDATION dataset ---")
        val_dataset, val_num_classes = get_data(args, transform, train=False)

        print(f"\n✓ Validation dataset loaded successfully")
        print(f"  Total examples: {len(val_dataset)}")
        print(f"  Number of classes: {val_num_classes}")

        # Verify class counts match
        if num_classes != val_num_classes:
            raise Exception(f"Class count mismatch: train={num_classes}, val={val_num_classes}")

        print(f"\n{'='*80}")
        print(f"✓ {config_name} PASSED")
        print(f"{'='*80}\n")
        return True

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ {config_name} FAILED")
        print(f"Error: {e}")
        print(f"{'='*80}\n")
        return False


def main():
    """Test all dataset configurations"""

    # Configuration 1: cat_faces,celeba
    print("\n\n" + "#"*80)
    print("# TEST 1: cat_faces,celeba pairing")
    print("#"*80)

    args1 = argparse.Namespace(
        data_path='cat_faces,celeba',
        image_size=64,
        captions=False,
        celeba_attributes=None,
        train_folder='train',
        val_folder='test',
        max_examples_per_dataset=4000
    )

    result1 = test_dataset_config("cat_faces,celeba", args1)


    # Configuration 2: ms_cats,celeba
    print("\n\n" + "#"*80)
    print("# TEST 2: ms_cats,celeba pairing")
    print("#"*80)

    args2 = argparse.Namespace(
        data_path='ms_cats,celeba',
        image_size=64,
        captions=False,
        celeba_attributes=None,
        train_folder='train',
        val_folder='test',
        max_examples_per_dataset=4000
    )

    result2 = test_dataset_config("ms_cats,celeba", args2)


    # Configuration 3: celeba with blond_glasses attributes
    print("\n\n" + "#"*80)
    print("# TEST 3: celeba with blond_glasses attribute labels")
    print("#"*80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    attr_file = os.path.join(script_dir, 'attribute_mappings', 'blond_glasses.json')

    args3 = argparse.Namespace(
        data_path='celeba',
        image_size=64,
        captions=False,
        celeba_attributes=attr_file,
        celeba_attr_file=None,  # Will use default path
        train_folder='train',
        val_folder='test',
        max_examples_per_dataset=4000  # Match training script
    )

    result3 = test_dataset_config("celeba blond_glasses", args3)


    # Summary
    print("\n\n" + "#"*80)
    print("# SUMMARY")
    print("#"*80)
    results = [
        ("cat_faces,celeba", result1),
        ("ms_cats,celeba", result2),
        ("celeba blond_glasses", result3)
    ]

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for config_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {config_name}")

    print(f"\nTotal: {passed}/{total} configurations passed")

    if passed == total:
        print("\n🎉 All dataset configurations are working correctly!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} configuration(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
