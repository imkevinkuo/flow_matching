"""
Dataset utilities for multi-dataset loading with caption support.
Adapted from Diffusion-Models-pytorch for flow matching.
"""

import os
import json
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image


class BalancedSubset(Dataset):
    """Dataset wrapper that creates a balanced subset with max_per_label examples per class

    Args:
        dataset: Base dataset
        max_per_label: Maximum number of examples per label
        num_labels: Total number of labels in the dataset
    """
    def __init__(self, dataset, max_per_label, num_labels):
        self.dataset = dataset
        self.max_per_label = max_per_label

        # Group indices by label
        label_to_indices = {i: [] for i in range(num_labels)}

        print(f"  Scanning dataset to collect up to {max_per_label} examples per label...")
        for idx in range(len(dataset)):
            try:
                result = dataset[idx]
                # Extract label (handle 2-tuple or 3-tuple formats)
                if len(result) == 2:
                    _, label = result
                elif len(result) == 3:
                    _, label, _ = result
                else:
                    raise ValueError(f"Unexpected dataset item format with {len(result)} elements")

                # Add to list if we haven't reached the limit for this label
                if len(label_to_indices[label]) < max_per_label:
                    label_to_indices[label].append(idx)

                # Early exit if all labels have enough examples
                if all(len(indices) >= max_per_label for indices in label_to_indices.values()):
                    break

            except (ValueError, KeyError) as e:
                # Skip images that don't match any label
                continue

        # Flatten all indices
        self.indices = []
        for label in sorted(label_to_indices.keys()):
            self.indices.extend(label_to_indices[label])

        # Print distribution
        print(f"  Balanced subset distribution:")
        for label in sorted(label_to_indices.keys()):
            print(f"    Label {label}: {len(label_to_indices[label])} examples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]


class LabelOffsetDataset(Dataset):
    """Dataset wrapper that adds an offset to all labels"""
    def __init__(self, dataset, label_offset=0):
        self.dataset = dataset
        self.label_offset = label_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        result = self.dataset[idx]
        # Handle different return formats (2-tuple or 3-tuple)
        if len(result) == 2:
            image, label = result
            return image, label + self.label_offset
        elif len(result) == 3:
            image, label, caption = result
            return image, label + self.label_offset, caption
        else:
            raise ValueError(f"Unexpected dataset item format with {len(result)} elements")


class ImageFolderWithAttributes(Dataset):
    """Dataset wrapper that assigns labels based on CelebA attribute combinations

    Attributes file format (JSON):
    Option 1 - Reference standard CelebA attributes:
    {
        "labels": [
            {"Male": true, "Eyeglasses": true},
            {"Male": false, "Eyeglasses": false}
        ]
    }

    Option 2 - Provide custom attributes inline:
    {
        "labels": [
            {"Male": true, "Eyeglasses": true},
            {"Male": false, "Eyeglasses": false}
        ],
        "attributes": {
            "000001": {"Male": false, "Eyeglasses": false},
            "000002": {"Male": true, "Eyeglasses": true},
            ...
        }
    }

    Each entry in "labels" defines a label (0, 1, 2, ...) as a combination of attribute values.
    Images matching those attributes are assigned that label.

    Args:
        image_dataset: Base ImageFolder dataset
        attributes_path: Path to JSON file defining attribute-based labels
        celeba_attr_path: Path to CelebA list_attr_celeba.txt (optional if attributes provided inline)
    """
    def __init__(self, image_dataset, attributes_path, celeba_attr_path=None):
        self.image_dataset = image_dataset

        if not os.path.exists(attributes_path):
            raise FileNotFoundError(f"Attributes file not found at {attributes_path}")

        # Load label definitions (attribute combinations)
        with open(attributes_path, 'r') as f:
            attr_data = json.load(f)

        if 'labels' not in attr_data:
            raise ValueError(f"'labels' key not found in {attributes_path}")

        self.label_definitions = attr_data['labels']
        print(f"Loaded {len(self.label_definitions)} attribute-based label definitions from {attributes_path}")

        # Load CelebA attributes - either from inline data or external file
        if 'attributes' in attr_data:
            self.celeba_attributes = attr_data['attributes']
            print(f"Loaded inline attributes for {len(self.celeba_attributes)} images")
        elif celeba_attr_path is not None:
            if not os.path.exists(celeba_attr_path):
                raise FileNotFoundError(f"CelebA attributes file not found at {celeba_attr_path}")
            self.celeba_attributes = self._load_celeba_attributes(celeba_attr_path)
            print(f"Loaded CelebA attributes for {len(self.celeba_attributes)} images")
        else:
            raise ValueError("Either 'attributes' must be provided in the JSON file or celeba_attr_path must be specified")

    def _load_celeba_attributes(self, celeba_attr_path):
        """Load CelebA attributes from list_attr_celeba.txt

        Format:
        202599
        image_id Attr1 Attr2 ...
        000001.jpg -1 1 1 ...
        """
        attributes = {}

        with open(celeba_attr_path, 'r') as f:
            lines = f.readlines()

        # First line is number of images
        num_images = int(lines[0].strip())

        # Second line is attribute names
        attr_names = lines[1].strip().split()

        # Remaining lines are image data
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            image_filename = parts[0]
            filename_no_ext = os.path.splitext(image_filename)[0]

            # Parse attribute values (-1 = False, 1 = True)
            attr_values = {}
            for i, attr_name in enumerate(attr_names):
                if i + 1 < len(parts):
                    attr_values[attr_name] = (int(parts[i + 1]) == 1)

            attributes[filename_no_ext] = attr_values

        return attributes

    def _match_attributes(self, image_attrs, label_def):
        """Check if image attributes match a label definition"""
        for attr_name, required_value in label_def.items():
            if attr_name not in image_attrs:
                raise ValueError(f"Attribute '{attr_name}' not found in CelebA attributes")
            if image_attrs[attr_name] != required_value:
                return False
        return True

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, original_label = self.image_dataset[idx]

        # Get image filename - handle different dataset types
        if isinstance(self.image_dataset, CelebADataset):
            # CelebADataset: directly access image_files list
            filename = self.image_dataset.image_files[idx]
        elif hasattr(self.image_dataset, 'dataset'):
            # Subset wrapper
            actual_idx = self.image_dataset.indices[idx]
            if isinstance(self.image_dataset.dataset, CelebADataset):
                filename = self.image_dataset.dataset.image_files[actual_idx]
            else:
                img_path = self.image_dataset.dataset.samples[actual_idx][0]
                filename = os.path.basename(img_path)
        elif hasattr(self.image_dataset, 'samples'):
            # ImageFolder
            img_path = self.image_dataset.samples[idx][0]
            filename = os.path.basename(img_path)
        else:
            raise Exception("Dataset does not have expected structure")

        # Extract filename without extension
        filename_no_ext = os.path.splitext(filename)[0]

        # Get attributes for this image
        if filename_no_ext not in self.celeba_attributes:
            raise KeyError(f"Attributes not found for CelebA image: {filename}")

        image_attrs = self.celeba_attributes[filename_no_ext]

        # Find matching label
        new_label = None
        for label_idx, label_def in enumerate(self.label_definitions):
            if self._match_attributes(image_attrs, label_def):
                new_label = label_idx
                break

        if new_label is None:
            raise ValueError(f"Image {filename} does not match any defined label. "
                           f"Attributes: {image_attrs}")

        return image, new_label


class CelebADataset(Dataset):
    """CelebA dataset loader that reads from the standard CelebA directory structure

    Directory structure:
    celeba/
        img_align_celeba/
            000001.jpg
            000002.jpg
            ...
        list_attr_celeba.txt
        list_eval_partition.txt

    Args:
        celeba_root: Root directory containing img_align_celeba folder and metadata files
        transform: Image transformations
        split: 'train' (0), 'val' (1), or 'test' (2)
    """
    def __init__(self, celeba_root, transform=None, split='train'):
        self.celeba_root = celeba_root
        self.transform = transform
        self.img_dir = os.path.join(celeba_root, 'img_align_celeba')

        # Map split name to partition value
        split_map = {'train': 0, 'val': 1, 'test': 2}
        if split not in split_map:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'")
        target_partition = split_map[split]

        # Load partition file
        partition_file = os.path.join(celeba_root, 'list_eval_partition.txt')
        if not os.path.exists(partition_file):
            raise FileNotFoundError(f"Partition file not found at {partition_file}")

        # Read partition assignments
        self.image_files = []
        with open(partition_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, partition = parts
                    if int(partition) == target_partition:
                        self.image_files.append(filename)

        print(f"CelebA {split} split: {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, filename)

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Return dummy label 0 (will be replaced if using attributes)
        return image, 0


class ImageFolderWithCaptions(Dataset):
    """Dataset wrapper that adds captions to ImageFolder dataset

    Dataset-specific matching strategies:
    - cifar10: Uses original_index, maps by numeric filename (e.g., 9.png -> 9)
    - ms_cats: Uses image_index, maps by extracting number from cat_00042.jpg -> 42
    - celeba: Uses image_filename, maps by exact filename (e.g., 000005.jpg)
    - cat_faces: Uses image_filename, maps by exact filename (e.g., 5.jpg)
    - wiki: Uses image_path, maps by full relative path (e.g., 00/10049200_1891-09-16_1958.jpg)
    """
    def __init__(self, image_dataset, captions_path=None):
        self.image_dataset = image_dataset
        # Load each caption file
        if captions_path is not None:
            if os.path.exists(captions_path):
                self._load_caption_file(captions_path)
            else:
                raise FileNotFoundError(f"Caption file not found at {captions_path}")

    def _load_caption_file(self, captions_path):
        """Load a single caption file and add it to caption_mappings"""
        with open(captions_path, 'r') as f:
            caption_data = json.load(f)

        # Detect dataset type from path
        dataset_type = None
        if 'cifar10' in captions_path.lower():
            dataset_type = 'cifar10'
        elif 'ms_cats' in captions_path.lower():
            dataset_type = 'ms_cats'
        elif 'celeba' in captions_path.lower():
            dataset_type = 'celeba'
        elif 'cat_faces' in captions_path.lower():
            dataset_type = 'cat_faces'
        elif 'wiki' in captions_path.lower():
            dataset_type = 'wiki'

        # Determine path pattern for this caption file
        path_pattern = None
        if hasattr(self, '_path_patterns') and captions_path in self._path_patterns:
            path_pattern = self._path_patterns[captions_path]
        else:
            # Auto-detect from dataset type
            if dataset_type == 'cifar10':
                path_pattern = '/cat/'
            elif dataset_type == 'celeba':
                path_pattern = '/faces/'
            elif dataset_type == 'ms_cats':
                path_pattern = '/cats/'
            elif dataset_type == 'cat_faces':
                path_pattern = '/cats/'
            # For other types, path_pattern remains None (matches all)

        # Build caption mapping
        captions = {}
        if 'captions' in caption_data:
            for item in caption_data['captions']:
                caption = item['caption']

                if dataset_type == 'cifar10':
                    if 'original_index' in item:
                        captions[item['original_index']] = caption

                elif dataset_type == 'ms_cats':
                    if 'image_index' in item:
                        captions[item['image_index']] = caption

                elif dataset_type == 'celeba':
                    if 'image_filename' in item:
                        filename_no_ext = os.path.splitext(item['image_filename'])[0]
                        captions[filename_no_ext] = caption

                elif dataset_type == 'cat_faces':
                    if 'image_filename' in item:
                        filename_no_ext = os.path.splitext(item['image_filename'])[0]
                        captions[filename_no_ext] = caption

                elif dataset_type == 'wiki':
                    if 'image_path' in item:
                        img_path = item['image_path'].replace('\\', '/')
                        captions[img_path] = caption

            self.caption_mappings = {
                'dataset_type': dataset_type,
                'captions': captions,
                'path_pattern': path_pattern
            }

            print(f"Loaded {len(captions)} captions from {captions_path}")
            print(f"  Dataset type: {dataset_type}, path pattern: {path_pattern}")
        else:
            raise ValueError(f"'captions' key not found in {captions_path}")

    def __len__(self):
        return len(self.image_dataset)

    def _get_caption_for_path(self, img_path, mapping):
        """Get caption for an image path using a specific caption mapping"""
        dataset_type = mapping['dataset_type']
        captions = mapping['captions']

        filename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(filename)[0]

        # Use dataset-specific matching strategy
        if dataset_type == 'cifar10':
            # CIFAR10: Extract numeric index from filename (e.g., 9.png -> 9)
            try:
                file_idx = int(filename_no_ext)
                if file_idx not in captions:
                    raise KeyError(f"Caption not found for CIFAR10 image with index {file_idx} (file: {filename})")
                return captions[file_idx]
            except ValueError:
                raise ValueError(f"Could not extract numeric index from CIFAR10 filename: {filename}")

        elif dataset_type == 'ms_cats':
            # MS Cats: Extract number from cat_00042.jpg -> 42
            import re
            match = re.search(r'(\d+)', filename_no_ext)
            if not match:
                raise ValueError(f"Could not extract numeric index from MS Cats filename: {filename}")
            file_idx = int(match.group(1))
            if file_idx not in captions:
                raise KeyError(f"Caption not found for MS Cats image with index {file_idx} (file: {filename})")
            return captions[file_idx]

        elif dataset_type == 'celeba':
            # CelebA: Direct filename match (e.g., 000005)
            if filename_no_ext not in captions:
                raise KeyError(f"Caption not found for CelebA image: {filename}")
            return captions[filename_no_ext]

        elif dataset_type == 'cat_faces':
            # Cat Faces: Direct filename match (e.g., 5)
            if filename_no_ext not in captions:
                raise KeyError(f"Caption not found for Cat Faces image: {filename}")
            return captions[filename_no_ext]

        elif dataset_type == 'wiki':
            # Wiki: Full relative path match (e.g., 00/10049200_1891-09-16_1958.jpg)
            img_path_normalized = img_path.replace('\\', '/')
            path_parts = img_path_normalized.split('/')

            # Extract relative path after train/test folder
            rel_path = None
            if 'train' in path_parts:
                train_idx = path_parts.index('train')
                if train_idx + 2 < len(path_parts):
                    rel_path = '/'.join(path_parts[train_idx+2:])
            elif 'test' in path_parts:
                test_idx = path_parts.index('test')
                if test_idx + 2 < len(path_parts):
                    rel_path = '/'.join(path_parts[test_idx+2:])

            if not rel_path:
                raise ValueError(f"Could not extract relative path from Wiki image path: {img_path}")
            if rel_path not in captions:
                raise KeyError(f"Caption not found for Wiki image: {rel_path}")
            return captions[rel_path]

        raise ValueError(f"Unknown dataset type: {dataset_type}")

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]

        # Handle case where image_dataset is a Subset
        if hasattr(self.image_dataset, 'dataset'):
            actual_idx = self.image_dataset.indices[idx]
            img_path = self.image_dataset.dataset.samples[actual_idx][0]
        elif hasattr(self.image_dataset, 'samples'):
            img_path = self.image_dataset.samples[idx][0]
        else:
            raise Exception("Dataset does not have 'samples' attribute")

        path_pattern = self.caption_mappings['path_pattern']
        if path_pattern is not None and path_pattern not in img_path:
            raise ValueError(f"Image path does not match expected pattern '{path_pattern}': {img_path}")

        caption = self._get_caption_for_path(img_path, self.caption_mappings)

        return image, label, caption


def get_data(args, transform, train=True):
    """
    Load datasets with support for multiple datasets, captions, and attribute-based labels.

    Args:
        args: Arguments object containing:
            - data_path: Comma-separated list of dataset names (e.g., "cat_faces,celeba")
            - captions: Boolean flag to enable caption loading
            - celeba_attributes: Path to attribute-based label definitions file (optional)
            - celeba_attr_file: Path to CelebA list_attr_celeba.txt (optional, default: <dataset>/list_attr_celeba.txt)
            - train_folder: Name of training folder (default: "train")
            - max_examples_per_dataset: Maximum number of examples to use from each dataset (optional)
        transform: Transform to apply to images
        train: Whether to load training or validation data

    Returns:
        dataset: Combined dataset (ConcatDataset if multiple datasets)
        num_classes: Total number of classes across all datasets
    """
    dataset_names = [ds.strip() for ds in args.dataset.split(',')]

    # Build full paths for each dataset
    base_path = args.data_path
    dataset_paths = []
    for ds_name in dataset_names:
        dataset_paths.append(os.path.join(base_path, ds_name))

    # Determine folder name
    folder_name = getattr(args, 'train_folder', 'train') if train else getattr(args, 'val_folder', 'test')

    datasets = []
    use_captions = getattr(args, 'captions', False)
    use_attributes = getattr(args, 'celeba_attributes', None)
    max_examples_per_dataset = getattr(args, 'max_examples_per_dataset', None)

    label_offset = 0
    total_classes = 0

    for i, dataset_path in enumerate(dataset_paths):
        dataset_name = dataset_names[i]
        if dataset_name == 'celeba':
            # Use CelebADataset for raw CelebA format
            split = 'train' if train else 'test'
            ds = CelebADataset(dataset_path, transform=transform, split=split)
            num_classes_in_dataset = 1  # Default: no class conditioning
        else:
            # Load ImageFolder dataset
            dataset_folder = os.path.join(dataset_path, folder_name)
            ds = torchvision.datasets.ImageFolder(dataset_folder, transform=transform)
            num_classes_in_dataset = len(ds.classes)

        # Handle attribute-based labels for CelebA
        if use_attributes and dataset_name == 'celeba':
            # Get CelebA attribute file path
            celeba_attr_file = getattr(args, 'celeba_attr_file', None)
            if celeba_attr_file is None:
                celeba_attr_file = os.path.join(dataset_path, 'list_attr_celeba.txt')

            ds = ImageFolderWithAttributes(ds, use_attributes, celeba_attr_file)

            # Load the attributes file to count number of labels
            with open(use_attributes, 'r') as f:
                attr_data = json.load(f)
            num_classes_in_dataset = len(attr_data['labels'])
            print(f"  Using attribute-based labels: {num_classes_in_dataset} classes")

        # If num_classes is 1 and no attributes, skip label conditioning
        if num_classes_in_dataset == 1 and not use_attributes:
            print(f"  Single class dataset - training without label conditioning")

        total_classes += num_classes_in_dataset

        # Add captions if enabled (not compatible with attributes)
        if use_captions and not (use_attributes and dataset_name == 'celeba'):
            captions_path = os.path.join(dataset_path, 'captions.json')
            ds = ImageFolderWithCaptions(ds, captions_path)

        # Limit number of examples if specified
        original_size = len(ds)
        if max_examples_per_dataset is not None and max_examples_per_dataset < len(ds):
            # Use balanced sampling for attribute-based labels (sample per label)
            if use_attributes and dataset_name == 'celeba':
                ds = BalancedSubset(ds, max_examples_per_dataset, num_classes_in_dataset)
            else:
                # For regular datasets, just take first N examples
                indices = list(range(max_examples_per_dataset))
                ds = torch.utils.data.Subset(ds, indices)

        # Apply label offset for all datasets except the first one
        if i > 0:
            ds = LabelOffsetDataset(ds, label_offset)

        datasets.append(ds)

        print(f"Loaded dataset {i+1}/{len(dataset_paths)}: {dataset_names[i]}")
        if dataset_name == 'celeba':
            print(f"  Path: {dataset_path}")
        else:
            print(f"  Path: {dataset_folder}")
        if max_examples_per_dataset is not None and max_examples_per_dataset < original_size:
            print(f"  Examples: {len(ds)} (limited from {original_size}), Classes: {num_classes_in_dataset}, Label offset: {label_offset}")
        else:
            print(f"  Examples: {len(ds)}, Classes: {num_classes_in_dataset}, Label offset: {label_offset}")

        # Update label offset for next dataset
        label_offset += num_classes_in_dataset

    # Concatenate datasets if multiple, otherwise return single dataset
    if len(datasets) > 1:
        dataset = torch.utils.data.ConcatDataset(datasets)
        print(f"\nMerged {len(datasets)} datasets: Total examples: {len(dataset)}, Total classes: {total_classes}")
    else:
        dataset = datasets[0]
        print(f"\nLoaded single dataset: Total examples: {len(dataset)}, Total classes: {total_classes}")

    return dataset, total_classes
