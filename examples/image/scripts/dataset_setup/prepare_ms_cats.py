"""
Script to download cats_vs_dogs dataset from HuggingFace and extract only cat images
"""
import os
from datasets import load_dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Create directory structure
# base_dir = "/data/matrix/projects/smith/kkuo2/concept_composition/datasets" 
base_dir = None
dataset_root = Path(f"{base_dir}/ms_cats")
if dataset_root.exists():
    raise Exception(f"Dataset root already exists at {dataset_root.absolute()}. Remove it first if you want to re-prepare the dataset.")

train_dir = dataset_root / "train" / "cats"
test_dir = dataset_root / "test" / "cats"

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

print("Downloading cats_vs_dogs dataset from HuggingFace...")
# Load the dataset from HuggingFace
dataset = load_dataset("microsoft/cats_vs_dogs")

# Process training set - extract only cats (label 0)
print("\nProcessing training set...")
train_dataset = dataset['train']
train_cat_count = 0

for idx, sample in enumerate(tqdm(train_dataset)):
    # Label 0 is cat, label 1 is dog
    if sample['labels'] == 0:
        img = sample['image']
        # Save the image
        img_path = train_dir / f"cat_{train_cat_count:05d}.jpg"
        img.save(img_path)
        train_cat_count += 1

print(f"Saved {train_cat_count} cat images to training set")

print("\nCreating train/test split...")
train_images = list(train_dir.glob("*.jpg"))
test_count = int(len(train_images) * 0.2)
test_images = train_images[:test_count]

print(f"Moving {test_count} images to test set...")
for img_path in tqdm(test_images):
    new_path = test_dir / img_path.name
    img_path.rename(new_path)

final_train_count = len(list(train_dir.glob("*.jpg")))
final_test_count = len(list(test_dir.glob("*.jpg")))

print(f"\nFinal dataset statistics:")
print(f"Training images: {final_train_count}")
print(f"Test images: {final_test_count}")
print(f"Total cat images: {final_train_count + final_test_count}")
print(f"Dataset saved to: {dataset_root.absolute()}")
