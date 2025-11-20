"""
Script to download cat faces 64x64 dataset from Kaggle and organize into train/test splits
"""
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

# Create directory structure
# base_dir = "/data/matrix/projects/smith/kkuo2/concept_composition/datasets"
base_dir = None
dataset_root = Path(f"{base_dir}/cat_faces")
if dataset_root.exists():
    raise Exception(f"Dataset root already exists at {dataset_root.absolute()}. Remove it first if you want to re-prepare the dataset.")

train_dir = dataset_root / "train" / "cats"
test_dir = dataset_root / "test" / "cats"

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# Download dataset using Kaggle API
print("Downloading cat faces dataset from Kaggle...")
temp_dir = dataset_root / "temp"
temp_dir.mkdir(exist_ok=True)

# Download the dataset
os.system(f"kaggle datasets download -d spandan2/cats-faces-64x64-for-generative-models -p {temp_dir}")

# Find the downloaded zip file
zip_files = list(temp_dir.glob("*.zip"))
if not zip_files:
    raise Exception("No zip file found after Kaggle download")

zip_path = zip_files[0]

# Extract the dataset
print(f"\nExtracting {zip_path.name}...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# Find all image files
print("\nCollecting image files...")
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
all_images = []
for ext in image_extensions:
    all_images.extend(temp_dir.rglob(f"*{ext}"))
    all_images.extend(temp_dir.rglob(f"*{ext.upper()}"))

if not all_images:
    raise Exception("No image files found in extracted dataset")

print(f"Found {len(all_images)} images")

# Split into train/test (80/20)
test_count = int(len(all_images) * 0.2)
test_images = all_images[:test_count]
train_images = all_images[test_count:]

# Copy images to train directory
print(f"\nCopying {len(train_images)} images to training set...")
for idx, img_path in enumerate(tqdm(train_images)):
    new_path = train_dir / f"cat_{idx:05d}.jpg"
    shutil.copy2(img_path, new_path)

# Copy images to test directory
print(f"Copying {len(test_images)} images to test set...")
for idx, img_path in enumerate(tqdm(test_images)):
    new_path = test_dir / f"cat_{idx:05d}.jpg"
    shutil.copy2(img_path, new_path)

# Clean up temporary directory
print("\nCleaning up temporary files...")
shutil.rmtree(temp_dir)

final_train_count = len(list(train_dir.glob("*.jpg")))
final_test_count = len(list(test_dir.glob("*.jpg")))

print(f"\nFinal dataset statistics:")
print(f"Training images: {final_train_count}")
print(f"Test images: {final_test_count}")
print(f"Total images: {final_train_count + final_test_count}")
print(f"Dataset saved to: {dataset_root.absolute()}")
