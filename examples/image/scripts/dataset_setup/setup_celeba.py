#!/usr/bin/env python3
"""
Setup script for downloading and extracting the CelebA dataset.
Downloads the aligned and cropped images from Google Drive.
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def install_gdown():
    """Install gdown if not already installed."""
    try:
        import gdown
    except ImportError:
        print("gdown not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    return gdown


def download_celeba(output_dir="data/celeba"):
    """
    Download CelebA dataset from Google Drive.

    Args:
        output_dir: Directory where the dataset will be saved
    """
    # Google Drive file ID for CelebA aligned and cropped images
    file_id = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    url = f"https://drive.google.com/uc?id={file_id}"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download zip file
    zip_path = output_path / "img_align_celeba.zip"

    print(f"Downloading CelebA dataset to {zip_path}...")
    print("This may take a while (dataset is ~1.4GB)...")

    gdown = install_gdown()

    try:
        gdown.download(url, str(zip_path), quiet=False)
    except Exception as e:
        print(f"Error downloading with gdown: {e}")
        print("\nAlternative method: Please manually download from:")
        print("https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?resourcekey=0-dYn9z10tMJOBAkviAcfdyQ")
        print(f"And place it in: {zip_path}")
        return False

    # Extract zip file
    print(f"\nExtracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print(f"Successfully extracted to {output_path}")

        # Optional: Remove zip file to save space
        remove = input("\nRemove zip file to save space? (y/n): ").lower().strip()
        if remove == 'y':
            zip_path.unlink()
            print("Zip file removed.")
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

    print("\nCelebA dataset setup complete!")
    print(f"Images are located in: {output_path / 'img_align_celeba'}")
    return True

# base_dir = "/data/matrix/projects/smith/kkuo2/concept_composition/datasets" 
base_dir = None
dataset_root = Path(f"{base_dir}/celeba")
if dataset_root.exists():
    raise Exception(f"Dataset root already exists at {dataset_root.absolute()}. Remove it first if you want to re-prepare the dataset.")
download_celeba(dataset_root)