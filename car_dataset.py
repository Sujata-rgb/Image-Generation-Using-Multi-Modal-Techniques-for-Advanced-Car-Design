"""
car_dataset.py

This module handles loading all the car images from your local dataset folder.
It searches for common image file extensions and prepares a list of dictionaries,
each representing an image file, to be processed by the AI later.
"""

import os
from pathlib import Path
from config import DATASET_PATH

# Supported image formats
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset(verbose: bool = True) -> list[dict]:
    """
    Scans the configured dataset folder and returns a list of dictionaries containing
    the file path and an initial "unknown" label that will later be classified by the AI.
    """
    root = Path(DATASET_PATH).resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found at: {root}\n"
            f"Please ensure your images are located inside the data/car_dataset folder."
        )

    # Collect all supported image files
    images = sorted([
        f for f in root.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    ], key=lambda p: _sort_key(p.stem))


    if not images:
        raise ValueError(f"No images found in the directory: {root}")

    records = []
    # Loop through all images and assign them an initial unique ID and an "unknown" label
    for idx, img_path in enumerate(images, start=1):
        records.append({
            "id":       f"C{idx:04d}",
            "label":    "unknown",       # To be populated later by the AI classifier
            "category": "unknown",       # To be populated later by the AI classifier
            "path":     str(img_path),
            "filename": img_path.name,
        })

    if verbose:
        print(f"Loaded {len(records)} images from {root}")
        print(f"First 5 images: {[r['filename'] for r in records[:5]]}")
        print(f"Last 5 images: {[r['filename'] for r in records[-5:]]}")

    return records


def _sort_key(stem: str):
    """
    Sorts filenames logically so that numerically named files (e.g. 1.jpg, 2.jpg) 
    are ordered numerically rather than alphabetically.
    """
    try:
        return (0, int(stem), '')
    except ValueError:
        return (1, 0, stem)
