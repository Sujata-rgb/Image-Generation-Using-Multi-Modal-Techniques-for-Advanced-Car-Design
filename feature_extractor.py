"""
feature_extractor.py

This module is responsible for loading the AI Vision-Language models (like OpenAI's CLIP).
It mathematically converts both images and text into arrays of numbers (called "embeddings" or "feature vectors").
This allows the system to compare a text sentence with a car image mathematically to see how similar they are.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

CACHE_DIR = "feature_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class CLIPExtractor:
    """
    Loads OpenAI's CLIP model. CLIP is a neural network trained on millions of images and text pairs.
    It excels at understanding what is inside an image based on human language.
    """
    MODEL_NAME = "ViT-B-32"
    PRETRAINED = "openai"
    DIM        = 512

    def __init__(self):
        try:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.MODEL_NAME, pretrained=self.PRETRAINED
            )
            self.model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model  = self.model.to(self.device)
            self.available = True
            self.name   = f"CLIP-{self.MODEL_NAME}"
            print(f"[CLIP] Model loaded on {self.device}. Outputting {self.DIM}-dimension vectors.")
        except Exception as e:
            print(f"[CLIP] Not available: {e}")
            self.available = False

    def extract(self, image_path: str) -> np.ndarray:
        """Converts a single image file into a mathematical vector."""
        img    = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
        return feat.cpu().numpy().flatten().astype(np.float32)

    def extract_text(self, text: str) -> np.ndarray:
        """Converts a text sentence into a mathematical vector."""
        import open_clip
        tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        tokens    = tokenizer([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
        return feat.cpu().numpy().flatten().astype(np.float32)

    def encode_text_batch(self, texts: list[str]) -> np.ndarray:
        """Encodes multiple text prompts at once. Useful for categorizing."""
        import open_clip
        tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        tokens    = tokenizer(texts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        return feats.cpu().numpy().astype(np.float32)


class ResNetExtractor:
    """
    A fallback AI model used only if CLIP fails to load.
    ResNet is a standard image classification model, but it cannot understand text.
    """
    DIM = 2048

    def __init__(self):
        import torchvision.models as models
        import torchvision.transforms as T

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        base        = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model  = torch.nn.Sequential(*list(base.children())[:-1])
        self.model.eval().to(self.device)
        self.available = True
        self.name   = "ResNet-50"

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"[ResNet-50] Model loaded on {self.device}. Outputting {self.DIM}-dimension vectors.")

    def extract(self, image_path: str) -> np.ndarray:
        img    = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        return feat.cpu().numpy().flatten().astype(np.float32)

    def extract_text(self, text: str) -> np.ndarray:
        raise NotImplementedError("ResNet cannot process text. Please install open-clip-torch.")


def get_extractor(prefer_clip: bool = True):
    """Factory function to get the best available AI model."""
    if prefer_clip:
        clip = CLIPExtractor()
        if clip.available:
            return clip
    return ResNetExtractor()


def extract_features(records: list[dict],
                     extractor,
                     force_recompute: bool = False) -> np.ndarray:
    """
    Runs all 910 car images through the AI model to convert them into mathematical vectors.
    To save time, the result is saved to the hard drive in the 'feature_cache' folder.
    The next time the server starts, it will instantly load the cache instead of recalculating.
    """
    tag        = extractor.name.replace("/", "-").replace(" ", "_")
    cache_file = os.path.join(CACHE_DIR, f"{tag}_features.npz")

    # Load cache if it exists and matches the dataset
    if os.path.exists(cache_file) and not force_recompute:
        data = np.load(cache_file, allow_pickle=True)
        if list(data["ids"]) == [r["id"] for r in records]:
            print(f"[Cache] Image vectors loaded instantly from {cache_file}.")
            return data["features"]
        print("[Cache] Dataset changed, recomputing vectors...")

    print(f"Converting {len(records)} images into mathematical vectors using {extractor.name}...")
    vectors = []
    for i, rec in enumerate(records):
        try:
            vec = extractor.extract(rec["path"])
        except Exception as e:
            print(f"Warning: Failed to process {rec['filename']}: {e}. Using blank vector.")
            vec = np.zeros(extractor.DIM, dtype=np.float32)
        vectors.append(vec)
        if (i + 1) % 100 == 0 or (i + 1) == len(records):
            print(f"{i+1}/{len(records)} images processed...")

    features = np.array(vectors, dtype=np.float32)

    # Save to disk for next time
    np.savez(cache_file, features=features, ids=np.array([r["id"] for r in records]))
    print(f"[Cache] Saved vectors to {cache_file}.")
    return features


def classify_categories(records: list[dict],
                        features: np.ndarray,
                        extractor,
                        categories: list[str],
                        force_recompute: bool = False) -> list[dict]:
    """
    Uses the AI model to automatically figure out what type of car is in each image 
    (e.g., SUV, Sedan, Sports Car). It does this by comparing the image vector 
    against the text vectors of the category names.
    """
    cache_file = os.path.join(CACHE_DIR, "categories.npz")

    if os.path.exists(cache_file) and not force_recompute:
        data = np.load(cache_file, allow_pickle=True)
        if list(data["ids"]) == [r["id"] for r in records]:
            labels = list(data["labels"])
            for rec, lbl in zip(records, labels):
                rec["category"] = lbl
                rec["label"]    = lbl.title()
            print(f"[Cache] AI Category classifications loaded from {cache_file}.")
            return records
        print("[Cache] Recomputing AI classifications...")

    if not isinstance(extractor, CLIPExtractor) or not extractor.available:
        for rec in records:
            rec["category"] = "car"
            rec["label"]    = "Car"
        return records

    print(f"Classifying {len(records)} images into {len(categories)} AI categories...")

    # Convert the category names into text vectors
    prompts     = [f"a photo of a {c}" for c in categories]
    text_feats  = extractor.encode_text_batch(prompts)
    text_norms  = normalize(text_feats, norm="l2")

    # Compare image vectors to the text vectors to find the best match
    img_norms   = normalize(features, norm="l2")
    sims        = img_norms @ text_norms.T
    best_idx    = np.argmax(sims, axis=1)

    for rec, idx in zip(records, best_idx):
        rec["category"] = categories[idx]
        rec["label"]    = categories[idx].title()

    # Save to disk for next time
    np.savez(cache_file, ids=np.array([r["id"] for r in records]), labels=np.array([r["category"] for r in records]))
    print(f"[Cache] AI Category classifications saved to {cache_file}.")
    return records


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """Normalizes features so they can be accurately compared using Cosine Similarity."""
    return normalize(features, norm="l2").astype(np.float32)


def apply_pca(features: np.ndarray, n_components: int = 64) -> tuple[np.ndarray, PCA]:
    """Compresses large vectors into smaller ones to save memory and speed up searches."""
    n_components = min(n_components, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features).astype(np.float32)
    return reduced, pca
