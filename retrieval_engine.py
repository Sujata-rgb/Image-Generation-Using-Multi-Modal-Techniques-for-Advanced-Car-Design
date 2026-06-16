"""
retrieval_engine.py

This module is responsible for the text-to-image search logic.
It takes the user's text description, converts it into a mathematical vector using the AI model,
and compares it against all the image vectors to find the closest visual matches.
"""

import numpy as np
from feature_extractor import l2_normalize, CLIPExtractor


class RetrievalEngine:
    """
    Handles the search process by storing the extracted features of all images
    and comparing them to text queries.
    """

    def __init__(self,
                 extractor,
                 records:  list[dict],
                 features: np.ndarray):
        self.extractor     = extractor
        self.records       = records
        self.features      = features                    # The raw mathematical features
        self.norm_features = l2_normalize(features)      # Normalized features for accurate comparison

    def retrieve(self,
                 prompt: str,
                 top_k:  int = 5) -> tuple[list[dict], list[float]]:
        """
        Receives a text prompt from the user and returns the top matched images.
        Returns a tuple containing the list of image records and their similarity scores.
        """
        if isinstance(self.extractor, CLIPExtractor) and self.extractor.available:
            return self._clip_retrieve(prompt, top_k)
        else:
            print("Notice: CLIP is unavailable. Falling back to simple visual similarity.")
            return self._visual_fallback(top_k)

    def _clip_retrieve(self, prompt: str, top_k: int):
        """
        Converts the text prompt into a vector and finds the images that 
        have the highest cosine similarity (closest match) to that text.
        """
        # Convert text into a vector using the AI model
        query = self.extractor.extract_text(prompt)
        
        # Normalize the text vector
        query_n = query / (np.linalg.norm(query) + 1e-8)
        
        # Calculate similarity scores against all image vectors using matrix multiplication
        sims = self.norm_features @ query_n
        
        # Sort the scores in descending order and grab the top results
        ranked = np.argsort(sims)[::-1][:top_k]
        
        return ([self.records[i] for i in ranked],
                [float(sims[i])  for i in ranked])

    def _visual_fallback(self, top_k: int):
        """
        A fallback method if the text AI is unavailable. 
        It returns the most visually average or central images in the dataset.
        """
        centroid = self.norm_features.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        sims = self.norm_features @ centroid
        ranked = np.argsort(sims)[::-1][:top_k]
        
        return ([self.records[i] for i in ranked],
                [float(sims[i])  for i in ranked])
