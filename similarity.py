"""
Similarity Module for AI Resume Screening Agent.
Calculates semantic similarity vectors between Job Descriptions and Resumes.
Uses SentenceTransformers (all-MiniLM-L6-v2) with TF-IDF Vectorizer fallback.
"""

import logging
import numpy as np
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Safely attempt importing SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SimilarityEngine:
    """Computes semantic embedding vectors and cosine similarity scores."""

    _model_instance = None  # Singleton model cache to avoid repeated model loading

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.use_st = SENTENCE_TRANSFORMERS_AVAILABLE

    @classmethod
    def get_model(cls, model_name: str = EMBEDDING_MODEL_NAME):
        """Loads and returns cached SentenceTransformer model instance."""
        if cls._model_instance is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading SentenceTransformer model '{model_name}'...")
                cls._model_instance = SentenceTransformer(model_name)
                logger.info("SentenceTransformer model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer model '{model_name}': {e}")
                cls._model_instance = None
        return cls._model_instance

    def compute_semantic_similarity(self, jd_text: str, resume_texts: List[str]) -> List[float]:
        """
        Calculates cosine similarity between a Job Description and a list of Resume texts.
        Returns a list of similarity percentage scores (0.0 to 100.0).
        """
        if not jd_text or not resume_texts:
            return [0.0] * len(resume_texts)

        model = self.get_model(self.model_name)

        if model is not None:
            try:
                # Compute embeddings using SentenceTransformers
                jd_embedding = model.encode([jd_text], convert_to_numpy=True)
                resume_embeddings = model.encode(resume_texts, convert_to_numpy=True)

                # Compute cosine similarities
                sim_matrix = cosine_similarity(jd_embedding, resume_embeddings)[0]
                
                # Convert to percentage and scale safely (0 to 100)
                scores = [float(np.clip(s * 100.0, 0.0, 100.0)) for s in sim_matrix]
                return scores
            except Exception as e:
                logger.warning(f"SentenceTransformer embedding error: {e}. Switching to TF-IDF fallback...")

        # Fallback to TF-IDF Cosine Similarity
        return self._compute_tfidf_similarity(jd_text, resume_texts)

    @staticmethod
    def _compute_tfidf_similarity(jd_text: str, resume_texts: List[str]) -> List[float]:
        """TF-IDF + Cosine Similarity Fallback Engine."""
        try:
            documents = [jd_text] + resume_texts
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(documents)

            jd_vector = tfidf_matrix[0:1]
            resume_vectors = tfidf_matrix[1:]

            similarities = cosine_similarity(jd_vector, resume_vectors)[0]
            scores = [float(np.clip(s * 100.0, 0.0, 100.0)) for s in similarities]
            return scores
        except Exception as e:
            logger.error(f"TF-IDF similarity calculation failed: {e}")
            return [0.0] * len(resume_texts)

    def calculate_single_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate similarity between two single strings."""
        results = self.compute_semantic_similarity(text_a, [text_b])
        return results[0] if results else 0.0
