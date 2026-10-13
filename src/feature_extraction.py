"""
feature_extraction.py
----------------------
Steps 2 & 3 of the pipeline: Tokenization + Feature Extraction.

Wraps a pretrained transformer (AutoModel) as a fixed, frozen feature extractor:
raw text in -> a 384/768-dim contextual dense embedding out.
XGBoost (train.py) learns the fraud/legit decision boundary on top of these
contextual representations.
"""

import logging
from typing import List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from . import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobertaFeatureExtractor:
    """Extracts contextual semantic embeddings using transformers.

    Usage:
        extractor = RobertaFeatureExtractor()
        embeddings = extractor.extract_features(["Job posting 1", "Job posting 2"])
    """

    def __init__(
        self,
        model_name: str = config.TRANSFORMER_MODEL_NAME,
        max_length: int = config.MAX_SEQ_LENGTH,
        pooling: str = config.POOLING_STRATEGY,
        device: str = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading transformer '%s' on %s...", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # weights are frozen

    def tokenize(self, texts: List[str]):
        """Converts strings into token tensors."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    @torch.inference_mode()
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        encoded = self.tokenize(texts)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        outputs = self.model(**encoded)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        if self.pooling == "cls":
            pooled = last_hidden[:, 0, :]
        else:
            # Mean-pooling with attention mask exclusion
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts

        return pooled.cpu().numpy()

    def extract_features(
        self,
        texts: Union[List[str], np.ndarray],
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        verbose: bool = True,
    ) -> np.ndarray:
        """Batch feature extraction for large lists of texts."""
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        n = len(texts)
        if n == 0:
            return np.empty((0, 384), dtype=np.float32)

        all_embeddings = []
        for start in range(0, n, batch_size):
            batch = texts[start:start + batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.append(embeddings)
            if verbose and ((start // batch_size) % 20 == 0 or start + batch_size >= n):
                logger.info("Extracted embeddings for %d / %d texts", min(start + batch_size, n), n)

        return np.vstack(all_embeddings).astype(np.float32)

    def extract_single(self, text: str) -> np.ndarray:
        """Extract features for a single job posting text string."""
        return self._embed_batch([text])[0]


if __name__ == "__main__":
    extractor = RobertaFeatureExtractor()
    sample = ["Senior Software Engineer at Apple Inc.", "URGENT work from home $$$ earn fast!"]
    feats = extractor.extract_features(sample, batch_size=2)
    print("Feature matrix shape:", feats.shape)
