"""Embed texts using a HuggingFace model's mean-pooled hidden states."""

import logging
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Produce fixed-length embeddings for texts using a pretrained LM.

    Uses the mean of the last hidden states as the embedding vector.
    Caches embeddings to disk to avoid re-computation.

    Args:
        model_name: HuggingFace model name used for embedding.
        cache_dir: Optional directory for caching embeddings.
        max_length: Tokenisation max length.
        batch_size: Texts per forward pass.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        cache_dir: Optional[str] = None,
        max_length: int = 128,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        import torch

        logger.info("Loading embedding model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._torch = torch

    def _cache_key(self, texts: List[str]) -> str:
        payload = json.dumps({"model": self.model_name, "texts": texts}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            Float32 array of shape (len(texts), hidden_size).
        """
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{self._cache_key(texts)}.npy"
            if cache_file.exists():
                logger.debug("Cache hit for %d embeddings", len(texts))
                return np.load(str(cache_file))

        self._load()
        all_embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            enc = self._tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                output = self._model(**enc, output_hidden_states=False)
                # Use last hidden state mean-pooled over non-padding tokens
                hidden = output.last_hidden_state  # (B, T, H)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                all_embeddings.append(pooled.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

        if self.cache_dir is not None:
            np.save(str(cache_file), embeddings)
            logger.debug("Cached %d embeddings to %s", len(texts), cache_file)

        return embeddings
