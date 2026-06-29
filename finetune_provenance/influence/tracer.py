"""Top-K lineage tracer: given a model output, find the most influential training examples."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

from .embedder import Embedder
from .attributor import cosine_attribution
from ..data.provenance import ProvenanceRecord, load_provenance_store

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """A single training example with its influence score."""

    provenance_id: str
    rank: int
    score: float
    text: str
    source: str
    domain: str
    quality_score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provenance_id": self.provenance_id,
            "rank": self.rank,
            "score": round(self.score, 6),
            "text": self.text[:300] + ("..." if len(self.text) > 300 else ""),
            "source": self.source,
            "domain": self.domain,
            "quality_score": self.quality_score,
        }


class ProvenanceTracer:
    """Trace the top-K most influential training examples for any model output.

    Workflow:
        1. Load provenance records from the training run.
        2. Embed all training texts (cached).
        3. For a given output text, embed it and run cosine attribution.
        4. Return ranked AttributionResult list.

    Args:
        provenance_store_path: Path to the provenance JSONL file.
        embedder: Embedder instance (shared with training for consistency).
        top_k: Default number of attributions to return.
    """

    def __init__(
        self,
        provenance_store_path: str,
        embedder: Optional[Embedder] = None,
        top_k: int = 5,
    ):
        self.top_k = top_k
        self.records: List[ProvenanceRecord] = load_provenance_store(provenance_store_path)
        self.embedder = embedder or Embedder()
        self._corpus_embeddings: Optional[np.ndarray] = None

    def _get_corpus_embeddings(self) -> np.ndarray:
        if self._corpus_embeddings is None:
            texts = [r.text for r in self.records]
            logger.info("Embedding %d training examples for attribution", len(texts))
            self._corpus_embeddings = self.embedder.embed(texts)
        return self._corpus_embeddings

    def trace(self, output_text: str, top_k: Optional[int] = None) -> List[AttributionResult]:
        """Find the top-K training examples most influential for the given output.

        Args:
            output_text: The model output to explain.
            top_k: Override default top_k.

        Returns:
            Ranked list of AttributionResult objects.
        """
        k = top_k if top_k is not None else self.top_k
        corpus = self._get_corpus_embeddings()
        query_emb = self.embedder.embed([output_text])[0]
        attributions = cosine_attribution(query_emb, corpus, top_k=k)

        results = []
        for rank, (idx, score) in enumerate(attributions, start=1):
            rec = self.records[idx]
            results.append(
                AttributionResult(
                    provenance_id=rec.provenance_id,
                    rank=rank,
                    score=score,
                    text=rec.text,
                    source=rec.source,
                    domain=rec.domain,
                    quality_score=rec.quality_score,
                )
            )
        return results

    def format_lineage_report(self, output_text: str, top_k: Optional[int] = None) -> str:
        """Return a human-readable lineage report for the given output."""
        results = self.trace(output_text, top_k=top_k)
        lines = [
            f"Lineage report for: {output_text[:100]!r}",
            f"Top-{len(results)} most influential training examples:",
            "",
        ]
        for r in results:
            lines.append(f"  [{r.rank}] Score={r.score:.4f} | ID={r.provenance_id[:12]}...")
            lines.append(f"       Source: {r.source} | Domain: {r.domain}")
            lines.append(f"       Text: {r.text[:120]!r}")
            lines.append("")
        return "\n".join(lines)
