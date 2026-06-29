"""Tests for embedding, attribution, and provenance tracing."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from finetune_provenance.influence.attributor import cosine_attribution, _normalize
from finetune_provenance.influence.embedder import Embedder
from finetune_provenance.influence.tracer import ProvenanceTracer
from finetune_provenance.data.provenance import (
    ProvenanceRecord,
    save_provenance_store,
    assign_provenance_ids,
)
from finetune_provenance.data.curator import curate_dataset


# ---------- Attributor ----------

def test_normalize():
    v = np.array([[3.0, 4.0]])
    n = _normalize(v)
    assert abs(np.linalg.norm(n[0]) - 1.0) < 1e-6


def test_cosine_attribution_returns_top_k():
    rng = np.random.default_rng(42)
    corpus = rng.standard_normal((10, 32)).astype(np.float32)
    query = rng.standard_normal(32).astype(np.float32)
    results = cosine_attribution(query, corpus, top_k=3)
    assert len(results) == 3
    # Results sorted descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_cosine_attribution_perfect_match():
    corpus = np.eye(5, dtype=np.float32)
    query = corpus[2]
    results = cosine_attribution(query, corpus, top_k=1)
    idx, score = results[0]
    assert idx == 2
    assert abs(score - 1.0) < 1e-5


def test_cosine_attribution_top_k_clipped():
    corpus = np.random.randn(3, 8).astype(np.float32)
    query = np.random.randn(8).astype(np.float32)
    results = cosine_attribution(query, corpus, top_k=10)
    assert len(results) == 3  # clipped to corpus size


# ---------- Embedder ----------

def test_embedder_shape():
    """Embedder produces vectors of consistent shape."""
    embedder = Embedder(model_name="gpt2", max_length=32, batch_size=4)
    texts = [
        "The transformer model processes sequences.",
        "Photosynthesis converts light to energy.",
        "The French Revolution changed history.",
    ]
    embs = embedder.embed(texts)
    assert embs.ndim == 2
    assert embs.shape[0] == len(texts)
    assert embs.shape[1] > 0


def test_embedder_deterministic():
    embedder = Embedder(model_name="gpt2", max_length=32)
    texts = ["Hello world this is a test sentence."]
    e1 = embedder.embed(texts)
    e2 = embedder.embed(texts)
    np.testing.assert_array_almost_equal(e1, e2)


def test_embedder_cache(tmp_path):
    embedder = Embedder(model_name="gpt2", max_length=32, cache_dir=str(tmp_path / "cache"))
    texts = ["Caching test text for embeddings."]
    e1 = embedder.embed(texts)
    # Second call should hit cache
    e2 = embedder.embed(texts)
    np.testing.assert_array_almost_equal(e1, e2)


def test_embedder_empty():
    embedder = Embedder(model_name="gpt2")
    result = embedder.embed([])
    assert result.shape[0] == 0


# ---------- Tracer ----------

@pytest.fixture
def provenance_store(tmp_path):
    examples = [
        {"text": "The transformer architecture revolutionized natural language processing and AI research.", "source": "arxiv", "domain": "technology"},
        {"text": "Photosynthesis is the biological process by which plants convert sunlight into chemical energy.", "source": "biology", "domain": "science"},
        {"text": "The French Revolution of 1789 fundamentally altered European political structures.", "source": "history", "domain": "history"},
        {"text": "Neural networks consist of layers of interconnected nodes that process information.", "source": "ml_book", "domain": "technology"},
        {"text": "Ancient Rome was one of the largest empires in the ancient world at its height.", "source": "history", "domain": "history"},
    ]
    curated = curate_dataset(examples)
    records = assign_provenance_ids(curated)
    store_path = str(tmp_path / "store.jsonl")
    save_provenance_store(records, store_path)
    return store_path, records


def test_tracer_returns_top_k(provenance_store):
    store_path, _ = provenance_store
    embedder = Embedder(model_name="gpt2", max_length=32)
    tracer = ProvenanceTracer(provenance_store_path=store_path, embedder=embedder, top_k=3)
    results = tracer.trace("Language models are trained on large text datasets.", top_k=3)
    assert len(results) == 3
    ranks = [r.rank for r in results]
    assert ranks == sorted(ranks)


def test_tracer_scores_descending(provenance_store):
    store_path, _ = provenance_store
    embedder = Embedder(model_name="gpt2", max_length=32)
    tracer = ProvenanceTracer(provenance_store_path=store_path, embedder=embedder)
    results = tracer.trace("Deep learning and neural networks for text generation.", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_tracer_attribution_fields(provenance_store):
    store_path, _ = provenance_store
    embedder = Embedder(model_name="gpt2", max_length=32)
    tracer = ProvenanceTracer(provenance_store_path=store_path, embedder=embedder)
    results = tracer.trace("Evolution of species over millions of years.")
    for r in results:
        assert r.provenance_id
        assert r.text
        assert r.source
        assert r.domain
        assert isinstance(r.score, float)


def test_tracer_format_lineage_report(provenance_store):
    store_path, _ = provenance_store
    embedder = Embedder(model_name="gpt2", max_length=32)
    tracer = ProvenanceTracer(provenance_store_path=store_path, embedder=embedder)
    report = tracer.format_lineage_report("Scientific experiments about biology.", top_k=3)
    assert "Top-3" in report
    assert "Score=" in report
    assert "Source:" in report
