"""Tests for data loading, curation, and provenance assignment."""

import json
import tempfile
from pathlib import Path

import pytest

from finetune_provenance.data.loader import load_dataset_from_jsonl
from finetune_provenance.data.curator import (
    deduplicate,
    quality_filter,
    tag_examples,
    curate_dataset,
    infer_domain,
)
from finetune_provenance.data.provenance import (
    assign_provenance_ids,
    save_provenance_store,
    load_provenance_store,
    _provenance_id,
)


# ---------- Fixtures ----------

@pytest.fixture
def sample_jsonl(tmp_path):
    data = [
        {"text": "The transformer architecture revolutionized natural language processing."},
        {"text": "Photosynthesis converts light into chemical energy in plants."},
        {"text": "The French Revolution began in 1789."},
    ]
    p = tmp_path / "data.jsonl"
    with open(p, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    return str(p)


@pytest.fixture
def sample_examples():
    return [
        {"text": "The transformer architecture revolutionized natural language processing.", "source": "arxiv"},
        {"text": "Photosynthesis converts light into chemical energy in plants.", "source": "biology"},
        {"text": "The French Revolution began in 1789.", "source": "history"},
    ]


# ---------- Loader ----------

def test_load_jsonl(sample_jsonl):
    examples = load_dataset_from_jsonl(sample_jsonl)
    assert len(examples) == 3
    assert all("text" in ex for ex in examples)


def test_load_jsonl_missing_text(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"no_text": "oops"}\n')
    with pytest.raises(ValueError, match="missing required 'text'"):
        load_dataset_from_jsonl(str(p))


def test_load_jsonl_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset_from_jsonl("/nonexistent/path.jsonl")


def test_load_jsonl_empty_lines(tmp_path):
    p = tmp_path / "sparse.jsonl"
    p.write_text('\n{"text": "hello world this is a test"}\n\n')
    examples = load_dataset_from_jsonl(str(p))
    assert len(examples) == 1


# ---------- Curator ----------

def test_deduplicate(sample_examples):
    dupes = sample_examples + [dict(sample_examples[0])]
    result = deduplicate(dupes)
    assert len(result) == len(sample_examples)


def test_quality_filter_removes_short():
    examples = [
        {"text": "hi"},  # too short
        {"text": "The transformer architecture revolutionized natural language processing models."},
    ]
    result = quality_filter(examples, min_length=10)
    assert len(result) == 1
    assert "quality_score" in result[0]


def test_quality_filter_adds_score(sample_examples):
    result = quality_filter(sample_examples)
    for ex in result:
        assert "quality_score" in ex
        assert 0.0 <= ex["quality_score"] <= 1.0


def test_tag_examples_sets_defaults():
    examples = [{"text": "Some random text about algorithms and software code."}]
    tagged = tag_examples(examples)
    assert tagged[0]["source"] == "unknown"
    assert "domain" in tagged[0]


def test_infer_domain():
    tech_text = "The algorithm runs on the computer network hardware software."
    sci_text = "The biology experiment tested the chemistry hypothesis."
    assert infer_domain(tech_text) == "technology"
    assert infer_domain(sci_text) == "science"


def test_curate_dataset(sample_examples):
    result = curate_dataset(sample_examples)
    assert len(result) > 0
    for ex in result:
        assert "domain" in ex
        assert "source" in ex


# ---------- Provenance ----------

def test_provenance_id_deterministic():
    pid1 = _provenance_id("hello world text", "source1", "domain1")
    pid2 = _provenance_id("hello world text", "source1", "domain1")
    assert pid1 == pid2


def test_provenance_id_differs_on_content():
    pid1 = _provenance_id("text A something", "src", "dom")
    pid2 = _provenance_id("text B something", "src", "dom")
    assert pid1 != pid2


def test_assign_provenance_ids(sample_examples):
    curated = curate_dataset(sample_examples)
    records = assign_provenance_ids(curated)
    assert len(records) == len(curated)
    ids = [r.provenance_id for r in records]
    assert len(set(ids)) == len(ids), "All provenance IDs should be unique"


def test_provenance_roundtrip(sample_examples, tmp_path):
    curated = curate_dataset(sample_examples)
    records = assign_provenance_ids(curated)
    store_path = str(tmp_path / "store.jsonl")
    save_provenance_store(records, store_path)
    loaded = load_provenance_store(store_path)
    assert len(loaded) == len(records)
    for orig, load in zip(records, loaded):
        assert orig.provenance_id == load.provenance_id
        assert orig.text == load.text
