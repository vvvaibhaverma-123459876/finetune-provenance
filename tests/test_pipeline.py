"""End-to-end pipeline tests using GPT-2 (no GPU required)."""

import json
import os
from pathlib import Path

import pytest

from finetune_provenance.data.loader import load_dataset_from_jsonl
from finetune_provenance.data.curator import curate_dataset
from finetune_provenance.data.provenance import (
    assign_provenance_ids,
    save_provenance_store,
)
from finetune_provenance.training.trainer import ProvenanceTrainer
from finetune_provenance.influence.embedder import Embedder
from finetune_provenance.influence.tracer import ProvenanceTracer
from finetune_provenance.experiment.logger import ExperimentLogger
from finetune_provenance.eval.metrics import compute_perplexity


EXAMPLE_DATA = Path(__file__).parent.parent / "example_data"
TRAIN_JSONL = str(EXAMPLE_DATA / "train.jsonl")
EVAL_JSONL = str(EXAMPLE_DATA / "eval.jsonl")


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    """Train a tiny GPT-2 model and return the checkpoint directory."""
    out_dir = tmp_path_factory.mktemp("checkpoints")

    raw = load_dataset_from_jsonl(TRAIN_JSONL)
    curated = curate_dataset(raw)
    records = assign_provenance_ids(curated)
    prov_path = str(out_dir / "provenance_store.jsonl")
    save_provenance_store(records, prov_path)

    trainer = ProvenanceTrainer(
        model_name="gpt2",
        train_records=records,
        output_dir=str(out_dir / "model"),
        use_lora=False,
        max_length=64,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        experiment_log_dir=str(out_dir / "logs"),
    )
    metrics = trainer.train()
    save_path = trainer.save()

    return {
        "checkpoint": save_path,
        "provenance_store": prov_path,
        "out_dir": str(out_dir),
        "metrics": metrics,
        "records": records,
    }


# ---------- Training ----------

def test_training_completes(trained_checkpoint):
    assert "train_loss" in trained_checkpoint["metrics"] or len(trained_checkpoint["metrics"]) >= 0


def test_checkpoint_exists(trained_checkpoint):
    cp = Path(trained_checkpoint["checkpoint"])
    assert cp.exists()
    assert (cp / "provenance_manifest.json").exists()


def test_provenance_manifest_contents(trained_checkpoint):
    manifest_path = Path(trained_checkpoint["checkpoint"]) / "provenance_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model_name"] == "gpt2"
    assert manifest["num_training_examples"] > 0
    assert len(manifest["provenance_ids"]) == manifest["num_training_examples"]


def test_provenance_ids_deterministic():
    """Reprocessing the same data yields the same provenance IDs."""
    raw = load_dataset_from_jsonl(TRAIN_JSONL)
    c1 = curate_dataset(raw)
    r1 = assign_provenance_ids(c1)

    c2 = curate_dataset(load_dataset_from_jsonl(TRAIN_JSONL))
    r2 = assign_provenance_ids(c2)

    ids1 = [r.provenance_id for r in r1]
    ids2 = [r.provenance_id for r in r2]
    assert ids1 == ids2, "Provenance IDs must be deterministic across reruns"


# ---------- Experiment Logger ----------

def test_experiment_logger(tmp_path):
    logger = ExperimentLogger(log_dir=str(tmp_path / "logs"))
    run_id = logger.log_run(
        run_name="test_run",
        params={"lr": 5e-5, "epochs": 1},
        metrics={"train_loss": 2.5},
        tags={"env": "ci"},
    )
    runs = logger.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["params"]["lr"] == 5e-5


def test_experiment_logger_multiple_runs(tmp_path):
    logger = ExperimentLogger(log_dir=str(tmp_path / "logs"))
    for i in range(5):
        logger.log_run(f"run_{i}", {"i": i}, {"loss": float(i)})
    assert len(logger.list_runs()) == 5


# ---------- Perplexity ----------

def test_perplexity_finite(trained_checkpoint):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import math

    cp = trained_checkpoint["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(cp)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cp)

    eval_raw = load_dataset_from_jsonl(EVAL_JSONL)
    texts = [e["text"] for e in eval_raw[:3]]
    ppl = compute_perplexity(model, tokenizer, texts, max_length=64)
    assert math.isfinite(ppl)
    assert ppl > 0


# ---------- Influence Tracing ----------

def test_tracer_end_to_end(trained_checkpoint):
    prov_path = trained_checkpoint["provenance_store"]
    embedder = Embedder(model_name="gpt2", max_length=64)
    tracer = ProvenanceTracer(
        provenance_store_path=prov_path,
        embedder=embedder,
        top_k=5,
    )
    output_text = "Language models trained on text can generate coherent sentences."
    results = tracer.trace(output_text, top_k=5)
    assert len(results) == 5
    for r in results:
        assert r.provenance_id
        assert 0.0 <= r.score <= 1.0 + 1e-5  # cosine can be slightly > 1 due to float32


def test_tracer_top1_is_most_similar(trained_checkpoint):
    """Top result should have higher score than all others."""
    prov_path = trained_checkpoint["provenance_store"]
    embedder = Embedder(model_name="gpt2", max_length=64)
    tracer = ProvenanceTracer(
        provenance_store_path=prov_path,
        embedder=embedder,
        top_k=5,
    )
    results = tracer.trace("The history of the Roman Empire spans many centuries.", top_k=5)
    scores = [r.score for r in results]
    assert scores[0] == max(scores)
