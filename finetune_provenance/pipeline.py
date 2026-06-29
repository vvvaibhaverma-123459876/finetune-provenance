"""End-to-end pipeline: load -> curate -> provenance -> train -> eval -> trace."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def run_training_pipeline(
    dataset_path: str,
    model_name: str = "gpt2",
    output_dir: str = "./checkpoints",
    eval_dataset_path: Optional[str] = None,
    use_lora: bool = False,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_length: int = 128,
    min_text_length: int = 10,
    max_text_length: int = 4096,
) -> Dict[str, Any]:
    """Full training pipeline with provenance tracking.

    Args:
        dataset_path: Path to training JSONL file.
        model_name: HuggingFace model name.
        output_dir: Checkpoint and log output directory.
        eval_dataset_path: Optional evaluation JSONL file.
        use_lora: Enable PEFT LoRA adapters.
        num_train_epochs: Training epochs.
        per_device_train_batch_size: Batch size per device.
        learning_rate: Optimizer learning rate.
        max_length: Token max length.
        min_text_length: Min char length for quality filter.
        max_text_length: Max char length for quality filter.

    Returns:
        Dict with training metrics and provenance store path.
    """
    from .data.loader import load_dataset_from_jsonl
    from .data.curator import curate_dataset
    from .data.provenance import assign_provenance_ids, save_provenance_store
    from .training.trainer import ProvenanceTrainer

    out = Path(output_dir)
    log_dir = out / "logs"

    # 1. Load
    logger.info("Loading training data from %s", dataset_path)
    raw_examples = load_dataset_from_jsonl(dataset_path)

    # 2. Curate
    curated = curate_dataset(
        raw_examples,
        min_length=min_text_length,
        max_length=max_text_length,
    )
    if not curated:
        raise ValueError("No examples remain after curation")

    # 3. Assign provenance IDs
    train_records = assign_provenance_ids(curated)
    prov_store_path = str(out / "provenance_store.jsonl")
    save_provenance_store(train_records, prov_store_path)

    # 4. Load eval if provided
    eval_records = None
    if eval_dataset_path:
        eval_raw = load_dataset_from_jsonl(eval_dataset_path)
        eval_curated = curate_dataset(eval_raw, min_length=min_text_length)
        eval_records = assign_provenance_ids(eval_curated)

    # 5. Train
    trainer = ProvenanceTrainer(
        model_name=model_name,
        train_records=train_records,
        eval_records=eval_records,
        output_dir=str(out / "model"),
        use_lora=use_lora,
        max_length=max_length,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        experiment_log_dir=str(log_dir),
    )
    metrics = trainer.train()
    save_path = trainer.save()

    return {
        "metrics": metrics,
        "provenance_store": prov_store_path,
        "checkpoint": save_path,
        "num_train_examples": len(train_records),
    }


def run_trace_pipeline(
    output_text: str,
    checkpoint_dir: str,
    top_k: int = 5,
    model_name: str = "gpt2",
) -> List[Dict[str, Any]]:
    """Trace the most influential training examples for a model output.

    Args:
        output_text: The text to explain.
        checkpoint_dir: Directory containing provenance_store.jsonl.
        top_k: Number of attributions to return.
        model_name: Model used for embedding.

    Returns:
        List of attribution dicts.
    """
    from .influence.embedder import Embedder
    from .influence.tracer import ProvenanceTracer

    prov_store = str(Path(checkpoint_dir) / "provenance_store.jsonl")
    embedder = Embedder(model_name=model_name, cache_dir=str(Path(checkpoint_dir) / ".embed_cache"))
    tracer = ProvenanceTracer(
        provenance_store_path=prov_store,
        embedder=embedder,
        top_k=top_k,
    )
    results = tracer.trace(output_text, top_k=top_k)
    return [r.to_dict() for r in results]
