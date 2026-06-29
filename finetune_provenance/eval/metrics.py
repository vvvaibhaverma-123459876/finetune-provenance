"""Evaluation metrics: perplexity and ROUGE."""

import logging
import math
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
    batch_size: int = 4,
    device: str = "cpu",
) -> float:
    """Compute mean perplexity of a language model over a list of texts.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: Corresponding tokenizer.
        texts: Evaluation texts.
        max_length: Tokenisation max length.
        batch_size: Examples per batch.
        device: Torch device string.

    Returns:
        Mean perplexity (lower is better).
    """
    import torch

    model.eval()
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_loss = 0.0
    total_count = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        total_loss += outputs.loss.item() * len(batch)
        total_count += len(batch)

    mean_loss = total_loss / max(total_count, 1)
    perplexity = math.exp(mean_loss)
    logger.info("Perplexity over %d examples: %.4f", len(texts), perplexity)
    return perplexity


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        predictions: Model-generated texts.
        references: Ground-truth reference texts.

    Returns:
        Dict with rouge1, rouge2, rougeL F1 scores (0–1).
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError("rouge-score package required: pip install rouge-score") from e

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals: Dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure

    n = max(len(predictions), 1)
    averaged = {k: round(v / n, 4) for k, v in totals.items()}
    logger.info("ROUGE scores: %s", averaged)
    return averaged
