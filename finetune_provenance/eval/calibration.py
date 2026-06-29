"""Calibration report: expected calibration error (ECE) on token-level probabilities."""

import logging
import math
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def calibration_report(
    model,
    tokenizer,
    texts: List[str],
    n_bins: int = 10,
    max_length: int = 128,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE) for a language model.

    For each token, the predicted probability is compared to whether the
    model predicted the correct next token. Bins the confidences and measures
    average accuracy vs. average confidence in each bin.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: Tokenizer.
        texts: Evaluation texts.
        n_bins: Number of probability bins.
        max_length: Token truncation length.
        device: Torch device.

    Returns:
        Dict with 'ece', 'bin_accuracies', 'bin_confidences', 'bin_counts'.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bin_correct = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)

    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # (1, T, V)

        # Predict token t+1 from position t
        probs = F.softmax(logits[0, :-1, :], dim=-1)   # (T-1, V)
        true_ids = input_ids[0, 1:]                    # (T-1,)
        max_probs, pred_ids = probs.max(dim=-1)

        for prob, pred, true in zip(
            max_probs.cpu().numpy(), pred_ids.cpu().numpy(), true_ids.cpu().numpy()
        ):
            bin_idx = min(int(prob * n_bins), n_bins - 1)
            bin_correct[bin_idx] += int(pred == true)
            bin_conf[bin_idx] += float(prob)
            bin_count[bin_idx] += 1

    # Compute ECE
    total = bin_count.sum()
    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    for i in range(n_bins):
        if bin_count[i] > 0:
            acc = bin_correct[i] / bin_count[i]
            conf = bin_conf[i] / bin_count[i]
            ece += (bin_count[i] / max(total, 1)) * abs(acc - conf)
            bin_accuracies.append(round(acc, 4))
            bin_confidences.append(round(conf, 4))
        else:
            bin_accuracies.append(None)
            bin_confidences.append(None)

    result = {
        "ece": round(ece, 6),
        "n_bins": n_bins,
        "total_tokens": int(total),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_count.tolist(),
    }
    logger.info("ECE: %.4f over %d tokens", ece, total)
    return result
