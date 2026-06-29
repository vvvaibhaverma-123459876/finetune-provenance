"""Load training data from JSONL files or HuggingFace Hub."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_dataset_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load examples from a JSONL file.

    Each line must be a JSON object with at least a "text" field.
    Optional fields: source, domain, quality, metadata.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of example dicts.
    """
    examples = []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}")
            if "text" not in example:
                raise ValueError(f"Example on line {line_num} missing required 'text' field")
            examples.append(example)

    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def load_dataset_from_hub(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load examples from a HuggingFace Hub dataset.

    Args:
        dataset_name: HuggingFace dataset name (e.g. 'wikitext').
        split: Dataset split to load.
        text_column: Column name containing the text.
        max_examples: Maximum number of examples to load (None = all).

    Returns:
        List of example dicts with at least a 'text' field.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets package required: pip install datasets") from e

    logger.info("Loading dataset '%s' split='%s' from HuggingFace Hub", dataset_name, split)
    hf_dataset = load_dataset(dataset_name, split=split)

    if text_column not in hf_dataset.column_names:
        available = hf_dataset.column_names
        raise ValueError(
            f"Column '{text_column}' not found in dataset. Available: {available}"
        )

    examples = []
    for i, row in enumerate(hf_dataset):
        if max_examples is not None and i >= max_examples:
            break
        text = row[text_column]
        if not isinstance(text, str) or not text.strip():
            continue
        example: Dict[str, Any] = {"text": text, "source": dataset_name}
        # Carry over any additional metadata columns
        for col in hf_dataset.column_names:
            if col != text_column:
                example.setdefault("metadata", {})[col] = row[col]
        examples.append(example)

    logger.info("Loaded %d examples from HuggingFace dataset '%s'", len(examples), dataset_name)
    return examples
