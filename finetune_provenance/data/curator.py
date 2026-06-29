"""Dataset curation: deduplication, quality filtering, and domain tagging."""

import hashlib
import logging
import re
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """Return a short SHA256 hash of text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def deduplicate(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove exact-duplicate texts, keeping first occurrence."""
    seen: set = set()
    unique = []
    for ex in examples:
        h = _content_hash(ex["text"])
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    removed = len(examples) - len(unique)
    if removed:
        logger.info("Deduplication removed %d duplicates", removed)
    return unique


def quality_filter(
    examples: List[Dict[str, Any]],
    min_length: int = 10,
    max_length: int = 4096,
    min_alpha_ratio: float = 0.5,
) -> List[Dict[str, Any]]:
    """Filter examples by basic quality heuristics.

    Args:
        examples: Input examples.
        min_length: Minimum character length.
        max_length: Maximum character length.
        min_alpha_ratio: Minimum ratio of alphabetic characters.

    Returns:
        Filtered examples with a 'quality_score' field added.
    """
    filtered = []
    for ex in examples:
        text = ex["text"]
        if len(text) < min_length or len(text) > max_length:
            continue
        alpha_count = sum(c.isalpha() for c in text)
        ratio = alpha_count / max(len(text), 1)
        if ratio < min_alpha_ratio:
            continue
        ex = dict(ex)
        ex["quality_score"] = round(ratio, 4)
        filtered.append(ex)

    removed = len(examples) - len(filtered)
    if removed:
        logger.info("Quality filter removed %d low-quality examples", removed)
    return filtered


# Simple keyword-based domain tagger
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "science": ["experiment", "hypothesis", "biology", "chemistry", "physics", "research"],
    "technology": ["software", "hardware", "algorithm", "computer", "network", "code"],
    "history": ["century", "ancient", "war", "empire", "dynasty", "historical"],
    "literature": ["novel", "poem", "story", "character", "narrative", "author"],
    "general": [],
}


def infer_domain(text: str) -> str:
    """Infer a domain label from text content using keyword heuristics."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if domain == "general":
            continue
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    best_domain = max(scores, key=lambda d: scores[d], default="general")
    if scores.get(best_domain, 0) == 0:
        return "general"
    return best_domain


def tag_examples(
    examples: List[Dict[str, Any]],
    default_source: str = "unknown",
) -> List[Dict[str, Any]]:
    """Tag each example with source, domain, and quality metadata.

    Existing fields are not overwritten.
    """
    tagged = []
    for ex in examples:
        ex = dict(ex)
        ex.setdefault("source", default_source)
        ex.setdefault("domain", infer_domain(ex["text"]))
        ex.setdefault("quality_score", None)
        tagged.append(ex)
    return tagged


def curate_dataset(
    examples: List[Dict[str, Any]],
    min_length: int = 10,
    max_length: int = 4096,
    min_alpha_ratio: float = 0.5,
    default_source: str = "unknown",
) -> List[Dict[str, Any]]:
    """Full curation pipeline: dedup -> quality filter -> tag.

    Args:
        examples: Raw loaded examples.
        min_length: Min character length for quality filter.
        max_length: Max character length for quality filter.
        min_alpha_ratio: Min alphabetic ratio for quality filter.
        default_source: Default source tag if not set.

    Returns:
        Curated examples.
    """
    examples = deduplicate(examples)
    examples = quality_filter(examples, min_length, max_length, min_alpha_ratio)
    examples = tag_examples(examples, default_source)
    logger.info("Curation complete: %d examples remain", len(examples))
    return examples
