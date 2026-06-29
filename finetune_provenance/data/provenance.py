"""Assign deterministic provenance IDs to training examples and persist metadata."""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _provenance_id(text: str, source: str = "", domain: str = "") -> str:
    """Deterministic SHA256 ID based on content + source + domain.

    Reproducible across reruns: same content always yields the same ID.
    """
    payload = json.dumps({"text": text, "source": source, "domain": domain}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ProvenanceRecord:
    """Metadata record for a single training example."""

    provenance_id: str
    text: str
    source: str = "unknown"
    domain: str = "general"
    quality_score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProvenanceRecord":
        return cls(
            provenance_id=d["provenance_id"],
            text=d["text"],
            source=d.get("source", "unknown"),
            domain=d.get("domain", "general"),
            quality_score=d.get("quality_score"),
            extra=d.get("extra", {}),
        )


def assign_provenance_ids(
    examples: List[Dict[str, Any]],
) -> List[ProvenanceRecord]:
    """Assign deterministic provenance IDs to a list of examples.

    Args:
        examples: Curated examples (must have 'text' field).

    Returns:
        List of ProvenanceRecord objects.
    """
    records = []
    for ex in examples:
        text = ex["text"]
        source = ex.get("source", "unknown")
        domain = ex.get("domain", "general")
        pid = _provenance_id(text, source, domain)
        extra = {k: v for k, v in ex.items() if k not in {"text", "source", "domain", "quality_score"}}
        record = ProvenanceRecord(
            provenance_id=pid,
            text=text,
            source=source,
            domain=domain,
            quality_score=ex.get("quality_score"),
            extra=extra,
        )
        records.append(record)
    logger.info("Assigned provenance IDs to %d examples", len(records))
    return records


def save_provenance_store(records: List[ProvenanceRecord], path: str) -> None:
    """Save provenance records to a JSONL file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")
    logger.info("Saved %d provenance records to %s", len(records), path)


def load_provenance_store(path: str) -> List[ProvenanceRecord]:
    """Load provenance records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ProvenanceRecord.from_dict(json.loads(line)))
    logger.info("Loaded %d provenance records from %s", len(records), path)
    return records
