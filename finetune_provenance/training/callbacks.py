"""HuggingFace Trainer callbacks for provenance and experiment logging."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


from transformers import TrainerCallback


class ProvenanceCallback(TrainerCallback):
    """Trainer callback that logs per-step provenance metadata.

    Implements the HuggingFace TrainerCallback interface.
    Records which provenance IDs were seen in each training step.
    """

    def __init__(self, provenance_ids: List[str], log_dir: str = "./logs"):
        """
        Args:
            provenance_ids: Ordered list of provenance IDs matching the training dataset.
            log_dir: Directory where provenance step logs are written.
        """
        self.provenance_ids = provenance_ids
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._step_log: List[Dict[str, Any]] = []

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each optimizer step."""
        step = state.global_step
        # Record the provenance IDs that would have been in this step's batch.
        # Exact per-sample tracking requires DataLoader integration; here we log
        # the step number and total examples seen as a provenance breadcrumb.
        entry = {
            "step": step,
            "epoch": state.epoch,
            "total_examples_seen": step * getattr(args, "per_device_train_batch_size", 1),
        }
        self._step_log.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        """Flush the step log at the end of training."""
        out_path = self.log_dir / "provenance_steps.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in self._step_log:
                f.write(json.dumps(entry) + "\n")
        logger.info("Provenance step log saved to %s (%d entries)", out_path, len(self._step_log))

    def get_step_log(self) -> List[Dict[str, Any]]:
        """Return the in-memory step log."""
        return list(self._step_log)
