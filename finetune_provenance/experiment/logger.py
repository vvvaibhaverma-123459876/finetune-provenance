"""JSON-based experiment logger — no external MLflow dependency."""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Append-only JSON experiment log.

    Each call to ``log_run`` appends one JSON object to
    ``<log_dir>/experiments.jsonl``.

    Example::

        el = ExperimentLogger(log_dir="./logs")
        el.log_run(
            run_name="gpt2_baseline",
            params={"lr": 5e-5, "epochs": 3},
            metrics={"train_loss": 2.1, "perplexity": 8.2},
        )
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "experiments.jsonl"

    def log_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[List[str]] = None,
    ) -> str:
        """Log a training run.

        Args:
            run_name: Human-readable run identifier.
            params: Hyperparameters and configuration.
            metrics: Numeric evaluation results.
            tags: Arbitrary key-value string tags.
            artifacts: Paths to associated files.

        Returns:
            Unique run ID.
        """
        run_id = str(uuid.uuid4())
        entry = {
            "run_id": run_id,
            "run_name": run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
            "tags": tags or {},
            "artifacts": artifacts or [],
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("Logged run '%s' (id=%s) to %s", run_name, run_id, self.log_file)
        return run_id

    def list_runs(self) -> List[Dict[str, Any]]:
        """Return all logged runs as a list of dicts."""
        if not self.log_file.exists():
            return []
        runs = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
        return runs

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single run by ID."""
        for run in self.list_runs():
            if run["run_id"] == run_id:
                return run
        return None
