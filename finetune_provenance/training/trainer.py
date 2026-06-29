"""HuggingFace Trainer wrapper with provenance and experiment logging."""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class ProvenanceTrainer:
    """Wraps HuggingFace Trainer to add provenance and experiment tracking.

    Usage::

        trainer = ProvenanceTrainer(
            model_name="gpt2",
            train_records=records,
            output_dir="./checkpoints",
        )
        trainer.train()
        trainer.save()
    """

    def __init__(
        self,
        model_name: str,
        train_records,
        output_dir: str = "./checkpoints",
        eval_records=None,
        use_lora: bool = False,
        max_length: int = 128,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 2,
        learning_rate: float = 5e-5,
        logging_steps: int = 10,
        save_steps: int = 50,
        experiment_log_dir: str = "./logs",
    ):
        self.model_name = model_name
        self.train_records = train_records
        self.eval_records = eval_records
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_lora = use_lora
        self.max_length = max_length
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.learning_rate = learning_rate
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.experiment_log_dir = Path(experiment_log_dir)
        self.experiment_log_dir.mkdir(parents=True, exist_ok=True)

        self._trainer = None
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model and tokenizer: %s", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        if self.use_lora:
            from .lora_config import build_lora_model
            model = build_lora_model(model)

        self._tokenizer = tokenizer
        self._model = model
        return model, tokenizer

    def _build_hf_dataset(self, records, tokenizer):
        from torch.utils.data import Dataset as TorchDataset

        texts = [r.text for r in records]

        class _TextDataset(TorchDataset):
            def __init__(self, texts, tokenizer, max_length):
                self.encodings = tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )

            def __len__(self):
                return self.encodings["input_ids"].shape[0]

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = item["input_ids"].clone()
                return item

        return _TextDataset(texts, tokenizer, self.max_length)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run the training loop.

        Returns:
            Training metrics dict.
        """
        from transformers import TrainingArguments, Trainer

        from .callbacks import ProvenanceCallback
        from ..experiment.logger import ExperimentLogger

        model, tokenizer = self._load_model_and_tokenizer()
        train_dataset = self._build_hf_dataset(self.train_records, tokenizer)
        eval_dataset = None
        if self.eval_records:
            eval_dataset = self._build_hf_dataset(self.eval_records, tokenizer)

        provenance_ids = [r.provenance_id for r in self.train_records]
        prov_callback = ProvenanceCallback(
            provenance_ids=provenance_ids,
            log_dir=str(self.experiment_log_dir),
        )

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_strategy="epoch" if eval_dataset else "no",
            report_to="none",
            use_cpu=True,  # CPU by default; GPU picked up automatically when available
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[prov_callback],
        )

        logger.info("Starting training: %d examples, %d epochs", len(train_dataset), self.num_train_epochs)
        train_result = trainer.train()
        self._trainer = trainer

        metrics = train_result.metrics
        exp_logger = ExperimentLogger(log_dir=str(self.experiment_log_dir))
        exp_logger.log_run(
            run_name=f"train_{self.model_name.replace('/', '_')}",
            params={
                "model": self.model_name,
                "use_lora": self.use_lora,
                "num_train_epochs": self.num_train_epochs,
                "batch_size": self.per_device_train_batch_size,
                "learning_rate": self.learning_rate,
                "num_examples": len(self.train_records),
            },
            metrics=metrics,
        )

        return metrics

    def save(self, path: Optional[str] = None) -> str:
        """Save model + tokenizer to disk.

        Returns the save path.
        """
        save_path = Path(path) if path else self.output_dir / "final"
        save_path.mkdir(parents=True, exist_ok=True)
        if self._trainer is None:
            raise RuntimeError("Call train() before save()")
        self._trainer.save_model(str(save_path))
        self._tokenizer.save_pretrained(str(save_path))

        # Save provenance manifest alongside the checkpoint
        prov_ids = [r.provenance_id for r in self.train_records]
        manifest = {
            "model_name": self.model_name,
            "num_training_examples": len(self.train_records),
            "provenance_ids": prov_ids,
        }
        with open(save_path / "provenance_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Model saved to %s", save_path)
        return str(save_path)
