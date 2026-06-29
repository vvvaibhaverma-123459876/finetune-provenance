# finetune-provenance

A production-grade Python pipeline for fine-tuning small language models (Phi-3-mini / GPT-2) that tracks **which training examples influenced which outputs** — full dataset-to-model lineage.

## Architecture

```
finetune_provenance/
├── data/
│   ├── loader.py        Load from JSONL or HuggingFace Hub
│   ├── curator.py       Dedup, quality filter, domain tagging
│   └── provenance.py    Deterministic SHA256 IDs per example
├── training/
│   ├── trainer.py       HF Trainer wrapper with provenance logging
│   ├── lora_config.py   PEFT LoRA auto-configuration
│   └── callbacks.py     Per-step provenance breadcrumb callback
├── influence/
│   ├── embedder.py      Mean-pooled hidden-state embeddings (cached)
│   ├── attributor.py    Cosine similarity attribution
│   └── tracer.py        Top-K lineage for any output text
├── eval/
│   ├── metrics.py       Perplexity + ROUGE
│   └── calibration.py  Expected Calibration Error (ECE)
├── experiment/
│   └── logger.py        Append-only JSON experiment log
├── pipeline.py          End-to-end train + trace pipelines
└── cli.py               Command-line interface
```

### Provenance ID design

Every training example receives a **deterministic SHA256 ID** derived from its text, source, and domain:

```python
provenance_id = sha256(json.dumps({
    "text": text,
    "source": source,
    "domain": domain
})).hexdigest()
```

Reruns on the same data always produce identical IDs — essential for reproducible attribution.

### Influence attribution

Influence is computed via **cosine similarity** between:
- The mean-pooled last hidden state of the *query output*
- The mean-pooled last hidden state of each *training example*

Both use the same base model for embedding consistency. Results are ranked and returned with provenance metadata.

## Installation

```bash
pip install -e ".[lora,dev]"
```

For GPU fine-tuning with Phi-3-mini, `peft` is included in the `lora` extra.

## Training

```bash
python -m finetune_provenance train \
    --dataset example_data/train.jsonl \
    --model gpt2 \
    --output-dir ./checkpoints \
    --epochs 1
```

For Phi-3-mini with LoRA on a GPU:

```bash
python -m finetune_provenance train \
    --dataset data/train.jsonl \
    --model microsoft/Phi-3-mini-4k-instruct \
    --output-dir ./checkpoints/phi3 \
    --lora \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-4 \
    --max-length 512
```

## Lineage tracing

After training, trace which examples most influenced any model output:

```bash
python -m finetune_provenance trace \
    "Language models trained on large corpora can generate coherent text." \
    --checkpoint ./checkpoints
```

### Example output

```
Top-5 influential training examples for:
  'Language models trained on large corpora can generate coherent text.'

  [1] Score=0.9821 | ID=a3f2c1e4b8d7...
       Source: ml_textbook | Domain: technology
       Text: 'A neural network consists of layers of interconnected nodes or neurons that process information...'

  [2] Score=0.9714 | ID=7b4e2d9f1c6a...
       Source: arxiv | Domain: technology
       Text: 'Fine-tuning a pretrained language model on domain-specific data can significantly improve performance...'

  [3] Score=0.9601 | ID=c8a1f3e2b5d4...
       Source: ai_textbook | Domain: technology
       Text: 'Natural language processing is a subfield of linguistics and artificial intelligence...'

  [4] Score=0.9432 | ID=d2e5a7c1f8b3...
       Source: ml_textbook | Domain: technology
       Text: 'Reinforcement learning is a type of machine learning where an agent learns to make decisions...'

  [5] Score=0.9287 | ID=f1b4c7e2a5d8...
       Source: ml_textbook | Domain: technology
       Text: 'Gradient descent is an optimization algorithm used to minimize a loss function...'
```

## Evaluation

```bash
python -m finetune_provenance eval \
    --checkpoint ./checkpoints/final \
    --dataset example_data/eval.jsonl
```

Outputs perplexity and Expected Calibration Error (ECE).

## Experiment tracking

All runs are appended to `<output_dir>/logs/experiments.jsonl`:

```json
{
  "run_id": "3f8a2c1e-...",
  "run_name": "train_gpt2",
  "timestamp": "2024-01-15T10:30:00Z",
  "params": {"model": "gpt2", "epochs": 1, "batch_size": 2},
  "metrics": {"train_loss": 2.43, "train_runtime": 12.5},
  "tags": {},
  "artifacts": []
}
```

## Running tests

Tests use GPT-2 and run on CPU (no GPU required):

```bash
python -m pytest tests/ -q
```

## JSONL data format

Each line must be a JSON object with a `text` field. Optional fields:

```json
{"text": "Training example text here.", "source": "arxiv", "domain": "technology"}
```

Supported domains (auto-inferred if not provided): `technology`, `science`, `history`, `literature`, `general`.
