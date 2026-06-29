"""Command-line interface for finetune-provenance."""

import argparse
import json
import logging
import sys


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )


def cmd_train(args):
    from .pipeline import run_training_pipeline

    result = run_training_pipeline(
        dataset_path=args.dataset,
        model_name=args.model,
        output_dir=args.output_dir,
        eval_dataset_path=args.eval_dataset,
        use_lora=args.lora,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_trace(args):
    from .pipeline import run_trace_pipeline
    from .influence.tracer import ProvenanceTracer
    from .influence.embedder import Embedder
    from pathlib import Path

    results = run_trace_pipeline(
        output_text=args.text,
        checkpoint_dir=args.checkpoint,
        top_k=args.top_k,
        model_name=args.model,
    )

    print(f"\nTop-{len(results)} influential training examples for:\n  {args.text[:120]!r}\n")
    for r in results:
        print(f"  [{r['rank']}] Score={r['score']:.4f} | ID={r['provenance_id'][:16]}...")
        print(f"       Source: {r['source']} | Domain: {r['domain']}")
        print(f"       Text: {r['text'][:120]!r}")
        print()


def cmd_eval(args):
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .data.loader import load_dataset_from_jsonl
    from .eval.metrics import compute_perplexity
    from .eval.calibration import calibration_report

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)

    examples = load_dataset_from_jsonl(args.dataset)
    texts = [e["text"] for e in examples]

    ppl = compute_perplexity(model, tokenizer, texts, max_length=args.max_length)
    cal = calibration_report(model, tokenizer, texts[:20], max_length=args.max_length)

    print(json.dumps({"perplexity": ppl, "calibration": cal}, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="python -m finetune_provenance",
        description="LLM fine-tuning with training data provenance",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # train subcommand
    p_train = sub.add_parser("train", help="Fine-tune a model with provenance tracking")
    p_train.add_argument("--dataset", required=True, help="Path to training JSONL")
    p_train.add_argument("--model", default="gpt2", help="HuggingFace model name (default: gpt2)")
    p_train.add_argument("--output-dir", default="./checkpoints", help="Output directory")
    p_train.add_argument("--eval-dataset", default=None, help="Path to eval JSONL (optional)")
    p_train.add_argument("--lora", action="store_true", help="Use PEFT LoRA adapters")
    p_train.add_argument("--epochs", type=int, default=1, help="Training epochs")
    p_train.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    p_train.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p_train.add_argument("--max-length", type=int, default=128, help="Token max length")
    p_train.set_defaults(func=cmd_train)

    # trace subcommand
    p_trace = sub.add_parser("trace", help="Trace top-K influential training examples for an output")
    p_trace.add_argument("text", help="Model output text to explain")
    p_trace.add_argument("--checkpoint", required=True, help="Checkpoint directory with provenance_store.jsonl")
    p_trace.add_argument("--top-k", type=int, default=5, help="Number of attributions to return")
    p_trace.add_argument("--model", default="gpt2", help="Embedding model name")
    p_trace.set_defaults(func=cmd_trace)

    # eval subcommand
    p_eval = sub.add_parser("eval", help="Evaluate model on a held-out set")
    p_eval.add_argument("--checkpoint", required=True, help="Model checkpoint directory")
    p_eval.add_argument("--dataset", required=True, help="Evaluation JSONL file")
    p_eval.add_argument("--max-length", type=int, default=128)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
