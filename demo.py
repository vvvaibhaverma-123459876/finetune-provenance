"""Demo: Fine-tune GPT-2 on sample data and trace training data influence."""
import tempfile
from pathlib import Path
from finetune_provenance.data.loader import load_dataset_from_jsonl
from finetune_provenance.data.curator import curate_dataset
from finetune_provenance.data.provenance import assign_provenance_ids, save_provenance_store
from finetune_provenance.training.trainer import ProvenanceTrainer
from finetune_provenance.influence.tracer import ProvenanceTracer

print("=== Fine-Tune Provenance Demo ===\n")

raw = load_dataset_from_jsonl("example_data/train.jsonl")
print(f"Loaded {len(raw)} training examples")

curated = curate_dataset(raw)
print(f"After curation: {len(curated)} examples remain")

records = assign_provenance_ids(curated)
print(f"Assigned provenance IDs to {len(records)} examples\n")

with tempfile.TemporaryDirectory() as tmpdir:
    prov_path = f"{tmpdir}/provenance_store.jsonl"
    save_provenance_store(records, prov_path)

    trainer = ProvenanceTrainer(
        model_name="gpt2",
        train_records=records,
        output_dir=tmpdir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
    )
    metrics = trainer.train()
    print(f"Training complete — loss: {metrics.get('train_loss', 'N/A')}")

    tracer = ProvenanceTracer(prov_path)
    query = "The capital of France is Paris"
    results = tracer.trace(query, top_k=3)
    print(f"\nTop 3 training examples most similar to:\n  '{query}'\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [score={r.score:.3f}] {r.text[:65]}...")
        print(f"       id: {r.provenance_id[:16]}...")

print("\nDemo complete.")
