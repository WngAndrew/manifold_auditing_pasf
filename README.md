# Manifold Auditing PASF

Minimal scaffold for collecting safety-related activations, fitting shape-aware manifolds, and running evaluation experiments.

## Layout

- `requirements.txt` – frozen dependencies for data collection, model fine-tuning, and SMDS analysis.
- `configs/` – YAML templates for model, data, and SMDS hyperparameters.
- `src/data/process_prompts.py` – loads benign (`tatsu-lab/alpaca`) and harmful (`walledai/AdvBench`) prompts and exports JSONL files under `data/prompts/`.
- `src/activations/collect.py` – placeholder entry point for running the forward pass over prompts and saving intermediate activations to `data/activations/`.
- `src/models/mad_finetune.py` – script stub for fine-tuning on the MAD dataset and persisting checkpoints under `models/`.
- `src/experiments/` – orchestration scripts:
  - `run_collect.py` – ingest prompts and dump activations.
  - `run_smds.py` – run SMDS on saved activations (extend `src/smds/` once kernels/distances are defined).
  - `run_manifold_evaluation.py` – evaluate fitted manifolds on downstream benchmarks.
- `src/smds/` – future home for SMDS kernels, manifold curvature code, and custom distance functions (currently empty stubs).

## Workflow

1. Prepare prompt data with `process_prompts.py` and store benign/harmful splits in `data/prompts/`.
2. Fine-tune or load the target model via `mad_finetune.py` (configurable through `configs/model.yaml`).
3. Run `run_collect.py` to capture labeled activations into `data/activations/`.
4. execute `run_smds.py` to fit manifolds across varying values of k and save them under `results/manifolds/`.
5. Use `run_manifold_evaluation.py` to score manifolds on harm-detection benchmarks, writing metrics to `results/metrics/`.

## Getting Started

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Populate the data/model config files, then follow the workflow above to iterate on SMDS experiments.
