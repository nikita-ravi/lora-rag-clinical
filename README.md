# LoRA × Retrieval Interaction on Clinical Evidence Interpretation

Does LoRA fine-tuning the generator in a clinical RAG system substitute for, or complement, retrieval quality — and does the answer depend on what the LoRA was trained on?

## Research Question

We hypothesize that the *type* of LoRA training determines the shape of the interaction with retrieval:

- **LoRA-A** (Q→A): Acts as a *substitute* for retrieval. Gains shrink as retrieval quality improves.
- **LoRA-A'** (Q+passages→A): Control condition. Same input as LoRA-B but without reasoning targets.
- **LoRA-B** (RAFT-style): Acts as a *complement* to retrieval. Gains grow or hold steady as retrieval quality improves.

The mechanistic claim: fine-tuning on retrieved-context inputs with reasoning targets teaches the model to *read* clinical evidence, and that skill only pays off when there is good evidence to read.

## Setup

**All commands must be run inside the project venv.** If `echo $VIRTUAL_ENV` is empty, run `source .venv/bin/activate` first.

```bash
# Create and activate venv (Python 3.11 recommended)
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your paths and API keys
```

## Quick Start

```bash
# Activate venv (required for all commands)
source .venv/bin/activate

# Run tests
pytest

# Build retrieval index
./scripts/build_index.sh

# Run full experiment (after training)
./scripts/run_inference.sh

# Reproduce paper results
./scripts/reproduce.sh
```

## Project Structure

```
├── configs/           # YAML configuration files
├── src/
│   ├── data/          # Dataset loaders and splits
│   ├── retrieval/     # Dense retrieval + reranking
│   ├── training/      # LoRA training recipes
│   ├── inference/     # Generation and cell runner
│   ├── eval/          # Metrics and statistical tests
│   ├── annotation/    # Error annotation tool
│   └── analysis/      # Tables and figures
├── notebooks/         # Exploration and Kaggle notebooks
├── scripts/           # Shell scripts for running experiments
├── tests/             # Unit tests
├── paper/             # LaTeX source
└── results/           # Experiment outputs
```

## Experimental Design

**12-cell design:** 4 model conditions × 3 retrieval conditions

| Model | none | strong | oracle |
|-------|------|--------|--------|
| base  | cell | cell   | cell   |
| LoRA-A | cell | cell   | cell   |
| LoRA-A' | cell | cell   | cell   |
| LoRA-B | cell | cell   | cell   |

Each LoRA cell is run with 3 random seeds. See `PLAN.md` for full details.

## Datasets

- **BioASQ Task B** (training): ~3,000 factoid + yes/no questions with gold snippets
- **PubMedQA pqa_labeled** (evaluation): ~1,000 yes/no/maybe questions with gold abstracts
- **MIRAGE subset** (external validation): ~500 examples, held-out

## Requirements

- Python 3.10+
- CUDA-capable GPU (for training) or free Kaggle/Colab account
- ~5GB disk space for data and indices

## License

MIT

## Citation

```bibtex
@article{author2025lora,
  title={LoRA × Retrieval Interaction on Clinical Evidence Interpretation},
  author={Author},
  year={2025}
}
```
