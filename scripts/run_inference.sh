#!/bin/bash
# Run inference for all 12 cells
# Usage: ./scripts/run_inference.sh [--offline]

set -e

OFFLINE_FLAG=""
if [ "$1" == "--offline" ]; then
    OFFLINE_FLAG="--offline"
    echo "Running in offline mode"
fi

echo "Running all 12 cells..."
python -c "
from src.inference.cells import run_all_cells
from src.retrieval.index import load_index
from src.data import load_pubmedqa, build_corpus
from pathlib import Path

# Load data
test_data = load_pubmedqa('test')
corpus = build_corpus()
corpus_dict = {p['id']: p for p in corpus}

# Load index
index, id_mapping = load_index(Path('index/corpus'))

# Run all cells
results = run_all_cells(
    test_data=test_data,
    corpus=corpus_dict,
    index=index,
    id_mapping=id_mapping,
    adapters_dir=Path('checkpoints'),
    output_dir=Path('results'),
)

print(f'Completed {len(results)} cells')
"
echo "Done. Results saved to results/"
