#!/bin/bash
# Build FAISS index over the corpus
# Usage: ./scripts/build_index.sh

set -e

echo "Building retrieval index..."
python -c "
from src.data.corpus import build_corpus
from src.retrieval.index import build_index
from pathlib import Path

corpus = build_corpus()
print(f'Corpus size: {len(corpus)} passages')

output_path = Path('index/corpus')
output_path.parent.mkdir(exist_ok=True)
build_index(corpus, output_path)
print(f'Index saved to {output_path}')
"
echo "Done."
