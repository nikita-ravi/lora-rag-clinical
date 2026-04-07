#!/bin/bash
# Reproduce all results from scratch
# Usage: ./scripts/reproduce.sh
#
# This script assumes:
# - BioASQ data is available at BIOASQ_DATA_PATH
# - Trained adapters are available in checkpoints/
# - Environment is set up (.env configured)
#
# To run training from scratch, use the Kaggle notebooks.

set -e

# Check that we're inside the project venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: not in venv. Run 'source .venv/bin/activate' first."
    exit 1
fi

echo "=========================================="
echo "Reproducing paper results"
echo "=========================================="

# Step 1: Verify environment
echo ""
echo "Step 1: Verifying environment..."
python -c "
from dotenv import load_dotenv
import os
load_dotenv()

required = ['BIOASQ_DATA_PATH']
missing = [k for k in required if not os.getenv(k)]
if missing:
    raise ValueError(f'Missing required env vars: {missing}')
print('Environment OK')
"

# Step 2: Build corpus and index
echo ""
echo "Step 2: Building index..."
./scripts/build_index.sh

# Step 3: Evaluate retrieval
echo ""
echo "Step 3: Evaluating retrieval quality..."
python -c "
from src.retrieval.eval_retrieval import evaluate_retrieval
from src.retrieval.index import load_index
from src.data import load_pubmedqa, build_corpus
from pathlib import Path

dev_data = load_pubmedqa('dev')
corpus = build_corpus()
corpus_dict = {p['id']: p for p in corpus}
index, id_mapping = load_index(Path('index/corpus'))

metrics = evaluate_retrieval(dev_data, index, id_mapping, corpus_dict)
print(f'Recall@5: {metrics[\"overall\"][\"recall@5\"]:.3f}')
print(f'MRR: {metrics[\"overall\"][\"mrr\"]:.3f}')
print(f'nDCG@5: {metrics[\"overall\"][\"ndcg@5\"]:.3f}')
"

# Step 4: Run inference
echo ""
echo "Step 4: Running inference..."
./scripts/run_inference.sh

# Step 5: Compute metrics
echo ""
echo "Step 5: Computing evaluation metrics..."
python -c "
from src.eval.metrics import compute_metrics
from src.eval.interaction_test import test_interaction_effect
import json
from pathlib import Path

results_dir = Path('results')
results = {}
for f in results_dir.glob('*.json'):
    with open(f) as fp:
        results[f.stem] = json.load(fp)

# Compute interaction test
interaction = test_interaction_effect(results)
print('Interaction test results:')
print(f'  LoRA-B interaction: {interaction[\"lora_b_interaction\"]:.2f}')
print(f'  LoRA-A\\' interaction: {interaction[\"lora_a_prime_interaction\"]:.2f}')
print(f'  Difference: {interaction[\"difference\"]:.2f}')
print(f'  p-value: {interaction[\"p_value\"]:.4f}')
"

# Step 6: Generate tables and figures
echo ""
echo "Step 6: Generating tables and figures..."
python -c "
from src.analysis.tables import generate_main_table
from src.analysis.figures import plot_accuracy_heatmap, plot_interaction
import json
from pathlib import Path

results_dir = Path('results')
results = {}
for f in results_dir.glob('*.json'):
    with open(f) as fp:
        results[f.stem] = json.load(fp)

# Generate outputs
output_dir = Path('paper/figures')
output_dir.mkdir(exist_ok=True)

plot_accuracy_heatmap(results, output_dir / 'accuracy_heatmap.pdf')
plot_interaction(results, output_dir / 'interaction.pdf')

table = generate_main_table(results)
with open('paper/tables/main_results.tex', 'w') as f:
    f.write(table)

print('Tables and figures generated')
"

echo ""
echo "=========================================="
echo "Reproduction complete!"
echo "Results: results/"
echo "Figures: paper/figures/"
echo "Tables: paper/tables/"
echo "=========================================="
