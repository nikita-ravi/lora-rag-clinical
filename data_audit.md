# Data Audit

> Auto-generated. Do not edit by hand.

Generated at: 2026-04-07T07:45:44.275859+00:00

## PubMedQA (Primary Evaluation)

- **Total examples:** 1000
- **Train:** 450
- **Dev:** 50
- **Test:** 500
- **Label distribution (yes/no/maybe):** {'yes': 552, 'maybe': 110, 'no': 338}
- **Supported retrieval conditions:** ['none', 'strong', 'oracle']

### Question Length (words)
- Avg: 12.9
- Median: 13
- Max: 31

### Context Length (words)
- Avg: 200.2
- Median: 201
- Max: 398

### Test Set Hash (SHA-256)
```
6a5bd9fb2d5b771975391378ad65d7e8f8739427f1bb0ed3ed4759a04d905bef
```

## BioASQ (Primary Training)

- **Total examples:** 3059 (factoid + yesno only)
- **Train:** 2447
- **Dev:** 305
- **Test:** 307
- **Question type distribution (factoid/yesno):** {'factoid': 1600, 'yesno': 1459}
- **Yes/No label distribution:** {'yes': 1069, 'no': 390}
- **Supported retrieval conditions:** ['none', 'strong', 'oracle']

### Question Length (words)
- Avg: 9.0

### Snippet Length (words)
- Avg: 29.4
- Avg snippets per question: 12.6

## MIRAGE (External Validation)

- **Total examples:** 500 (held-out only)
- **Source distribution:** {'medmcqa': 170, 'medqa_us': 165, 'mmlu_med': 165}
- **Answer distribution (A/B/C/D):** {'B': 123, 'D': 120, 'A': 127, 'C': 130}
- **Supported retrieval conditions:** ['none', 'strong']
- **Format:** All MCQ with 4 options (A/B/C/D)

### Overall Question Length (words)
- Avg: 57.7
- Median: 25
- Max: 268

### Per-Source Statistics

**mmlu_med** (165 examples)
- Question length: avg=39.3, median=15, max=239
- Answer distribution: {'D': 52, 'C': 37, 'B': 41, 'A': 35}
- Unique subjects: 6

**medqa_us** (165 examples)
- Question length: avg=120.2, median=114, max=268
- Answer distribution: {'B': 37, 'D': 39, 'A': 44, 'C': 45}
- Unique subjects: 2

**medmcqa** (170 examples)
- Question length: avg=14.8, median=11, max=88
- Answer distribution: {'B': 45, 'D': 29, 'A': 48, 'C': 48}
- Unique subjects: 17

**Note:** No gold passages available. Supports only 'none' and 'strong' retrieval conditions.

**Subject field semantics:** The `subject` field has different semantics across sources — MMLU subset name for MMLU-Med (e.g., `professional_medicine`), USMLE step for MedQA-US (e.g., `step1`), medical specialty for MedMCQA (e.g., `Pharmacology`). Cross-source subject comparisons are not meaningful.

### Test Set Hash (SHA-256)
```
515931421bb81ea6c09ff35d15a3a1fbc60ce18b44b14e79d5b41b9ddc6ea14b
```

## Test Set Verification

Run `pytest tests/test_splits.py` to verify test set integrity and data loading.

## Retrieval Index (M3)

Built: 2026-04-07

### Index Configuration
- **Embedding model:** BAAI/bge-base-en-v1.5
- **Reranker model:** BAAI/bge-reranker-base
- **Index type:** FAISS IndexFlatIP (exact search)
- **Retrieval pipeline:** Dense top-20 → Rerank to top-5

### Index Statistics
- **Passages indexed:** [TO BE UPDATED after rebuild]
- **Source distribution:** PubMedQA (1000) + BioASQ snippets (~36k)
- **Embedding dimension:** 768
- **Index path:** `data/indices/pubmedqa.faiss`

### Notes
- One passage per abstract (PubMedQA) or snippet (BioASQ), no chunking for v1
- BGE query instruction prefix applied to queries: `"Represent this sentence for searching relevant passages: "`
- Index file not committed to git (too large); rebuild with `scripts/build_index.sh`

## Retrieval Eval History

| Eval | Corpus | Passages | Recall@5 | MRR | nDCG@5 | Status |
|------|--------|----------|----------|-----|--------|--------|
| 1 | PubMedQA only | 1,000 | 0.980 | 0.937 | 0.948 | **Band violation** (>0.95) |
| 2 | PubMedQA + BioASQ | ~37k | [pending] | [pending] | [pending] | [pending] |

**Hard exit criterion:** 0.70 < Recall@5 < 0.95

Eval 1 violated the upper band (R@5 = 0.980). PubMedQA questions are derived from abstract titles, so dense retrieval trivially matches questions to their source abstracts. Corpus expanded with BioASQ snippets to increase difficulty.
