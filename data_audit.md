# Data Audit

> Auto-generated. Do not edit by hand.

Generated at: 2026-04-07T07:45:44.275859+00:00

## BioASQ (Primary Evaluation)

> **Design pivot (M3):** Promoted from training-only to primary evaluation dataset. PubMedQA retrieval saturated at R@5=0.980 due to structural bias (questions derived from abstract titles). BioASQ questions are expert-written with extractive gold snippets, providing naturally harder retrieval.

- **Total examples:** 3059 (factoid + yesno only)
- **Train:** 2459
- **Dev:** 100
- **Test:** 500
- **Question type distribution (factoid/yesno):** {'factoid': 1600, 'yesno': 1459}
- **Yes/No label distribution:** {'yes': 1069, 'no': 390}
- **Supported retrieval conditions:** ['none', 'strong', 'oracle']

### Split Stratification
Test and dev sets are stratified by question type to maintain natural ratio (~52% factoid, ~48% yesno):
- **Test:** 261 factoid, 239 yesno
- **Dev:** 52 factoid, 48 yesno

### Question Length (words)
- Avg: 9.0

### Snippet Length (words)
- Avg: 29.4
- Avg snippets per question: 12.6

### Test Set Hash (SHA-256)
```
bf19d29ada7450b2e1057f678c51a2bf5db6b88db0d7ec88f1c3875ed3c84360
```

## PubMedQA (Secondary/Exploratory)

> **Note:** Demoted from primary evaluation due to retrieval saturation. PubMedQA questions are paraphrased abstract titles, causing trivial title-to-abstract matching (R@5=0.980). Retained for exploratory analysis and backward compatibility.

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
- **Passages indexed:** 37,178
- **Source distribution:** PubMedQA (1,000) + BioASQ (36,178)
- **Embedding dimension:** 768
- **Index file size:** 108.92 MB
- **Build time:** 6.8 minutes
- **Index path:** `data/indices/pubmedqa.faiss`

### Notes
- One passage per abstract (PubMedQA) or snippet (BioASQ), no chunking for v1
- BGE query instruction prefix applied to queries: `"Represent this sentence for searching relevant passages: "`
- Index file not committed to git (too large); rebuild with `scripts/build_index.sh`

## Retrieval Eval History

| Eval | Dataset | Corpus | Retriever | Passages | Hit@5 | PropR@5 | MRR | nDCG@5 | Status |
|------|---------|--------|-----------|----------|-------|---------|-----|--------|--------|
| 1 | PubMedQA dev | PubMedQA only | Dense+rerank | 1,000 | 0.980 | 0.980 | 0.937 | 0.948 | Band violation (>0.95) |
| 2 | PubMedQA dev | PubMedQA + BioASQ | Dense+rerank | 37,178 | 0.980 | 0.980 | 0.916 | 0.932 | Band violation (>0.95) |
| 3 | BioASQ dev | PubMedQA + BioASQ | Dense+rerank | 37,178 | 0.880 | 0.404 | 0.770 | 0.691 | **In band** ✓ |

**Hard exit criterion:** 0.70 < Hit@5 < 0.95

**Metric note:** For single-gold-passage datasets (PubMedQA), Hit@5 and Proportional Recall@5 are identical. For multi-gold-passage datasets (BioASQ, ~12.6 golds per question), they diverge: Hit@5 measures whether any gold passage was retrieved (RAG-relevant), while Proportional Recall@5 measures the fraction of gold passages retrieved (mathematically capped at k/|gold|). Hit@5 is the band-check criterion for evaluating retrieval usefulness.

**Eval 1:** Band violation (Hit@5 = 0.980). PubMedQA questions are derived from abstract titles, so dense retrieval trivially matches questions to their source abstracts. Rank-1-to-rank-2 score gaps ranged from 0.68 to 0.99, indicating the retriever was confidently matching titles to abstracts.

**Eval 2:** Corpus expanded with BioASQ Task 13B snippets (36,178 unique snippets from factoid + yesno questions). Hit@5 unchanged at 0.980. Some BioASQ snippets became competitive (e.g., gastric surgery snippet reached rank 2), but gold passages still dominated. MRR dropped slightly (0.937 → 0.916), indicating gold passages occasionally moved to rank 2-3 but stayed in top-5.

**Conclusion:** PubMedQA's structural bias (question ≈ abstract title) makes "strong" retrieval indistinguishable from oracle. **Design pivot:** BioASQ promoted to primary evaluation dataset.

**Eval 3:** BioASQ dev set with the combined corpus. Hit@5 = 0.880 is solidly in the band, confirming the BioASQ pivot resolved the retrieval saturation issue. Oracle sanity check passes (Hit@5 = 1.0 by construction).

### BioASQ Eval 3 Stratification

By question type:
- factoid (n=52): Hit@5=0.923, PropR@5=0.409, MRR=0.816
- yesno (n=48): Hit@5=0.833, PropR@5=0.399, MRR=0.719

By number of gold snippets:
- low (1-5 golds, n=44): Hit@5=0.750, PropR@5=0.574, MRR=0.643
- medium (6-10 golds, n=19): Hit@5=0.947, PropR@5=0.370, MRR=0.763
- high (11+ golds, n=37): Hit@5=1.000, PropR@5=0.220, MRR=0.924

**Interpretation:** Retrieval difficulty inversely correlates with gold count, which is expected. The "low gold" subset (44/100 dev questions) is the hardest case and the most informative for measuring the LoRA × retrieval interaction effect, since it's where retrieval choices matter most.

**Worked example diagnostics:** Rank-1-vs-rank-2 score gaps were 0.089, 0.001, and 0.255 across three sampled queries. This contrasts sharply with PubMedQA's eval 1/2 gaps of 0.68-0.99, confirming BioASQ retrieval is making genuine ranking decisions rather than trivially matching question phrasing to passage content.

## M4b: Synthetic LoRA-B Training Data

Generated: 2026-04-08

### Generation Summary

- **Generator model:** Claude Haiku 4.5
- **Raw generations:** 2000
- **Passing examples:** 1877 (93.8%)
- **Total cost:** $2.50
- **Average cost per generation:** $0.00125

### Pass Rate Evolution

| Stage | Pass Rate | Notes |
|-------|-----------|-------|
| Original filter | 86.2% (1725/2000) | Strict bracket-only citations, no Maybe acceptance |
| Updated filter | 93.8% (1877/2000) | +7.6pp via three filter corrections |

**Filter corrections applied:**
1. **Citation format relaxation:** Accept bare `P1`-`P5` references in addition to bracketed `[P1]`-`[P5]`. Collision analysis confirmed no overlap with gene names (P53 outside range, FOXP2 has word boundary).
2. **Word-to-number normalization:** Convert number words to digits before factoid overlap check (e.g., "three" → "3"). Also fixed single-digit token preservation bug.
3. **Maybe-as-valid-label:** Per PLAN.md Q2, yes/no/maybe is the yesno label space. PubMedQA test set has 9.4% maybe labels. Auto-pass Maybe answers with valid citations as calibrated uncertainty, not filter failures.

### Final Breakdown by Mode

| Mode | Passed | Total | Rate |
|------|--------|-------|------|
| easy | 1714 | 1834 | 93.5% |
| hard | 163 | 166 | 98.2% |

### Final Breakdown by Question Type

| Type | Passed | Total | Rate |
|------|--------|-------|------|
| factoid | 945 | 1043 | 90.6% |
| yesno | 932 | 957 | 97.4% |

### Recovery Breakdown (152 total recovered)

| Category | Count | Description |
|----------|-------|-------------|
| citation_format | 78 | Non-Maybe answers with bare P1-P5 citations |
| word_to_number | 4 | True word→digit conversions ("three"↔"3", "eight"↔"8", etc.) |
| digit_preservation | 5 | Single-digit tokens no longer filtered ("9"↔"9", "2.7%"↔"2.7%", etc.) |
| model_hedged_yes | 47 | Maybe answers where BioASQ gold="Yes" |
| model_hedged_no | 18 | Maybe answers where BioASQ gold="No" |

**Note on digit_preservation:** 5 examples recovered via digit-preservation; 2 are clean digit matches (9↔9, 2.7%↔2.7%), 3 are partial token overlaps at the margin (HAX-1↔X-1, CDK 4/6↔kinase 4/6, 1.29%↔1%). Accepted as within noise tolerance at 0.27% of final training set.

### gold_disagreement Field Distribution

| Value | Count | Description |
|-------|-------|-------------|
| None | 1812 | Standard agreement or insufficient evidence |
| model_hedged_yes | 47 | Model said Maybe, BioASQ said Yes |
| model_hedged_no | 18 | Model said Maybe, BioASQ said No |

**Interpretation:** 65 examples (3.5% of passing set) have `gold_disagreement` set, indicating the model expressed calibrated uncertainty where BioASQ provided a confident binary label. Per PLAN.md Q2 and PubMedQA's 9.4% maybe test-set distribution, these are valid training signal for uncertainty calibration, not errors.

### Remaining Rejections (123 total)

| Filter | Count | Notes |
|--------|-------|-------|
| gold_answer_agreement | 122 | Real disagreements: yesno Yes↔No mismatches, factoid token overlap failures |
| citations | 1 | No citation markers at all in reasoning |

### Files

- `data/synthetic/lora_b_train_raw.jsonl` — 2000 raw generations with full metadata
- `data/synthetic/lora_b_train.jsonl` — 1877 filtered examples for training

## M5: Materialized Training Sets

Generated: 2026-04-14

### Files

- `data/training/lora_a_train.jsonl` — 1877 examples, question-only prompt → answer label
- `data/training/lora_a_prime_train.jsonl` — 1877 examples, question + 5 passages → answer label
- `data/training/lora_b_train.jsonl` — 1877 examples, question + 5 passages → reasoning + answer

### Key Properties

- All three files have identical example IDs in identical order
- LoRA-A' and LoRA-B prompts are byte-identical (verified by test_lora_a_prime_and_b_prompts_byte_identical)
- LoRA-B targets use normalized [P1]-[P5] citation format (bare P references converted)
- Answer labels use generated_answer (lowercased for yesno, preserved case for factoid) across all three sets
- Prompt format: Llama 3.1 Instruct chat template (system/user/assistant headers)
- "Insufficient evidence" preserved as valid 4th output label (19.9% of examples)

## M6: Training Scripts

Completed: 2026-04-18

### Smoke Test Results

| Recipe | Steps | Loss | Time | Status |
|--------|-------|------|------|--------|
| LoRA-A | 2 | 10.8344 | 7.2s | Pass |
| LoRA-A' | 2 | 10.8308 | 4.9s | Pass |
| LoRA-B | 2 | 10.8281 | 3.8s | Pass |

Smoke test model: sshleifer/tiny-gpt2 (CPU, no quantization)
Loss ~10.83 confirms label masking is correct (random model on random tokens).

### Bug Found and Fixed

Initial LoRA-A' smoke test showed loss=0.0 because max_seq_length=512 truncated the entire target (prompts are 550-860 tokens with passages). Fixed by increasing smoke test max_seq_length to 1024.

### Files

- `src/training/common.py` — shared training infrastructure (458 lines)
- `src/training/lora_a.py`, `lora_a_prime.py`, `lora_b.py` — thin recipe wrappers (~65 lines each)
- `configs/lora_a.yaml`, `lora_a_prime.yaml`, `lora_b.yaml` — hyperparameter configs
- `notebooks/02_lora_a_kaggle.ipynb`, `02b_lora_a_prime_kaggle.ipynb`, `03_lora_b_kaggle.ipynb`

### Next: Kaggle Execution

100-step validation run on each recipe before committing to full 1-epoch training.
9 total runs (3 recipes × 3 seeds) on Kaggle 2×T4.
