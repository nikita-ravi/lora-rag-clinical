# PLAN.md — LoRA × Retrieval Interaction on Clinical Evidence Interpretation

## Design Change Log

**2025-04-07: Plan approved with additional directives**

User approved plan with answers to Q1-Q4 and additional directives. Key decisions documented below in "Resolved Questions" section.

**2025-04-07: Added LoRA-A' condition in response to methodological review**

During planning, a confound was identified: the original LoRA-A (Q→A) vs LoRA-B (Q+passages→reasoned A) comparison conflates three variables:
1. Presence of passages in the input
2. Whether the model is trained to reason over passages
3. Total token count seen during training

To isolate "trained to reason over evidence" as the key variable, we added a third LoRA condition:

- **LoRA-A'** (Q + 5 passages → A): Same passages as LoRA-B, but target is just the answer label (no reasoning). This is the "passages-as-noise" control.

The study now has:
- **PubMedQA:** 4 model conditions × 3 retrieval conditions = 12 cells (full design with oracle)
- **MIRAGE:** 4 model conditions × 2 retrieval conditions = 8 cells (none + strong only; no oracle because MMLU-Med, MedQA-US, and MedMCQA don't provide gold passages)
- **Total:** 20 cells, 50 inference passes

The cleanest test of the mechanistic claim is LoRA-B vs LoRA-A', where the only difference is whether the training target reasons over the passages.

---

## Milestones

```
[M1] Plan & Scaffold
 │
 ▼
[M2] Data Layer ──────────────────┐
 │                                │
 ▼                                ▼
[M3] Retrieval Layer         [M4] Synthetic Data Generation
 │                                │
 └────────────┬───────────────────┘
              ▼
[M5] Prompt Templates & Distractor Sampling
              │
              ▼
[M6] Training Scripts (LoRA-A, LoRA-A', LoRA-B)
              │
              ▼
[M7] Inference & Eval Pipeline
              │
              ▼
[M8] Annotation Tool
              │
              ▼
[M9] Analysis & Paper Scaffolding
              │
              ▼
[HANDOFF] Real training runs on Kaggle
```

---

## Milestone Details

### M1: Plan & Scaffold
**Estimated hours:** 3–4
**Dependencies:** None
**Deliverables:**
- This PLAN.md (done when you're reading this)
- Repository structure with empty files
- `pyproject.toml` with dependencies
- Config files with defaults
- Test stubs that pass (as skipped)
- README skeleton
- `.env.example`
- `IDEAS.md` for out-of-scope extensions

**Exit criteria:** `pytest` runs and reports all tests skipped, `uv sync` or `poetry install` succeeds.

---

### M2: Data Layer
**Estimated hours:** 8–10
**Dependencies:** M1
**Deliverables:**
- `src/data/bioasq.py` — loader for BioASQ Task B (factoid + yes/no only, exclude list/summary), reads from `BIOASQ_DATA_PATH`
- `src/data/pubmedqa.py` — loader for pqa_labeled split
- `src/data/mirage.py` — loader for MIRAGE subset (~500 examples stratified across MMLU-Med, MedQA-US, and MedMCQA; seed=42; supports only none+strong retrieval conditions — no oracle due to lack of gold passages)
- `src/data/splits.py` — locked test set mechanism with SHA-256 hash committed to repo
- `src/data/corpus.py` — unified corpus of all abstracts/snippets for indexing
- `data_audit.md` — auto-generated file reporting:
  - Total examples per dataset
  - Train/dev/test sizes
  - Label distribution
  - Average question/passage length
  - Test set hash
- Tests that verify:
  - Test set hash matches committed value
  - Loaders return expected schema
  - No overlap between train/dev/test
  - BioASQ path is in .gitignore

**Exit criteria:** Can load 5 example records from each dataset, test set hash is committed and verified, `data_audit.md` committed.

**Key decisions:**
- BioASQ = primary training source (~3,000 examples after filtering for gold snippets)
- PubMedQA pqa_labeled = primary evaluation (held-out test set)
- MIRAGE = external validity check only (never used for training/tuning)
- Exclude BioASQ list questions (different answer format complicates evaluation)

---

### M3: Retrieval Layer

**Status:** ✅ Complete (2026-04-07). Hit@5 = 0.880 on BioASQ dev set, in band.
Design pivoted from PubMedQA-primary to BioASQ-primary mid-milestone due to
PubMedQA retrieval saturation. See data_audit.md "Retrieval Eval History"
and preregistration.md "Design Change Log" for full chronology.

**Estimated hours:** 10–12
**Dependencies:** M2
**Deliverables:**
- `src/retrieval/index.py` — build FAISS index over corpus using BGE-base embeddings
- `src/retrieval/retrieve.py` — dense retrieval + BGE-reranker, returns top-5
- `src/retrieval/oracle.py` — gold passage retriever (returns gold + pads with strong retrieval)
- `src/retrieval/eval_retrieval.py` — computes Recall@5, MRR, nDCG@5, stratified by:
  - Question type (factoid vs yes/no)
  - Gold passage length (short/medium/long terciles)
- `configs/retrieval.yaml` — embedding model, reranker model, top-k settings
- Notebooks: `00_data_inspection.ipynb`, `01_retrieval_smoke_test.ipynb`

**Exit criteria:** Retrieval metrics reported on dev set. "Strong" condition must show:
- Recall@5 > 0.7 (meaningfully better than random)
- Recall@5 < 0.95 (meaningfully worse than oracle)
- No dramatic differences across stratification slices (if retrieval is 0.9 for factoid but 0.5 for yes/no, that's a confound we need to address)

If strong retrieval is too weak or too strong, we adjust the pipeline before proceeding.

---

### M4: Synthetic Data Generation
**Estimated hours:** 6–8
**Dependencies:** M2
**Deliverables:**
- `src/training/synthetic.py` — generates LoRA-B training targets using Claude Haiku 4.5
- `src/training/synthetic_prompts.py` — the generation prompt template (strict grounding format)
- `src/training/synthetic_filter.py` — filters: label match, valid citation indices, minimum length
- `data/synthetic/generation_metadata.json` — model name, version, prompt hash, filter criteria, seed
- `data/synthetic/cache/` — cached generation outputs for reproducibility
- Script to run generation and save results

**Reproducibility requirements:**
- Deterministic seed for all random operations (seed=42)
- Saved cache of all API responses so regeneration produces identical outputs
- If a reviewer asks "does your result depend on which specific synthetic examples Claude generated?", we can say "here's the cache, here's the seed, run it yourself"

**Generation prompt requirements:**
- Every claim must reference a specific passage by index (e.g., "[1]", "[3]")
- Final label must match gold label exactly
- Fixed output format: "Based on the provided evidence... [reasoning with citations]... Answer: [label]"
- Target: ~4,000 examples, budget ~$3–5

**Filtering criteria:**
- Drop if generated label ≠ gold label
- Drop if citation indices reference non-existent passages
- Drop if output < 50 tokens (too short to contain real reasoning)
- Target: keep ≥ 80% of generated examples

**Exit criteria:**
- 100 examples hand-spot-checked and approved
- Generation pipeline documented transparently for paper
- Cache committed (or gitignored with hash of contents committed)

---

### M5: Prompt Templates & Distractor Sampling
**Estimated hours:** 4–5
**Dependencies:** M2, M3
**Deliverables:**
- `src/training/prompts.py` — all prompt templates in one file:
  - LoRA-A: question only → answer
  - LoRA-A': question + 5 passages → answer (passages as noise)
  - LoRA-B: question + 5 passages → reasoned answer with citations
  - Inference prompts for all 3 retrieval conditions
- `src/training/distractors.py` — distractor sampling:
  - 80% probability: gold passage + 4 random distractors
  - 20% probability: 5 random distractors (no gold)
  - Same passages used for LoRA-A' and LoRA-B (critical for clean ablation)
- Tests for prompt formatting and distractor distribution

**Exit criteria:** 3 worked examples each of LoRA-A, LoRA-A', LoRA-B inputs shown and approved.

---

### M6: Training Scripts
**Estimated hours:** 12–15
**Dependencies:** M4, M5
**Deliverables:**
- `src/training/lora_a.py` — Q→A recipe
- `src/training/lora_a_prime.py` — Q+passages→A recipe (passages as noise)
- `src/training/lora_b.py` — RAFT-style recipe with synthetic targets
- `src/training/common.py` — shared QLoRA setup (Unsloth, 4-bit, hyperparameters)
- `configs/lora_a.yaml`, `configs/lora_a_prime.yaml`, `configs/lora_b.yaml`
- Kaggle notebooks: `02_lora_a_kaggle.ipynb`, `02b_lora_a_prime_kaggle.ipynb`, `03_lora_b_kaggle.ipynb`
- `--offline` flag support for local smoke tests

**Hyperparameters (shared):**
- Base: Llama 3.1 8B Instruct, 4-bit quantized
- LoRA: rank 16, alpha 32, dropout 0.05
- Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- LR: 2e-4, cosine schedule, 3% warmup
- Batch: 1, grad accum 16, max seq len 2048
- 1 epoch, checkpoint every 200 steps

**Training runs:**
- 3 recipes × 3 seeds = 9 runs
- ~4 hours each on Kaggle 2×T4
- Total: 36 GPU-hours (within 1 week free Kaggle quota)

**Validation before real runs:**
Before any real training run, do a 100-step training run on each recipe with logging at every step. Inspect the loss curve:
- If flat: learning rate too low or data issue
- If exploding: learning rate too high or gradient issue
- If oscillating: batch size or data ordering issue
This is cheap insurance — fix issues before burning a full 4-hour run.

**Exit criteria:** Smoke test (1 step, 2 examples) completes on CPU for all 3 recipes. 100-step validation run shows healthy loss curves.

---

### M7: Inference & Eval Pipeline
**Estimated hours:** 10–12
**Dependencies:** M3, M6
**Deliverables:**
- `src/inference/generate.py` — batched generation with standard HF (no vLLM in v1)
- `src/inference/cells.py` — 12-cell experiment runner
- `src/eval/metrics.py` — accuracy, macro F1, ECE (calibration)
- `src/eval/faithfulness.py` — lexical overlap + LLM-as-judge on 200-example subset (budget capped at $10)
- `src/eval/bootstrap.py` — paired bootstrap significance tests (1000 resamples)
- `src/eval/interaction_test.py` — pre-registered primary test
- `src/eval/power_simulation.py` — simulate analysis on synthetic data with known effect sizes (3, 5, 8 points) to verify test has adequate power at our sample size
- `preregistration.md` — committed BEFORE seeing real results, contains:
  - Exact hypotheses
  - Statistical tests to be used
  - Decision criteria (alpha level, effect size thresholds)
  - Timestamp of commit
- `scripts/run_inference.sh` — runs all 12 cells

**Inference passes:**
- **PubMedQA (primary, 3 retrieval conditions: none/strong/oracle):**
  - Base (deterministic): 3 retrieval × 1 = 3 passes
  - LoRA-A, LoRA-A', LoRA-B: 3 recipes × 3 retrieval × 3 seeds = 27 passes
  - Subtotal: 30 passes
- **MIRAGE (external validation, 2 retrieval conditions: none/strong only):**
  - Base (deterministic): 2 retrieval × 1 = 2 passes
  - LoRA-A, LoRA-A', LoRA-B: 3 recipes × 2 retrieval × 3 seeds = 18 passes
  - Subtotal: 20 passes
- **Total: 50 inference passes**

**Pre-registered tests (PRIMARY on PubMedQA):**
1. LoRA-B vs base, within each retrieval condition
2. LoRA-B vs LoRA-A, within each retrieval condition
3. LoRA-B vs LoRA-A', within each retrieval condition (cleanest mechanistic test)
4. Interaction test: is (LoRA-B gain at oracle) − (LoRA-B gain at none) significantly different from the same for LoRA-A'?

**Pre-registered tests (SECONDARY on MIRAGE):**
5. External validity: none-vs-strong contrast for LoRA-B vs base, LoRA-B vs LoRA-A, LoRA-B vs LoRA-A' across three held-out distributions (MMLU-Med, MedQA-US, MedMCQA)

**Note:** MIRAGE oracle retrieval is not reported because MMLU-Med, MedQA-US, and MedMCQA do not provide gold supporting passages. The full retrieval-quality interaction is characterized on PubMedQA, where gold abstracts are available, while MIRAGE provides external validity for the none-vs-strong contrast across three held-out distributions.

**Power simulation:** Before running the real interaction test, verify the test has adequate power. If 3 seeds is too few to detect a 5-point effect at p<0.05, we know before spending compute and can budget for 5 seeds instead.

**Exit criteria:** Base model inference runs on all 3 retrieval conditions, numbers reported. `preregistration.md` committed. Power simulation confirms adequate power or we adjust seed count.

---

### M8: Annotation Tool
**Estimated hours:** 4–5
**Dependencies:** M7
**Deliverables:**
- `src/annotation/app.py` — Streamlit UI
- `src/annotation/schema.py` — error rubric (7 categories)
- Local SQLite or JSON storage for annotations
- Export to CSV for analysis

**Error categories:**
1. Factual recall error
2. Statistical misinterpretation
3. Population overgeneralization
4. Stale evidence anchoring
5. Reasoning error
6. Refusal / non-answer
7. Other

**Exit criteria:** App runs on laptop, can annotate one example end-to-end.

---

### M9: Analysis & Paper Scaffolding
**Estimated hours:** 6–8
**Dependencies:** M7, M8
**Deliverables:**
- `src/analysis/tables.py` — LaTeX table generation
- `src/analysis/figures.py` — matplotlib plots (error rate × retrieval × model)
- `src/analysis/stratify.py` — breakdown by question type
- `paper/main.tex` — skeleton with sections, placeholders for numbers, including:
  - **Negative results section template** — written regardless of how results come out. If LoRA-B beats LoRA-A', fill in positive story. If not, the section is ready to be honest about it. Forces thinking about both outcomes equally during planning.
- `paper/refs.bib` — bibliography with cited prior work
- `scripts/reproduce.sh` — one command to run full pipeline

**Key figures:**
1. 12-cell accuracy heatmap (model × retrieval)
2. Error category breakdown (grouped bar chart)
3. Interaction effect visualization (LoRA-B gain vs retrieval quality)

**Exit criteria:** LaTeX compiles, figures generate with placeholder data. Negative results section template exists.

---

## Total Estimated Hours

| Milestone | Hours |
|-----------|-------|
| M1: Plan & Scaffold | 3–4 |
| M2: Data Layer | 8–10 |
| M3: Retrieval Layer | 10–12 |
| M4: Synthetic Data Generation | 6–8 |
| M5: Prompts & Distractors | 4–5 |
| M6: Training Scripts | 12–15 |
| M7: Inference & Eval | 10–12 |
| M8: Annotation Tool | 4–5 |
| M9: Analysis & Paper | 6–8 |
| **Total** | **63–79 hours** |

**Realistic estimate: ~100 hours.** Research engineering with debugging and dead ends typically runs 1.5× the optimistic estimate. Plan for this and don't treat overruns as failures. If M3 takes 18 hours instead of 12 because the BGE index won't build the first time, that's normal.

At 10 hrs/week, this is 8–10 weeks of engineering before real training runs begin.

---

## Five Highest-Risk Parts

### 1. Synthetic data quality for LoRA-B (HIGH RISK)
**What could go wrong:** Claude Haiku might generate plausible-sounding but unfaithful reasoning, or fail to properly cite passages by index. If the training targets don't actually demonstrate "reading evidence," LoRA-B learns nothing useful and the mechanistic claim collapses.

**Mitigation:** Strict prompt template, aggressive filtering, 100-example spot check before any training. If quality is poor, iterate on prompt or escalate to Claude Sonnet (higher cost but better instruction-following).

**Fallback:** If synthetic generation fails entirely, we could try option (a) from the original discussion—reformatting existing rationales—but this weakens the paper significantly.

### 2. LoRA-A' vs LoRA-B distinction might not materialize (HIGH RISK)
**What could go wrong:** LoRA-A' and LoRA-B might perform identically, meaning "trained to reason over passages" doesn't matter—only "saw passages during training" matters. This would invalidate the mechanistic hypothesis.

**Mitigation:** None—this is an empirical question. If it happens, the paper pivots to reporting a null result on the mechanistic claim while still contributing the LoRA-A vs retrieval interaction findings.

**Fallback:** A null result is still publishable if framed honestly: "Contrary to our hypothesis, training the model to explicitly reason over passages did not improve evidence utilization beyond simply exposing it to passages during training."

**IMPORTANT:** A null result here is valuable and should not be avoided by changing the design. If LoRA-A' and LoRA-B perform identically, that IS the finding — "exposure to passages during training is what matters, not the reasoning format." Do not try to rescue the mechanistic claim by changing the design after seeing results. Pre-registration matters.

### 3. "Strong" retrieval might not be strong enough (MEDIUM RISK)
**What could go wrong:** BGE + reranker on the relatively small corpus might achieve Recall@5 > 0.9, making "strong" nearly indistinguishable from "oracle." This compresses the retrieval axis and reduces statistical power to detect interaction effects.

**Mitigation:** Measure retrieval quality early (M3) before any training. If it's too high, we can artificially degrade it (e.g., skip reranking, reduce to top-3) or add a "medium" condition.

**Fallback:** If strong ≈ oracle, collapse them into one condition and reframe as "retrieval vs no-retrieval" study. Less interesting but still valid.

### 4. Free-tier compute constraints (MEDIUM RISK)
**What could go wrong:** Llama 3.1 8B + 2048 context might OOM on T4 even with QLoRA + Unsloth. Or training might take >6 hours per run, exhausting Kaggle quota.

**Mitigation:** Smoke test memory budget in M6 before committing to real runs. If OOM, reduce max_seq_len to 1536 or 1024 (will truncate some long passages but should still work). If time budget is tight, reduce checkpointing frequency.

**Fallback:** Switch to Llama 3.2 3B as base model. Smaller, cheaper, but weaker—results might not generalize to larger models.

### 5. Statistical power with 3 seeds (LOW-MEDIUM RISK)
**What could go wrong:** 3 seeds might not be enough to detect a real but small interaction effect. High variance across seeds could swamp the signal.

**Mitigation:** Use paired bootstrap tests (more powerful than t-tests for small n). Report effect sizes, not just p-values. If variance is high, be honest about it in the paper.

**Fallback:** If results are noisy, run 2 additional seeds (5 total) at the cost of another week of Kaggle time.

---

## Open Risks

These are uncertainties I cannot resolve through planning—they require empirical investigation or your judgment.

### R1: Synthetic data generation quality
**Uncertainty:** Will Claude Haiku 4.5 reliably produce training targets that (a) correctly cite passages by index, (b) contain substantive reasoning rather than generic hedging, and (c) don't leak recognizable "Claude-isms" into the trained model?

**Status:** Unknown until M4 is complete and 100 examples are reviewed.

**Your action needed:** After M4, you'll review the spot-check and decide whether quality is acceptable.

### R2: LoRA-A' vs LoRA-B empirical distinction
**Uncertainty:** Will training on reasoned targets (LoRA-B) produce measurably different behavior than training on answer-only targets with the same input passages (LoRA-A')?

**Status:** Unknown until real training runs are complete. This is the core empirical question of the study.

**Your action needed:** None now. If the distinction doesn't appear, we discuss how to frame the paper.

### R3: Statistical power for interaction effects
**Uncertainty:** With ~1,000 test examples (PubMedQA) and 3 seeds, do we have enough statistical power to detect a real interaction effect of plausible size?

**Back-of-envelope:** If base accuracy is ~60% and we're looking for a 5-point interaction effect (LoRA-B gains 10 points at oracle vs 5 points at none), paired bootstrap with 3 seeds should detect this at p<0.05—but it's borderline. A 3-point effect might be undetectable.

**Your action needed:** Decide now whether to commit to 3 seeds or budget for 5. I recommend starting with 3 and adding 2 more only if results are noisy.

### R4: MIRAGE subset feasibility
**Uncertainty:** MIRAGE is a meta-benchmark with multiple constituent datasets. Some may not have gold passages in a clean format. Constructing a stratified subset might require significant data wrangling.

**Status:** RESOLVED in M2.5. MIRAGE loads 500 examples stratified across MMLU-Med (165), MedQA-US (165), and MedMCQA (170). Gold passages are NOT available for these sources, so MIRAGE supports only "none" and "strong" retrieval conditions (no oracle). This is acceptable — MIRAGE tests external validity of the none-vs-strong contrast across three held-out distributions, while PubMedQA tests the full 3-condition design including oracle.

**Your action needed:** None — resolved.

### R5: Inference cost and time
**Uncertainty:** Running 30 inference passes on ~1,000 test examples each = 30,000 generations. On free Colab T4 with standard HF generation (not vLLM), this could take 10+ hours total. vLLM would be faster but adds setup complexity.

**Status:** Unknown until M7 smoke test.

**Your action needed:** Decide whether inference speed is worth adding vLLM complexity, or whether we just let it run overnight.

---

## Resolved Questions

### Q1: BioASQ question types
**Decision:** Factoid + yes/no only. Exclude list questions — different answer format complicates evaluation. Document exclusion in data section of paper.

### Q2: Answer format
**Decision:** Unified format. Single template ending with "Answer: [label]" where label is yes/no/maybe (for yes/no questions) or entity string (for factoid questions). Reasoning prefix is the same across types.
- LoRA-A targets: "Answer: [label]" only
- LoRA-A' targets: "Answer: [label]" only (same as LoRA-A, just with passages in input)
- LoRA-B targets: "Based on... [reasoning with citations]... Answer: [label]"

### Q3: Faithfulness metric
**Decision:** Implement both. Lexical overlap is primary metric. If synthetic data generation comes in under $5 (expected $3–5), run LLM-as-judge with Claude Haiku on 200-example stratified subsample, capped at $10 total spend. If synthetic generation overruns, drop LLM judge and document as limitation.

### Q4: vLLM
**Decision:** No vLLM in v1. Start with HF generate. Inference is embarrassingly parallel across cells — let it run overnight. If single pass takes >4 hours, revisit.

---

## IDEAS.md Policy

`IDEAS.md` is a graveyard, not a roadmap. Anything that smells like "we could also try DoRA" or "what about clinical voice with Whisper" goes in IDEAS.md and never comes back into v1. The project finishes by maintaining scope discipline.

---

## Status

- [x] Plan approved (2025-04-07)
- [x] Q1–Q4 answered
- [x] Additional directives incorporated
- [x] M1 scaffold complete (2025-04-07)
- [x] M2 data layer complete (2026-04-07)
- [x] M3 retrieval layer complete (2026-04-07) — Hit@5=0.880 in band, BioASQ pivot
- [ ] M4 synthetic data generation
- [ ] M5 prompts & distractors
- [ ] M6 training scripts
- [ ] M7 inference & eval
- [ ] M8 annotation tool
- [ ] M9 analysis & paper

---

*Last updated: 2026-04-07*
