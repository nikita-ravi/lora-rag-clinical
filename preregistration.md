# Pre-registration: LoRA × Retrieval Interaction Study

> **IMPORTANT:** This file MUST be committed with a real timestamp before any inference run on the test set. The git commit hash and timestamp at finalization are the proof of pre-registration.

**Status:** DRAFT - To be finalized before seeing any real results (M7)

**Commit timestamp:** [TO BE FILLED WHEN COMMITTED]
**Git commit hash:** [TO BE FILLED WHEN COMMITTED]

---

## Study Design

### Datasets

1. **BioASQ (Primary Evaluation):** 500 test examples, factoid + yesno questions
   - Expert-written questions with extractive gold snippets
   - Supports 3 retrieval conditions: none, strong, oracle
   - Full 12-cell design (4 model × 3 retrieval)
   - Stratified by question type: ~52% factoid, ~48% yesno
   - Test set hash: `bf19d29ada7450b2e1057f678c51a2bf5db6b88db0d7ec88f1c3875ed3c84360`

2. **PubMedQA (Secondary/Exploratory):** 500 test examples, yes/no/maybe classification
   - Demoted from primary due to retrieval saturation (R@5=0.980)
   - Questions derived from abstract titles cause trivial title-to-abstract matching
   - Supports 3 retrieval conditions: none, strong, oracle
   - Retained for exploratory analysis and comparison with BioASQ findings

3. **MIRAGE (External Validation):** 500 examples stratified across 3 sources
   - MMLU-Med: 165 examples
   - MedQA-US: 165 examples
   - MedMCQA: 170 examples
   - Supports 2 retrieval conditions: none, strong (no oracle — gold passages unavailable)
   - 8-cell design (4 model × 2 retrieval)

### Model Conditions

1. **Base:** Llama 3.1 8B Instruct (no fine-tuning)
2. **LoRA-A:** Q→A training (question only)
3. **LoRA-A':** Q+passages→A training (passages as noise control)
4. **LoRA-B:** RAFT-style Q+passages→reasoned A with citations

---

## Primary Hypothesis (BioASQ)

The type of LoRA training determines the shape of the interaction with retrieval quality:

- **H1:** LoRA-B (RAFT-style) shows a larger gain over baseline when retrieval is strong/oracle compared to when retrieval is absent.
- **H2:** LoRA-A' (passages-as-noise) does NOT show this same interaction pattern.
- **H3:** The difference in interaction effects (LoRA-B vs LoRA-A') is statistically significant.

## Statistical Tests

### Primary Test: Interaction Effect (BioASQ)

**Test:** Is (LoRA-B gain at oracle) − (LoRA-B gain at none) significantly different from (LoRA-A' gain at oracle) − (LoRA-A' gain at none)?

**Method:** Paired bootstrap test with 1000 resamples

**Alpha level:** 0.05 (two-tailed)

**Effect size:** Cohen's h for proportions

**Dataset:** BioASQ test set (500 examples)

**Justification for primary dataset:** BioASQ has expert-written questions with extractive gold snippets, enabling meaningful oracle retrieval. PubMedQA was demoted because its questions (paraphrased abstract titles) caused retrieval saturation (R@5=0.980), making "strong" retrieval indistinguishable from oracle.

### Secondary Tests on BioASQ

1. LoRA-B vs base, within each retrieval condition
2. LoRA-B vs LoRA-A, within each retrieval condition
3. LoRA-B vs LoRA-A', within each retrieval condition

**Multiple comparison correction:** Bonferroni for secondary tests

**Stratification:** Results reported separately for factoid vs yesno question types.

### Exploratory Tests on PubMedQA

**Purpose:** Compare interaction patterns between BioASQ (harder retrieval) and PubMedQA (saturated retrieval).

**Tests:**
4. Same 12-cell design as BioASQ (4 model × 3 retrieval)
5. Compare interaction effect magnitudes between datasets

**Note:** PubMedQA retrieval saturates at R@5=0.980 due to structural bias (question ≈ abstract title). These tests are exploratory and may reveal whether the LoRA × retrieval interaction depends on retrieval difficulty.

### External Validity Tests on MIRAGE

**Purpose:** Test whether the none-vs-strong contrast generalizes to held-out medical QA distributions.

**Tests:**
6. LoRA-B vs base: (gain at strong) − (gain at none) across MIRAGE sources
7. LoRA-B vs LoRA-A: same contrast
8. LoRA-B vs LoRA-A': same contrast

**Note:** Oracle retrieval is not tested on MIRAGE because MMLU-Med, MedQA-US, and MedMCQA do not provide gold supporting passages. These tests are secondary and confirmatory — the primary interaction test (with oracle) is conducted on BioASQ.

**Stratification:** Results reported separately for each MIRAGE source to characterize generalization.

---

## Decision Criteria

### Strong support for hypothesis:
- H3 p-value < 0.05 on BioASQ
- LoRA-B interaction > LoRA-A' interaction
- Effect size Cohen's h > 0.3 (small-to-medium)
- Consistent direction on MIRAGE external validity tests

### Moderate support:
- H3 p-value < 0.10 on BioASQ
- Direction consistent with hypothesis
- At least 2/3 MIRAGE sources show consistent direction

### Null result:
- H3 p-value > 0.10 on BioASQ
- Report as: "Contrary to hypothesis, training format did not affect interaction with retrieval quality"

---

## Sample Size and Power

- **BioASQ test set:** 500 examples (261 factoid, 239 yesno)
- **PubMedQA test set:** 500 examples (exploratory)
- **MIRAGE:** 500 examples (165 + 165 + 170)
- **Seeds:** 3 per LoRA condition
- **Power simulation:** [TO BE COMPLETED IN M7]

Expected detectable effect: 5-point accuracy difference with 80% power

---

## Analysis Plan

### Primary Analysis (BioASQ)
1. Compute accuracy for all 12 cells (4 model × 3 retrieval)
2. Compute gains relative to base model within each retrieval condition
3. Compute interaction effects: (gain at oracle) − (gain at none) for each LoRA condition
4. Run paired bootstrap test comparing LoRA-B vs LoRA-A' interaction effects
5. Report means ± std across 3 seeds
6. Report p-values and effect sizes
7. Stratify by question type (factoid vs yesno)

### Exploratory Analysis (PubMedQA)
8. Compute accuracy for all 12 cells (4 model × 3 retrieval)
9. Compare interaction effect magnitudes with BioASQ findings
10. Assess whether retrieval saturation affects interaction patterns

### External Validity Analysis (MIRAGE)
11. Compute accuracy for all 8 cells (4 model × 2 retrieval)
12. Compute gains relative to base model
13. Report none-vs-strong contrast by source (MMLU-Med, MedQA-US, MedMCQA)
14. Assess consistency of direction with BioASQ findings

---

## What Will NOT Change After This Commit

- The primary hypothesis
- The primary statistical test (on BioASQ)
- The alpha level (0.05)
- The test sets (hashes committed separately)
- The designation of BioASQ as primary, PubMedQA as exploratory, and MIRAGE as external validation

## What May Change (With Documentation)

- Secondary analyses may be added (marked as exploratory)
- If power simulation reveals inadequate power, seed count may increase (documented)
- Stratified analyses by question type may be added (exploratory)

---

## Limitations to Be Reported

1. **MIRAGE oracle retrieval:** Not reported because MMLU-Med, MedQA-US, and MedMCQA do not provide gold supporting passages. The full retrieval-quality interaction is characterized on BioASQ, where gold snippets are available, while MIRAGE provides external validity for the none-vs-strong contrast across three held-out distributions.

2. **PubMedQA retrieval saturation:** PubMedQA retrieval saturates at R@5=0.980 because questions are paraphrased abstract titles. This structural bias makes "strong" retrieval nearly indistinguishable from oracle. PubMedQA results are reported as exploratory, comparing interaction patterns between hard (BioASQ) and saturated (PubMedQA) retrieval settings.

3. **BioASQ split design:** BioASQ train/dev/test splits were created with stratified random sampling (seed=42) from BioASQ Task B years 1-11. Train set is used for LoRA fine-tuning; test set is held out for evaluation.

---

*This document will be committed before any real inference results are observed.*
