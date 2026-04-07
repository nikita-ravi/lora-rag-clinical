# Pre-registration: LoRA × Retrieval Interaction Study

> **IMPORTANT:** This file MUST be committed with a real timestamp before any inference run on the test set. The git commit hash and timestamp at finalization are the proof of pre-registration.

**Status:** DRAFT - To be finalized before seeing any real results (M7)

**Commit timestamp:** [TO BE FILLED WHEN COMMITTED]
**Git commit hash:** [TO BE FILLED WHEN COMMITTED]

---

## Study Design

### Datasets

1. **PubMedQA (Primary Evaluation):** 500 test examples, yes/no/maybe classification
   - Supports 3 retrieval conditions: none, strong, oracle
   - Full 12-cell design (4 model × 3 retrieval)
   - Gold abstracts available for oracle retrieval

2. **MIRAGE (External Validation):** 500 examples stratified across 3 sources
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

## Primary Hypothesis (PubMedQA)

The type of LoRA training determines the shape of the interaction with retrieval quality:

- **H1:** LoRA-B (RAFT-style) shows a larger gain over baseline when retrieval is strong/oracle compared to when retrieval is absent.
- **H2:** LoRA-A' (passages-as-noise) does NOT show this same interaction pattern.
- **H3:** The difference in interaction effects (LoRA-B vs LoRA-A') is statistically significant.

## Statistical Tests

### Primary Test: Interaction Effect (PubMedQA only)

**Test:** Is (LoRA-B gain at oracle) − (LoRA-B gain at none) significantly different from (LoRA-A' gain at oracle) − (LoRA-A' gain at none)?

**Method:** Paired bootstrap test with 1000 resamples

**Alpha level:** 0.05 (two-tailed)

**Effect size:** Cohen's h for proportions

**Dataset:** PubMedQA test set (500 examples)

**Justification for primary dataset:** PubMedQA is the only dataset with gold passages, enabling oracle retrieval. The full 3-condition retrieval design (none/strong/oracle) is necessary to characterize how LoRA training interacts with retrieval quality.

### Secondary Tests on PubMedQA

1. LoRA-B vs base, within each retrieval condition
2. LoRA-B vs LoRA-A, within each retrieval condition
3. LoRA-B vs LoRA-A', within each retrieval condition

**Multiple comparison correction:** Bonferroni for secondary tests

### External Validity Tests on MIRAGE

**Purpose:** Test whether the none-vs-strong contrast generalizes to held-out medical QA distributions.

**Tests:**
4. LoRA-B vs base: (gain at strong) − (gain at none) across MIRAGE sources
5. LoRA-B vs LoRA-A: same contrast
6. LoRA-B vs LoRA-A': same contrast

**Note:** Oracle retrieval is not tested on MIRAGE because MMLU-Med, MedQA-US, and MedMCQA do not provide gold supporting passages. These tests are secondary and confirmatory — the primary interaction test (with oracle) is conducted on PubMedQA.

**Stratification:** Results reported separately for each MIRAGE source to characterize generalization.

---

## Decision Criteria

### Strong support for hypothesis:
- H3 p-value < 0.05 on PubMedQA
- LoRA-B interaction > LoRA-A' interaction
- Effect size Cohen's h > 0.3 (small-to-medium)
- Consistent direction on MIRAGE external validity tests

### Moderate support:
- H3 p-value < 0.10 on PubMedQA
- Direction consistent with hypothesis
- At least 2/3 MIRAGE sources show consistent direction

### Null result:
- H3 p-value > 0.10 on PubMedQA
- Report as: "Contrary to hypothesis, training format did not affect interaction with retrieval quality"

---

## Sample Size and Power

- **PubMedQA test set:** 500 examples
- **MIRAGE:** 500 examples (165 + 165 + 170)
- **Seeds:** 3 per LoRA condition
- **Power simulation:** [TO BE COMPLETED IN M7]

Expected detectable effect: 5-point accuracy difference with 80% power

---

## Analysis Plan

### Primary Analysis (PubMedQA)
1. Compute accuracy for all 12 cells (4 model × 3 retrieval)
2. Compute gains relative to base model within each retrieval condition
3. Compute interaction effects: (gain at oracle) − (gain at none) for each LoRA condition
4. Run paired bootstrap test comparing LoRA-B vs LoRA-A' interaction effects
5. Report means ± std across 3 seeds
6. Report p-values and effect sizes

### External Validity Analysis (MIRAGE)
7. Compute accuracy for all 8 cells (4 model × 2 retrieval)
8. Compute gains relative to base model
9. Report none-vs-strong contrast by source (MMLU-Med, MedQA-US, MedMCQA)
10. Assess consistency of direction with PubMedQA findings

---

## What Will NOT Change After This Commit

- The primary hypothesis
- The primary statistical test (on PubMedQA)
- The alpha level (0.05)
- The test sets (hashes committed separately)
- The designation of PubMedQA as primary and MIRAGE as secondary

## What May Change (With Documentation)

- Secondary analyses may be added (marked as exploratory)
- If power simulation reveals inadequate power, seed count may increase (documented)
- Stratified analyses by question type may be added (exploratory)

---

## Limitations to Be Reported

1. **MIRAGE oracle retrieval:** Not reported because MMLU-Med, MedQA-US, and MedMCQA do not provide gold supporting passages. The full retrieval-quality interaction is characterized on PubMedQA, where gold abstracts are available, while MIRAGE provides external validity for the none-vs-strong contrast across three held-out distributions.

2. **BioASQ as training data:** BioASQ is used for training only. Test set results are reported on PubMedQA (a different distribution) to avoid train-test contamination.

---

*This document will be committed before any real inference results are observed.*
