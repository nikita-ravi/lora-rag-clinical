# IDEAS.md — Out of Scope Extensions

This file is a graveyard, not a roadmap. Everything here is explicitly out of scope for v1.
Do not implement any of these. Do not sneak them in. The project finishes by maintaining scope discipline.

---

## Model Variants
- [ ] Multiple base models (Llama 3.2 3B, Mistral 7B, etc.)
- [ ] Larger LoRA variants (DoRA, LoRA+, rsLoRA)
- [ ] Different quantization schemes (GPTQ, AWQ)
- [ ] Instruction-tuning the base model further before LoRA

## Retrieval Variants
- [ ] Adversarial retrieval conditions (intentionally misleading passages)
- [ ] Weak BM25-only retrieval condition
- [ ] Joint retriever-generator training (RA-DIT style)
- [ ] Different embedding models (E5, Contriever)
- [ ] Hybrid sparse+dense retrieval

## Evaluation Extensions
- [ ] Multilingual evaluation
- [ ] Long-context evaluation (>2048 tokens)
- [ ] Additional medical QA benchmarks (MedQA, MedMCQA)
- [ ] Human evaluation beyond error annotation

## Infrastructure
- [ ] Web demo / hosted version
- [ ] vLLM for faster inference
- [ ] Multi-GPU training

## Analysis Extensions
- [ ] Probing experiments to understand what LoRA-B learns
- [ ] Attention analysis on evidence passages
- [ ] Confidence calibration analysis by question type

---

*Add new ideas below with date and brief description. Then forget about them until v2.*

## M4 Observations for Paper (2026-04-08)

**Emergent comma-separated citation behavior**: During M4a pilot inspection, we observed that Claude Haiku spontaneously produced both individual citations [P1] and grouped citations [P1, P3] for multi-source claims. This was not specified in the prompt - Haiku inferred that multiple passages supporting a single claim could be compactly cited together. We accept both formats in our filtering pipeline. This emergent compactness in citation behavior may be a useful pattern for future RAFT-style training data generation. Worth mentioning in the M9 paper as a small methodological finding about LLM behavior under citation-heavy prompts.

**Emergent calibrated hedging on yesno questions (2026-04-08)**: During M4b full-run filtering, we observed that Haiku produced "Maybe" answers on 65/957 yesno questions (6.8%), of which 47 had BioASQ gold labels of "Yes" and 18 of "No". Manual inspection of 5 random "Maybe vs Yes" cases (seed=300) showed Haiku performing literal passage-reading and correctly identifying hedged or partial evidence (e.g., distinguishing "reduced nucleosome occupancy" from "nucleosome-free regions", "cardiac ischemia/reperfusion protection" from "heart failure"). Because PLAN.md Q2 specifies yes/no/maybe as the yesno label space and M4's goal is faithful evidence-grounded reasoning, we updated the filter to accept Maybe responses with valid citations rather than rejecting them. This recovered 65 examples and is itself a small finding: BioASQ's binary yes/no gold labels appear to overstate certainty in ~5% of cases relative to literal passage evidence. Worth mentioning in the M9 paper as a methodological observation.

**Inferential gold labels in BioASQ yesno questions (2026-04-13)**: M5 investigation into the 19.9% "Insufficient evidence" rate in the M4b training set surfaced a related but distinct pattern from the Maybe-hedging finding. Of 373 Insufficient evidence examples, 280 (75%) are easy-mode (gold passage present in input). Manual inspection of 5 random easy-mode yesno cases (seed=700) found Haiku was correctly identifying that BioASQ's gold passages sometimes support the gold answer only through inference, not direct statement. Examples: "rivaroxaban excreted via kidney" vs. gold label "metabolized in kidneys" (different processes); "tear lysozyme" (2 words) vs. gold label "abundant in tears" (requires quantitative claim); one example had the gold passage incorrectly linked to the question entirely (no mention of amifostine in the passage for a question about amifostine's effect on HIF-1α). Combined with the Maybe-hedging observation, this suggests BioASQ yesno gold labels and their cited gold passages are not always consistent at a literal reading level — some fraction of gold labels require readers to bring external biomedical knowledge to the passage. This is a methodological observation worth a short workshop note on its own, and it validates the Insufficient evidence label as a legitimate faithfulness signal rather than a filter artifact. Also note the asymmetry: easy-mode Insufficient yesno is 61%/39% yes/no; hard-mode is 82%/18% yes/no, suggesting BioASQ "yes" questions are systematically harder to answer from retrieval alone.
