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
