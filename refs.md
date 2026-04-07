# Reading List & Bibliography
## LoRA × Retrieval Interaction on Clinical Evidence Interpretation

This is the working bibliography for the project. Papers are grouped by what role they play in the work, with notes on why each one matters and what to take from it. Read the **must-read** ones cover-to-cover before scaffolding code. Skim the **reference** ones — you need to know they exist and what they claim, but you don't need to internalize every detail.

---

## Group 1: The methodological core (must-read, fully)

These four papers define everything you're building. If you don't understand all four deeply, the project won't survive peer review.

### 1. LoRA — Hu et al., 2021
**"LoRA: Low-Rank Adaptation of Large Language Models"**
arXiv: https://arxiv.org/abs/2106.09685
PDF: https://arxiv.org/pdf/2106.09685

The foundational paper for everything you're doing. LoRA freezes the pretrained weights and trains small low-rank decomposition matrices injected into each transformer layer. This is what makes the entire project possible on free compute. You need to understand: the rank parameter, alpha scaling, which modules to target (q_proj, k_proj, v_proj, o_proj, etc.), and why LoRA does not add inference latency. The "intrinsic rank" hypothesis matters because it justifies why rank 16 works as well as rank 64 for most adaptation tasks.

**Take from it:** the math of how A and B matrices work, the standard hyperparameter ranges, the targeting choices.

---

### 2. QLoRA — Dettmers et al., 2023
**"QLoRA: Efficient Finetuning of Quantized LLMs"**
arXiv: https://arxiv.org/abs/2305.14314
GitHub: https://github.com/artidoro/qlora

The reason you can run this project on free Kaggle GPUs. QLoRA quantizes the base model to 4 bits, freezes it, and trains LoRA adapters in higher precision through it. Three key innovations: 4-bit NormalFloat (NF4), double quantization, and paged optimizers. Without QLoRA, fine-tuning Llama 3.1 8B on a free T4 is not feasible.

**Take from it:** why bf16 compute dtype, why NF4, what the memory savings actually are, and the failure modes of low-bit quantization. Also: QLoRA is reported to match 16-bit fine-tuning performance, which means your results won't be confounded by the quantization choice.

---

### 3. RAFT — Zhang et al., 2024
**"RAFT: Adapting Language Model to Domain Specific RAG"**
arXiv: https://arxiv.org/abs/2403.10131
Berkeley blog: https://gorilla.cs.berkeley.edu/blogs/9_raft.html

This is the **direct ancestor of your LoRA-B recipe**. RAFT trains models on (question, retrieved documents → chain-of-thought answer) where the retrieved documents are a mix of "oracle" passages (containing the answer) and distractor passages. Critically, for some fraction of training examples (default 20%), the oracle is omitted entirely so the model learns to fall back on parametric knowledge when retrieval fails.

The paper evaluates on PubMed, HotpotQA, and the Gorilla API benchmark. It shows that RAFT-trained models significantly outperform both vanilla RAG and standard supervised fine-tuning, especially in handling distractors. The "open-book exam" analogy is the central framing — and it's also exactly the framing your paper will inherit and extend.

**Take from it:** the exact training data construction (P fraction with oracle, 1−P with only distractors), the chain-of-thought answer format with verbatim quotes, and the comparison baselines (DSF, DSF+RAG). Your LoRA-B recipe should follow RAFT closely so reviewers immediately recognize what you're doing.

**The key extension your work makes over RAFT:** RAFT presents this as a method paper ("here is a better recipe"). Your paper presents it as a controlled scientific question ("when does this recipe help vs hurt as a function of retrieval quality, and why?"). Same building block, different research question.

---

### 4. Soudani et al., 2024
**"Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge"**
arXiv: https://arxiv.org/abs/2403.01432
Code: https://github.com/informagi/RAGvsFT
Published at SIGIR-AP 2024.

**The most important prior work for framing your contribution.** This is the paper Claude Code's analysis flagged that I missed in my first pass — it's the closest existing study that varies retrieval quality as an independent variable, and it concludes that fine-tuning and RAG are *complementary* (both improve together as retrieval/data quality improves). The study is on entity-popularity QA, not clinical reasoning.

You cannot claim "nobody has studied the interaction." You have to position your paper as: *"Soudani et al. showed the interaction is positive on entity-popularity QA. We extend to a domain where reading skill matters (clinical evidence interpretation), decompose by LoRA recipe (Q→A vs RAFT-style), and show the interaction shape depends on what the LoRA was trained to do."*

**Take from it:** the experimental design pattern of varying retriever quality, the framing of complementarity vs substitution, and the explicit ack that this is your closest neighbor in the literature.

---

## Group 2: Background and foundational context (must-read, can skim some sections)

### 5. RAG — Lewis et al., 2020
**"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"**
arXiv: https://arxiv.org/abs/2005.11401
Published at NeurIPS 2020.

The original RAG paper. Defines the architecture: parametric memory (a seq2seq model) plus non-parametric memory (a dense vector index). Even though modern RAG implementations don't follow this paper's exact architecture (most use a frozen retriever with in-context augmentation rather than end-to-end training), every RAG paper cites this. You need to know it exists and have a one-paragraph summary in your head.

**Take from it:** the vocabulary (parametric vs non-parametric memory, RAG-Sequence vs RAG-Token), and the canonical citation.

---

### 6. Ovadia et al., 2023
**"Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs"**
arXiv: https://arxiv.org/abs/2312.05934
EMNLP 2024.

The paper that everyone cites when arguing "RAG beats fine-tuning." The headline finding is that unsupervised fine-tuning struggles to inject new factual knowledge, while RAG consistently outperforms it. Important caveats: their fine-tuning is *unsupervised* (continued pretraining on documents), not supervised QA fine-tuning, and their evaluation is on factual recall, not reasoning. So this paper is often misquoted as "fine-tuning is dead" when what it actually shows is "naive continued pretraining is a bad way to inject facts."

**Take from it:** the contrast you'll draw in your paper. Your LoRA-A recipe is closer to supervised QA fine-tuning than unsupervised continued pretraining, and your task is reasoning over evidence rather than factual recall. So you'd expect a different outcome from Ovadia, and that difference is part of what makes your study worth doing.

---

## Group 3: Datasets you'll actually use (must-read the dataset papers)

### 7. PubMedQA — Jin et al., 2019
**"PubMedQA: A Dataset for Biomedical Research Question Answering"**
arXiv: https://arxiv.org/abs/1909.06146
ACL Anthology: https://aclanthology.org/D19-1259/
Homepage: https://pubmedqa.github.io/
GitHub: https://github.com/pubmedqa/pubmedqa

Your **primary held-out evaluation set.** Yes/no/maybe biomedical questions paired with the corresponding PubMed abstract as gold context. Three subsets: PQA-L (1k expert-labeled), PQA-U (61k unlabeled), PQA-A (211k auto-generated). You will use the `pqa_labeled` config for evaluation. Critically: PubMedQA is described by its authors as "the first QA dataset where reasoning over biomedical research texts, especially their quantitative contents, is required" — which is exactly the kind of task where your "reading skill" hypothesis should pay off.

The reasoning-required setting is harder: human single-annotator performance is 78% accuracy, BioBERT baseline is 68%, majority baseline is 55%. Plenty of headroom for your study to detect effects.

**Take from it:** the data format (question, context = abstract minus conclusion, long answer = conclusion, label = yes/no/maybe), the train/dev/test split convention, and the baseline numbers to position your results against.

---

### 8. BioASQ Task B
**Tsatsaronis et al., 2015** — "An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition"
BMC Bioinformatics: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0564-6
Participants area: http://participants-area.bioasq.org/

Your **primary training source.** Annual biomedical QA challenge with thousands of expert-curated questions, each linked to gold PubMed snippets and ideal answers. Question types: factoid, yes/no, list, summary. You'll use the most recent year's training data (Task 13B or 12B) — register on the participants area, agree to the terms, download the JSON locally. **Do not redistribute or commit to your repo.**

**Take from it:** the JSON schema (questions have `type`, `body`, `documents`, `snippets`, `exact_answer`, `ideal_answer`), and the fact that gold snippets exist for every question — which is what makes the oracle retrieval condition possible.

---

### 9. MIRAGE — Xiong et al., 2024
**"Benchmarking Retrieval-Augmented Generation for Medicine"**
arXiv: https://arxiv.org/abs/2402.13178
ACL Findings 2024: https://aclanthology.org/2024.findings-acl.372/
GitHub: https://github.com/Teddy-XiongGZ/MIRAGE
MedRAG toolkit (same authors): https://github.com/Teddy-XiongGZ/MedRAG

Your **stress-test set for external validity.** MIRAGE is a medical RAG benchmark composed of five datasets: MMLU-Med, MedQA-US, MedMCQA, PubMedQA*, and BioASQ-Y/N. Note that MIRAGE includes a modified version of PubMedQA — be careful about overlap with your training data. The paper also introduces important findings you should reference: the **log-linear scaling** of model performance with retrieved snippets, the **lost-in-the-middle effect** in medical RAG, and the value of combining retrievers (BM25 + MedCPT).

**Take from it:** the benchmark composition, the lost-in-the-middle finding (important for your top-k choice), and the MedRAG toolkit code (you can reuse parts of it for your retrieval pipeline). Use a stratified subsample of ~500 examples from MIRAGE *excluding* anything that overlaps with your training data.

---

## Group 4: Tools and infrastructure (read the docs, not papers)

### 10. BGE embeddings & reranker — Xiao et al., 2023
**"C-Pack: Packaged Resources To Advance General Chinese Embedding"**
arXiv: https://arxiv.org/abs/2309.07597
Models on HF:
- Embedding: https://huggingface.co/BAAI/bge-base-en-v1.5
- Reranker: https://huggingface.co/BAAI/bge-reranker-base
GitHub (FlagEmbedding): https://github.com/FlagOpen/FlagEmbedding

The retriever and reranker for your "strong" retrieval condition. BGE is a strong open-source dense embedding model trained with contrastive learning, and the BGE reranker is a cross-encoder you'll use for the second stage. The standard pipeline is: retrieve top-20 with BGE bi-encoder, rerank to top-5 with BGE cross-encoder.

**Don't read the paper unless you have time to spare.** Read the model cards on HF, they have everything you need to use the models.

---

### 11. Unsloth (no paper, just docs)
GitHub: https://github.com/unslothai/unsloth
Llama 3.1 notebook: https://docs.unsloth.ai/get-started/unsloth-notebooks

Not a paper — a library. Unsloth is a dramatically faster, more memory-efficient implementation of LoRA/QLoRA training, optimized for free-tier GPUs. It is the difference between "your training run OOMs" and "your training run completes." You don't need to understand the kernels they wrote, but you do need to read their Llama 3.1 8B QLoRA notebook end-to-end before writing any training code, because the conventions matter (chat templates, dataset formatting, target modules).

---

### 12. PEFT library (Hugging Face)
GitHub: https://github.com/huggingface/peft
Docs: https://huggingface.co/docs/peft

The standard implementation of LoRA, QLoRA, and other parameter-efficient fine-tuning methods, integrated with Hugging Face transformers. Unsloth wraps PEFT under the hood. Read the LoRA config docs so you know what every hyperparameter means.

---

## Group 5: Reference papers (skim, cite if relevant)

These are not papers you need to read deeply, but you should know they exist so you can cite them when reviewers ask "what about X?"

### 13. LoRA+ — Hayou et al., 2024
**"LoRA+: Efficient Low Rank Adaptation of Large Models"**
arXiv: https://arxiv.org/abs/2402.12354

Shows that using different learning rates for the A and B matrices improves LoRA performance. You're not using LoRA+ in v1 (vanilla LoRA only, per the brief), but you should mention it as a follow-up direction.

---

### 14. ALoFTRAG — Devine, 2025
**"ALoFTRAG: Automatic Local Fine Tuning for Retrieval Augmented Generation"**
arXiv: https://arxiv.org/abs/2501.11929

A method paper that uses synthetic data generation + LoRA fine-tuning to improve RAG, evaluated across 20 datasets in 26 languages. Reports modest but consistent gains. Cite as a related but methodologically different approach (they propose a method; you're running a controlled study).

---

### 15. RA-DIT — Lin et al., 2023
**"RA-DIT: Retrieval-Augmented Dual Instruction Tuning"**
arXiv: https://arxiv.org/abs/2310.01352

Joint retriever + generator fine-tuning. You're explicitly *not* doing joint tuning (retriever is fixed in your design), but cite as the alternative to single-side fine-tuning. Important to acknowledge so reviewers don't ask "why didn't you tune the retriever too?"

---

### 16. Lost in the Middle — Liu et al., 2023
**"Lost in the Middle: How Language Models Use Long Contexts"**
arXiv: https://arxiv.org/abs/2307.03172

Shows that LLMs disproportionately attend to the beginning and end of long contexts, ignoring the middle. Relevant because: (a) your top-5 retrieved passages need to be ordered carefully, (b) MIRAGE replicates this finding for medical RAG, and (c) it's a confound you should at least acknowledge in the paper's limitations section.

---

### 17. DPR — Karpukhin et al., 2020
**"Dense Passage Retrieval for Open-Domain Question Answering"**
arXiv: https://arxiv.org/abs/2004.04906

The foundational dense retrieval paper. You're using BGE rather than DPR (BGE is stronger), but DPR is the standard citation for "dense retrieval as an approach." Cite it once in the related work section.

---

### 18. Llama 3 — Meta AI, 2024
**"The Llama 3 Herd of Models"**
arXiv: https://arxiv.org/abs/2407.21783
Model: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

The technical report for the model family you're using. You don't need to read it, but cite it as your base model.

---

## Group 6: Domain context (skim if curious, optional)

These are clinical RAG papers that establish the domain context. Useful if you want to understand the broader landscape, not strictly required for the project.

### 19. Optimizing Medical QA Systems — 2025 (arXiv 2512.05863)
arXiv: https://arxiv.org/abs/2512.05863

LoRA fine-tuning + RAG on PubMedQA and MedMCQA, with LLaMA 2 and Falcon. Reaches 71.8% on PubMedQA. Cite as a comparable medical RAG + LoRA setup. Also useful for the error analysis: they found retrieval-augmented generation reduced factual errors from 35% to 14%, with remaining errors being statistical misinterpretation, population overgeneralization, and stale evidence — which is exactly the error taxonomy you're going to use.

### 20. Medical LLMs: Fine-Tuning vs RAG — PMC12292519
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12292519/

FT vs RAG vs FT+RAG comparison across multiple open models on MedQuAD. Closest existing "horse race" study. Cite to position your work as different (you're studying the *interaction*, not running another horse race).

### 21. rsDoRA+ — 2024 (arXiv 2410.16088)
arXiv: https://arxiv.org/abs/2410.16088

A LoRA variant integrated with self-reflective RAG on medical QA. Cite as evidence that the LoRA-meets-RAG question is active in the medical domain.

---

## Reading order recommendation

If you're reading these for the first time, do them in this order:

1. **LoRA** (understand the math)
2. **QLoRA** (understand why this works on free GPUs)
3. **PubMedQA** (understand the data you're evaluating on)
4. **RAFT** (understand the LoRA-B recipe in depth)
5. **Soudani et al.** (understand the prior work you're building on)
6. **MIRAGE** (understand the medical RAG benchmarking landscape)
7. **Ovadia et al.** (understand the contrarian framing you're addressing)
8. **Lewis et al. (RAG)** (foundational, but you can skim — most of it is now standard knowledge)

Total reading time, honestly: 15–25 hours if you take notes. Don't skip this. Every hour you spend reading saves you a day of debugging confused experimental design later.

---

## What to do with this list in Claude Code

When you give the proposal to Claude Code, also paste in this list with the instruction: *"These are the papers I have read and that ground this project. Do not search for additional related work — these are sufficient. If you think we're missing something important, flag it but do not act on it without my approval."* This prevents Claude Code from going down a rabbit hole of background research when you want it building code.
