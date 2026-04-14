#!/usr/bin/env python3
"""
M5 Investigation: Insufficient Evidence Decision

Investigates the 373 "Insufficient evidence" examples in the M4b filtered training set
to determine whether they represent filter failures or legitimate faithfulness signals.

Seed: 700 (for random sampling of easy-mode yesno cases)

Findings:
- 280/373 (75%) are easy-mode (gold passage present)
- Manual spot-check of 5 random easy-mode yesno cases showed Haiku correctly identifying
  BioASQ gold labels that require inference beyond literal passage text
- Decision: Keep all 1877 examples; Insufficient evidence is valid calibrated uncertainty

This script provides reproducibility evidence for the Hypothesis 1 decision documented
in IDEAS.md (2026-04-13) and the dual-reporting decision in PLAN.md M7.
"""

import json
import random
from collections import Counter, defaultdict

# Set seed for reproducibility
random.seed(700)

# Load the filtered training set
data = []
with open("data/synthetic/lora_b_train.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

print(f"Total examples in filtered training set: {len(data)}")

# Find all Insufficient evidence cases
insufficient_cases = [ex for ex in data if ex.get("generated_answer") == "Insufficient evidence"]
print(f"\nTotal 'Insufficient evidence' examples: {len(insufficient_cases)}")

# ===== Step 1: Cross-tab by mode and question_type =====
print("\n" + "="*80)
print("STEP 1: Cross-tab Insufficient evidence by mode and question_type")
print("="*80)

mode_counts = defaultdict(lambda: {"factoid": 0, "yesno": 0})

for ex in insufficient_cases:
    mode = ex.get("mode", "unknown")
    qtype = ex.get("question_type", "unknown")
    mode_counts[mode][qtype] += 1

print("\nCross-tab:")
print(f"{'Mode':<15} {'Factoid':<10} {'Yesno':<10} {'Total':<10}")
print("-" * 50)
for mode in sorted(mode_counts.keys()):
    factoid = mode_counts[mode]["factoid"]
    yesno = mode_counts[mode]["yesno"]
    total = factoid + yesno
    print(f"{mode:<15} {factoid:<10} {yesno:<10} {total:<10}")

# ===== Step 2: Sanity check - easy mode has is_gold=True =====
print("\n" + "="*80)
print("STEP 2: Sanity check - easy-mode Insufficient has is_gold=True")
print("="*80)

easy_insufficient = [ex for ex in insufficient_cases if ex.get("mode") == "easy"]
print(f"\nEasy-mode Insufficient evidence examples: {len(easy_insufficient)}")

issues = []
for i, ex in enumerate(easy_insufficient):
    passages = ex.get("passages", [])
    gold_passages = [p for p in passages if p.get("is_gold", False)]
    if len(gold_passages) != 1:
        issues.append({
            "index": i,
            "question_id": ex.get("question_id"),
            "gold_count": len(gold_passages)
        })

if issues:
    print(f"\n⚠️  Found {len(issues)} easy-mode examples without exactly 1 gold passage:")
    for issue in issues[:5]:
        print(f"   Index {issue['index']}: question_id={issue['question_id']}, gold_count={issue['gold_count']}")
else:
    print("\n✓ All easy-mode Insufficient examples have exactly 1 gold passage (is_gold=True)")

# ===== Step 3: Spot-check 5 random easy-mode yesno Insufficient examples =====
print("\n" + "="*80)
print("STEP 3: Spot-check 5 random easy-mode yesno Insufficient examples (seed=700)")
print("="*80)

easy_yesno_insufficient = [
    ex for ex in insufficient_cases
    if ex.get("mode") == "easy" and ex.get("question_type") == "yesno"
]
print(f"\nEasy-mode yesno Insufficient evidence examples: {len(easy_yesno_insufficient)}")

# Sample 5 random examples
sample_size = min(5, len(easy_yesno_insufficient))
samples = random.sample(easy_yesno_insufficient, sample_size)

for i, ex in enumerate(samples, 1):
    print(f"\n{'─'*80}")
    print(f"EXAMPLE {i}")
    print(f"{'─'*80}")

    print(f"\nQuestion ID: {ex.get('question_id')}")
    print(f"Question: {ex.get('question')}")
    print(f"BioASQ Gold Answer: {ex.get('gold_answer')}")

    # Find the gold passage
    passages = ex.get("passages", [])
    gold_passage = next((p for p in passages if p.get("is_gold", False)), None)

    if gold_passage:
        print(f"\nGold Passage Text:")
        print(f"  {gold_passage.get('text', 'N/A')}")
    else:
        print("\n⚠️  No gold passage found!")

    print(f"\nGenerated Reasoning:")
    reasoning = ex.get("generated_reasoning", "N/A")
    # Truncate if too long
    if len(reasoning) > 500:
        print(f"  {reasoning[:500]}...")
    else:
        print(f"  {reasoning}")

# ===== Step 4: Overlap with gold_disagreement field =====
print("\n" + "="*80)
print("STEP 4: Overlap with gold_disagreement field")
print("="*80)

hedged_insufficient = [
    ex for ex in insufficient_cases
    if ex.get("gold_disagreement") in ["model_hedged_yes", "model_hedged_no"]
]

print(f"\nInsufficient evidence examples with gold_disagreement = model_hedged_*: {len(hedged_insufficient)}")

if hedged_insufficient:
    print("\n⚠️  Found overlap between Insufficient and hedging:")
    for ex in hedged_insufficient[:5]:
        print(f"   question_id={ex.get('question_id')}, gold_disagreement={ex.get('gold_disagreement')}")
else:
    print("\n✓ No overlap (as expected: Insufficient and Maybe are different generated_answer values)")

# ===== Step 5: Gold answer distribution by mode =====
print("\n" + "="*80)
print("STEP 5: Gold answer distribution for Insufficient evidence yesno examples")
print("="*80)

yesno_insufficient = [ex for ex in insufficient_cases if ex.get("question_type") == "yesno"]

easy_yesno = [ex for ex in yesno_insufficient if ex.get("mode") == "easy"]
hard_yesno = [ex for ex in yesno_insufficient if ex.get("mode") == "hard"]

easy_gold_answers = Counter([ex.get("gold_answer") for ex in easy_yesno])
hard_gold_answers = Counter([ex.get("gold_answer") for ex in hard_yesno])

print(f"\nEasy-mode yesno Insufficient evidence (n={len(easy_yesno)}):")
for answer, count in easy_gold_answers.most_common():
    print(f"  {answer}: {count} ({100*count/len(easy_yesno):.1f}%)")

print(f"\nHard-mode yesno Insufficient evidence (n={len(hard_yesno)}):")
for answer, count in hard_gold_answers.most_common():
    print(f"  {answer}: {count} ({100*count/len(hard_yesno):.1f}%)")

print("\n" + "="*80)
print("INVESTIGATION COMPLETE - AWAITING DECISION")
print("="*80)
