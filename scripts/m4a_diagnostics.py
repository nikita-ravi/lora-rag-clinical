"""M4a diagnostics: Analyze pilot results without generating new examples."""

import json
import random
import re
from pathlib import Path
from collections import Counter


def load_pilot_data():
    """Load pilot examples from jsonl file."""
    pilot_file = Path("data/synthetic/lora_b_pilot.jsonl")
    examples = []
    with open(pilot_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def diagnostic_1_citation_rejections(examples):
    """Analyze the 12 citation-filter rejections."""
    print("=" * 80)
    print("DIAGNOSTIC 1: Citation Filter Rejections")
    print("=" * 80)
    print()

    # Find all examples rejected by citations filter
    citation_rejects = [
        ex for ex in examples
        if not ex["filter_passed"] and ex["filter_failed_at"] == "citations"
    ]

    print(f"Found {len(citation_rejects)} examples rejected by citations filter\n")

    # Analyze each rejection
    print("| Question ID | Mode | Type | Generated Answer | Is 'Insufficient evidence'? |")
    print("|-------------|------|------|------------------|----------------------------|")

    insufficient_count = 0
    confident_count = 0

    for ex in citation_rejects:
        qid = ex["question_id"]
        mode = ex["mode"]
        qtype = ex["question_type"]
        answer = ex.get("generated_answer", "N/A")

        # Check if answer is "Insufficient evidence" (case-insensitive)
        is_insufficient = "insufficient evidence" in answer.lower() if answer else False

        if is_insufficient:
            insufficient_count += 1
            insufficient_flag = "YES"
        else:
            confident_count += 1
            insufficient_flag = "NO"

        # Truncate long answers for table display
        answer_display = answer[:50] + "..." if len(answer) > 50 else answer

        print(f"| {qid[:20]}... | {mode:4} | {qtype:7} | {answer_display:16} | {insufficient_flag:27} |")

    print()
    print(f"Summary:")
    print(f"  'Insufficient evidence' cases: {insufficient_count}/{len(citation_rejects)}")
    print(f"  Confident answer without citations: {confident_count}/{len(citation_rejects)}")
    print()


def diagnostic_2_sample_examples(examples):
    """Show 5 passing factoid and 5 passing yesno examples."""
    print("=" * 80)
    print("DIAGNOSTIC 2: Sample Passing Examples by Question Type")
    print("=" * 80)
    print()

    # Filter to passing examples
    passing = [ex for ex in examples if ex["filter_passed"]]

    # Separate by question type
    factoid_passing = [ex for ex in passing if ex["question_type"] == "factoid"]
    yesno_passing = [ex for ex in passing if ex["question_type"] == "yesno"]

    print(f"Total passing factoid: {len(factoid_passing)}")
    print(f"Total passing yesno: {len(yesno_passing)}")
    print()

    # Sample 5 of each (seed 43)
    rng = random.Random(43)
    factoid_sample = rng.sample(factoid_passing, min(5, len(factoid_passing)))
    yesno_sample = rng.sample(yesno_passing, min(5, len(yesno_passing)))

    # Show factoid examples
    print("=" * 80)
    print("PASSING FACTOID EXAMPLES (5 random, seed 43)")
    print("=" * 80)
    print()

    for i, ex in enumerate(factoid_sample, 1):
        print(f"--- Factoid Example {i} ---")
        print(f"Question ID: {ex['question_id']}")
        print(f"Mode: {ex['mode']}")
        print(f"Question: {ex['question']}")
        print(f"Gold answer: {ex['gold_answer']}")
        print(f"Generated answer: {ex['generated_answer']}")
        print(f"Generated reasoning: {ex['generated_reasoning']}")
        print()

        # Citation analysis
        citations = re.findall(r"\[P\d\]", ex['generated_reasoning'])
        print(f"  → Citations used: {', '.join(citations) if citations else 'None'}")
        print()

    # Show yesno examples
    print("=" * 80)
    print("PASSING YESNO EXAMPLES (5 random, seed 43)")
    print("=" * 80)
    print()

    for i, ex in enumerate(yesno_sample, 1):
        print(f"--- Yes/No Example {i} ---")
        print(f"Question ID: {ex['question_id']}")
        print(f"Mode: {ex['mode']}")
        print(f"Question: {ex['question']}")
        print(f"Gold answer: {ex['gold_answer']}")
        print(f"Generated answer: {ex['generated_answer']}")
        print(f"Generated reasoning: {ex['generated_reasoning']}")
        print()

        # Citation analysis
        citations = re.findall(r"\[P\d\]", ex['generated_reasoning'])
        print(f"  → Citations used: {', '.join(citations) if citations else 'None'}")
        print()


def diagnostic_3_citation_density(examples):
    """Analyze citation density in passing examples."""
    print("=" * 80)
    print("DIAGNOSTIC 3: Citation Density in Passing Examples")
    print("=" * 80)
    print()

    # Filter to passing examples
    passing = [ex for ex in examples if ex["filter_passed"]]

    citation_counts = []
    for ex in passing:
        reasoning = ex.get("generated_reasoning", "")
        citations = re.findall(r"\[P\d\]", reasoning)
        citation_counts.append(len(citations))

    if not citation_counts:
        print("No passing examples found!")
        return

    # Compute statistics
    citation_counts_sorted = sorted(citation_counts)
    n = len(citation_counts_sorted)
    min_cites = min(citation_counts)
    max_cites = max(citation_counts)
    median_cites = citation_counts_sorted[n // 2]
    mean_cites = sum(citation_counts) / n

    # Distribution
    cite_distribution = Counter(citation_counts)

    print(f"Total passing examples: {n}")
    print()
    print(f"Citation count statistics:")
    print(f"  Min: {min_cites}")
    print(f"  Median: {median_cites}")
    print(f"  Mean: {mean_cites:.1f}")
    print(f"  Max: {max_cites}")
    print()
    print(f"Distribution:")
    for count in sorted(cite_distribution.keys()):
        num_examples = cite_distribution[count]
        print(f"  {count} citations: {num_examples} examples ({num_examples / n * 100:.1f}%)")
    print()

    # Categorize
    single_citation = sum(1 for c in citation_counts if c == 1)
    two_citations = sum(1 for c in citation_counts if c == 2)
    three_plus = sum(1 for c in citation_counts if c >= 3)

    print(f"Categorization:")
    print(f"  Only 1 citation: {single_citation} examples ({single_citation / n * 100:.1f}%)")
    print(f"  Exactly 2 citations: {two_citations} examples ({two_citations / n * 100:.1f}%)")
    print(f"  3+ citations: {three_plus} examples ({three_plus / n * 100:.1f}%)")
    print()


def main():
    print("Loading pilot data from data/synthetic/lora_b_pilot.jsonl...")
    examples = load_pilot_data()
    print(f"Loaded {len(examples)} examples")
    print()

    # Run diagnostics
    diagnostic_1_citation_rejections(examples)
    diagnostic_2_sample_examples(examples)
    diagnostic_3_citation_density(examples)

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    citation_rejects = [
        ex for ex in examples
        if not ex["filter_passed"] and ex["filter_failed_at"] == "citations"
    ]

    insufficient_count = sum(
        1 for ex in citation_rejects
        if "insufficient evidence" in ex.get("generated_answer", "").lower()
    )

    print("Citation Filter Bug Analysis:")
    print()
    print(f"Of {len(citation_rejects)} citation-filter rejections:")
    print(f"  - {insufficient_count} are 'Insufficient evidence' cases")
    print(f"  - {len(citation_rejects) - insufficient_count} are confident answers without citations")
    print()

    if insufficient_count >= 9:
        print("VERDICT: The citation filter has a bug.")
        print()
        print("The filter is incorrectly rejecting 'Insufficient evidence' responses that")
        print("legitimately have no citations (because there's nothing to cite when passages")
        print("don't support an answer). These should auto-pass the citations filter.")
        print()
        print("RECOMMENDED FIX: Add exception to filter_citations() to auto-pass when the")
        print("answer is 'Insufficient evidence'.")
    elif len(citation_rejects) - insufficient_count >= 5:
        print("VERDICT: Haiku has a prompt compliance issue.")
        print()
        print("A significant number of examples produce confident answers without proper")
        print("citation formatting. This suggests the prompt isn't reliably enforcing the")
        print("[P1], [P2], etc. citation format.")
        print()
        print("RECOMMENDED FIX: Revise prompt to more strongly enforce citation format,")
        print("or add examples showing the exact format required.")
    else:
        print("VERDICT: Mixed case.")
        print()
        print("Both 'Insufficient evidence' false rejections and genuine formatting lapses")
        print("are present. Recommend applying the citation filter exception first, then")
        print("re-evaluating if prompt changes are needed.")


if __name__ == "__main__":
    main()
