"""Re-apply corrected filters to existing pilot examples."""

import json
from pathlib import Path
from collections import defaultdict

from src.synthetic.filters import apply_filters


def main():
    print("=" * 80)
    print("RE-FILTERING PILOT EXAMPLES WITH CORRECTED FILTER")
    print("=" * 80)
    print()

    # Load original pilot data
    pilot_file = Path("data/synthetic/lora_b_pilot.jsonl")
    examples = []
    with open(pilot_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples from {pilot_file}")
    print()

    # Re-apply filters
    print("Re-applying filters with bug fix...")
    for example in examples:
        # Clear old filter results
        example.pop("filter_passed", None)
        example.pop("filter_failed_at", None)
        example.pop("filter_failure_reason", None)

        # Re-apply filters
        apply_filters(example)

    print("Filters re-applied")
    print()

    # Save corrected results
    output_file = Path("data/synthetic/lora_b_pilot_corrected.jsonl")
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Saved corrected results to {output_file}")
    print()

    # Compute new statistics
    print("=" * 80)
    print("CORRECTED FILTERING RESULTS")
    print("=" * 80)
    print()

    passed = [ex for ex in examples if ex["filter_passed"]]
    failed = [ex for ex in examples if not ex["filter_passed"]]

    print(f"Total examples: {len(examples)}")
    print(f"Total passed: {len(passed)} ({len(passed) / len(examples) * 100:.1f}%)")
    print(f"Total failed: {len(failed)} ({len(failed) / len(examples) * 100:.1f}%)")
    print()

    # Compare to original
    original_passed = 36
    original_failed = 14
    print(f"Change from original:")
    print(f"  Passed: {original_passed} → {len(passed)} ({len(passed) - original_passed:+d})")
    print(f"  Failed: {original_failed} → {len(failed)} ({len(failed) - original_failed:+d})")
    print()

    # Rejection counts by filter
    rejection_counts = defaultdict(int)
    for ex in failed:
        filter_name = ex["filter_failed_at"]
        rejection_counts[filter_name] += 1

    print("Rejections by filter:")
    for filter_name in ["format", "citations", "citation_validity", "reasoning_length", "answer_format", "gold_answer_agreement"]:
        count = rejection_counts[filter_name]
        if count > 0:
            print(f"  {filter_name}: {count}")
        else:
            print(f"  {filter_name}: 0")
    print()

    # Pass rate by mode
    easy_passed = sum(1 for ex in passed if ex["mode"] == "easy")
    easy_total = sum(1 for ex in examples if ex["mode"] == "easy")
    hard_passed = sum(1 for ex in passed if ex["mode"] == "hard")
    hard_total = sum(1 for ex in examples if ex["mode"] == "hard")

    print("Pass rate by mode:")
    print(f"  easy: {easy_passed}/{easy_total} ({easy_passed / easy_total * 100:.1f}%)")
    print(f"  hard: {hard_passed}/{hard_total} ({hard_passed / hard_total * 100:.1f}%)")
    print()

    # Compare to original
    print("Change from original:")
    print(f"  easy: 20/25 (80.0%) → {easy_passed}/{easy_total} ({easy_passed / easy_total * 100:.1f}%) [{easy_passed - 20:+d}]")
    print(f"  hard: 16/25 (64.0%) → {hard_passed}/{hard_total} ({hard_passed / hard_total * 100:.1f}%) [{hard_passed - 16:+d}]")
    print()

    # Pass rate by question type
    factoid_passed = sum(1 for ex in passed if ex["question_type"] == "factoid")
    factoid_total = sum(1 for ex in examples if ex["question_type"] == "factoid")
    yesno_passed = sum(1 for ex in passed if ex["question_type"] == "yesno")
    yesno_total = sum(1 for ex in examples if ex["question_type"] == "yesno")

    print("Pass rate by question type:")
    print(f"  factoid: {factoid_passed}/{factoid_total} ({factoid_passed / factoid_total * 100:.1f}%)")
    print(f"  yesno: {yesno_passed}/{yesno_total} ({yesno_passed / yesno_total * 100:.1f}%)")
    print()

    # Compare to original
    print("Change from original:")
    print(f"  factoid: 13/22 (59.1%) → {factoid_passed}/{factoid_total} ({factoid_passed / factoid_total * 100:.1f}%) [{factoid_passed - 13:+d}]")
    print(f"  yesno: 23/28 (82.1%) → {yesno_passed}/{yesno_total} ({yesno_passed / yesno_total * 100:.1f}%) [{yesno_passed - 23:+d}]")
    print()


if __name__ == "__main__":
    main()
