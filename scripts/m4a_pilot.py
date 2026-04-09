"""M4a pilot: Generate 50 examples with filtering.

Picks 25 questions from BioASQ train (seed 42), generates 1 easy + 1 hard for each,
applies filters, and saves to data/synthetic/lora_b_pilot.jsonl.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

from src.data.bioasq import load_bioasq
from src.data.corpus import build_corpus, build_corpus_dict
from src.retrieval.index import load_index
from src.synthetic.build_examples import build_example_for_question
from src.synthetic.lora_b_generator import generate_lora_b_example
from src.synthetic.filters import apply_filters


def main():
    print("=" * 80)
    print("M4a Pilot: 50-example generation with filtering")
    print("=" * 80)
    print()

    # Load BioASQ train
    print("Loading BioASQ train split...")
    train_questions = load_bioasq("train")
    print(f"Loaded {len(train_questions)} training questions")
    print()

    # Pick 25 questions with seed 42
    # (pick extra in case some don't have enough distractors)
    rng = random.Random(42)
    candidate_questions = rng.sample(train_questions, min(100, len(train_questions)))
    selected_questions = []
    skipped_questions = []

    print(f"Selecting 25 questions for pilot (checking for sufficient distractors)...")
    print()

    # We'll verify distractor availability during generation and skip if needed

    # Load corpus and index
    print("Loading corpus and index...")
    corpus_list = build_corpus()
    corpus_dict = build_corpus_dict(corpus_list)
    print(f"Corpus size: {len(corpus_dict)} passages")

    index_path = Path("data/indices/pubmedqa")
    index, id_mapping = load_index(index_path)
    print(f"Index loaded: {index.ntotal} vectors")
    print()

    # Generate examples
    print("=" * 80)
    print("Generating 50 examples...")
    print("=" * 80)
    print()

    all_results = []
    total_cost = 0.0
    question_idx = 0

    while len(all_results) < 50 and question_idx < len(candidate_questions):
        question = candidate_questions[question_idx]

        # Try both modes for this question
        modes_to_try = ["easy", "hard"]
        question_success = True

        for mode in modes_to_try:
            if len(all_results) >= 50:
                break

            example_num = len(all_results) + 1
            print(f"Example {example_num}/50: {question['id']} ({mode} mode)...", end=" ", flush=True)

            try:
                # Build example input
                example_input = build_example_for_question(
                    bioasq_question=question,
                    index=index,
                    id_mapping=id_mapping,
                    corpus_dict=corpus_dict,
                    mode=mode,
                )

                # Generate
                result = generate_lora_b_example(
                    question_dict=example_input["question_dict"],
                    gold_snippets=example_input["gold_snippets"],
                    distractor_passages=example_input["distractor_passages"],
                    mode=mode,
                )

                total_cost += result["cost_estimate"]
                all_results.append(result)

                print(f"${result['cost_estimate']:.6f}")

            except ValueError as e:
                # Insufficient distractors, skip this question
                print(f"SKIP ({e})")
                skipped_questions.append((question['id'], str(e)))
                question_success = False
                break  # Skip both modes for this question

        if question_success and question['id'] not in [q['question_id'] for q in all_results]:
            selected_questions.append(question)

        question_idx += 1

    print()
    print(f"Generation complete. Total cost: ${total_cost:.4f}")
    if skipped_questions:
        print(f"Skipped {len(skipped_questions)} questions due to insufficient distractors:")
        for qid, reason in skipped_questions[:5]:
            print(f"  - {qid}: {reason}")
        if len(skipped_questions) > 5:
            print(f"  ... ({len(skipped_questions) - 5} more)")
    print()

    # Apply filters
    print("=" * 80)
    print("Applying filters...")
    print("=" * 80)
    print()

    for example in all_results:
        apply_filters(example)

    # Save to file
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "lora_b_pilot.jsonl"

    with open(output_file, "w") as f:
        for example in all_results:
            f.write(json.dumps(example) + "\n")

    print(f"Saved {len(all_results)} examples to {output_file}")
    print()

    # Compute statistics
    print("=" * 80)
    print("FILTERING RESULTS")
    print("=" * 80)
    print()

    passed = [ex for ex in all_results if ex["filter_passed"]]
    failed = [ex for ex in all_results if not ex["filter_passed"]]

    print(f"Total examples generated: {len(all_results)}")
    print(f"Total examples passed: {len(passed)} ({len(passed) / len(all_results) * 100:.1f}%)")
    print(f"Total examples failed: {len(failed)} ({len(failed) / len(all_results) * 100:.1f}%)")
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
    print()

    # Pass rate by mode
    easy_passed = sum(1 for ex in passed if ex["mode"] == "easy")
    easy_total = sum(1 for ex in all_results if ex["mode"] == "easy")
    hard_passed = sum(1 for ex in passed if ex["mode"] == "hard")
    hard_total = sum(1 for ex in all_results if ex["mode"] == "hard")

    print("Pass rate by mode:")
    print(f"  easy: {easy_passed}/{easy_total} ({easy_passed / easy_total * 100:.1f}%)")
    print(f"  hard: {hard_passed}/{hard_total} ({hard_passed / hard_total * 100:.1f}%)")
    print()

    # Pass rate by question type
    factoid_passed = sum(1 for ex in passed if ex["question_type"] == "factoid")
    factoid_total = sum(1 for ex in all_results if ex["question_type"] == "factoid")
    yesno_passed = sum(1 for ex in passed if ex["question_type"] == "yesno")
    yesno_total = sum(1 for ex in all_results if ex["question_type"] == "yesno")

    print("Pass rate by question type:")
    print(f"  factoid: {factoid_passed}/{factoid_total} ({factoid_passed / factoid_total * 100:.1f}%)")
    print(f"  yesno: {yesno_passed}/{yesno_total} ({yesno_passed / yesno_total * 100:.1f}%)")
    print()

    # Cost tracking
    # Note: calibration cost from previous runs was $0.0118
    calibration_cost = 0.0118
    total_m4a_cost = calibration_cost + total_cost
    remaining_budget = 6.50 - total_m4a_cost

    print("=" * 80)
    print("COST TRACKING")
    print("=" * 80)
    print(f"Calibration cost (10 examples): ${calibration_cost:.4f}")
    print(f"Pilot cost (50 examples): ${total_cost:.4f}")
    print(f"Total M4a cost: ${total_m4a_cost:.4f}")
    print(f"Revised budget: $6.50")
    print(f"Remaining budget: ${remaining_budget:.4f}")
    print()

    # Sample passing examples (5 random, seed 42)
    if passed:
        sample_rng = random.Random(42)
        sample_passed = sample_rng.sample(passed, min(5, len(passed)))

        print("=" * 80)
        print("SAMPLE PASSING EXAMPLES (5 random, seed 42)")
        print("=" * 80)
        for i, ex in enumerate(sample_passed, 1):
            print(f"\n--- Passing Example {i} ---")
            print(f"Question ID: {ex['question_id']}")
            print(f"Question type: {ex['question_type']}")
            print(f"Mode: {ex['mode']}")
            print(f"Question: {ex['question']}")
            print(f"Gold answer: {ex['gold_answer']}")
            print()
            print("Input passages:")
            for p in ex['passages']:
                text_preview = p['text'][:200] + "..." if len(p['text']) > 200 else p['text']
                print(f"  [P{p['position']}] (gold={p['is_gold']}) {text_preview}")
            print()
            print(f"Generated reasoning: {ex['generated_reasoning']}")
            print(f"Generated answer: {ex['generated_answer']}")
            print()

    # Sample rejected examples (3, deterministic selection)
    if failed:
        sample_failed = failed[:3]  # First 3 failures

        print("=" * 80)
        print("SAMPLE REJECTED EXAMPLES (first 3)")
        print("=" * 80)
        for i, ex in enumerate(sample_failed, 1):
            print(f"\n--- Rejected Example {i} ---")
            print(f"Question ID: {ex['question_id']}")
            print(f"Mode: {ex['mode']}")
            print(f"Question: {ex['question']}")
            print(f"Generated answer: {ex.get('generated_answer', 'N/A')}")
            print(f"Gold answer: {ex['gold_answer']}")
            print(f"Rejection filter: {ex['filter_failed_at']}")
            print(f"Rejection reason: {ex['filter_failure_reason']}")
            print()


if __name__ == "__main__":
    main()
