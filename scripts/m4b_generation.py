"""M4b full generation: ~4918 examples (2459 questions × 2 modes).

Generates LoRA-B training data for all BioASQ train questions with incremental
saving and cost tracking.
"""

import json
import time
from pathlib import Path
from collections import defaultdict

from src.data.bioasq import load_bioasq
from src.data.corpus import build_corpus, build_corpus_dict
from src.retrieval.index import load_index
from src.synthetic.build_examples import build_example_for_question
from src.synthetic.lora_b_generator import generate_lora_b_example
from src.synthetic.filters import apply_filters


MAX_TOTAL_GENERATIONS = 2000  # Hard cap on total generations (existing + new)


def main():
    print("=" * 80)
    print("M4b FULL GENERATION")
    print("=" * 80)
    print()

    start_time = time.time()

    # Load BioASQ train - ALL questions
    print("Loading BioASQ train split...")
    train_questions = load_bioasq("train")
    print(f"Loaded {len(train_questions)} training questions")
    print()

    # Load corpus and index
    print("Loading corpus and index...")
    corpus_list = build_corpus()
    corpus_dict = build_corpus_dict(corpus_list)
    print(f"Corpus size: {len(corpus_dict)} passages")

    index_path = Path("data/indices/pubmedqa")
    index, id_mapping = load_index(index_path)
    print(f"Index loaded: {index.ntotal} vectors")
    print()

    # Prepare output file
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_file = output_dir / "lora_b_train_raw.jsonl"

    print(f"Output file: {raw_output_file}")
    print()

    # Resume logic: load existing generations
    existing_completed = set()  # set of (question_id, mode) tuples
    existing_cost = 0.0
    existing_count = 0

    if raw_output_file.exists():
        with open(raw_output_file, "r") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    existing_completed.add((ex["question_id"], ex["mode"]))
                    existing_cost += ex.get("cost_estimate", 0.0)
                    existing_count += 1
                except json.JSONDecodeError:
                    continue

    print(f"Resuming: {existing_count} existing generations on disk")
    print(f"Existing cost: ${existing_cost:.4f}")
    print(f"Target: {MAX_TOTAL_GENERATIONS} total generations")
    print(f"Remaining to generate: {MAX_TOTAL_GENERATIONS - existing_count}")
    print()

    # Generation loop
    print("=" * 80)
    print("GENERATING EXAMPLES")
    print("=" * 80)
    print()

    # Initialize counters from existing state
    total_cost = existing_cost
    completed_generations = existing_count
    skipped_questions = []

    cost_cap = 6.40  # Hard stop at $6.40 (under $6.50 budget)

    for q_idx, question in enumerate(train_questions, 1):
        question_skipped = False

        # For NEW generations, only generate easy mode (faster, simpler).
        # Existing hard-mode generations from earlier runs are preserved.
        modes_for_this_question = ["easy"]
        for mode in modes_for_this_question:
            # Skip if already completed (resume)
            if (question["id"], mode) in existing_completed:
                continue

            # Stop if we've reached the target
            if completed_generations >= MAX_TOTAL_GENERATIONS:
                print(f"Target reached: {completed_generations}/{MAX_TOTAL_GENERATIONS}")
                break

            if total_cost >= cost_cap:
                print()
                print(f"COST CAP REACHED: ${total_cost:.4f} >= ${cost_cap:.2f}")
                print("Stopping generation to stay under budget.")
                break

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

                # Track cost
                total_cost += result["cost_estimate"]
                completed_generations += 1

                # Save immediately (incremental save)
                with open(raw_output_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
                    f.flush()

                # Progress reporting every 100 generations
                if completed_generations % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = completed_generations / elapsed if elapsed > 0 else 0
                    remaining = len(train_questions) * 2 - completed_generations
                    eta_minutes = (remaining / rate / 60) if rate > 0 else 0

                    avg_cost = total_cost / completed_generations

                    print(f"Progress: {completed_generations}/{len(train_questions) * 2} "
                          f"| Cost: ${total_cost:.4f} "
                          f"| Avg: ${avg_cost:.6f}/gen "
                          f"| ETA: {eta_minutes:.1f}min")

            except ValueError as e:
                # Insufficient distractors - skip both modes for this question
                if not question_skipped:
                    skipped_questions.append({
                        "question_id": question["id"],
                        "reason": str(e)
                    })
                    question_skipped = True
                break  # Skip the other mode for this question

        # Check cost cap after each question
        if total_cost >= cost_cap:
            break

        # Check generation target after each question
        if completed_generations >= MAX_TOTAL_GENERATIONS:
            break

    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print()

    elapsed_time = time.time() - start_time
    print(f"Wall clock time: {elapsed_time / 60:.1f} minutes")
    print(f"Total questions in BioASQ train: {len(train_questions)}")
    print(f"Total generations completed: {completed_generations}")
    print(f"Total questions skipped (insufficient distractors): {len(skipped_questions)}")
    print(f"Total raw cost: ${total_cost:.4f}")
    print(f"Average cost per generation: ${total_cost / completed_generations:.6f}")
    print()

    if skipped_questions:
        print("Sample skipped questions (first 5):")
        for sq in skipped_questions[:5]:
            print(f"  - {sq['question_id']}: {sq['reason']}")
        if len(skipped_questions) > 5:
            print(f"  ... ({len(skipped_questions) - 5} more)")
        print()

    # Apply filters to all generated examples
    print("=" * 80)
    print("APPLYING FILTERS")
    print("=" * 80)
    print()

    print(f"Loading raw generations from {raw_output_file}...")
    raw_examples = []
    with open(raw_output_file, "r") as f:
        for line in f:
            raw_examples.append(json.loads(line))

    print(f"Loaded {len(raw_examples)} raw examples")
    print()

    print("Applying filters...")
    for example in raw_examples:
        apply_filters(example)

    print("Filters applied")
    print()

    # Save filtered results
    filtered_output_file = output_dir / "lora_b_train.jsonl"
    passing_examples = [ex for ex in raw_examples if ex["filter_passed"]]

    with open(filtered_output_file, "w") as f:
        for example in passing_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Saved {len(passing_examples)} passing examples to {filtered_output_file}")
    print()

    # Compute filter statistics
    print("=" * 80)
    print("FILTERING RESULTS")
    print("=" * 80)
    print()

    failed_examples = [ex for ex in raw_examples if not ex["filter_passed"]]

    print(f"Total raw generations: {len(raw_examples)}")
    print(f"Total passing: {len(passing_examples)} ({len(passing_examples) / len(raw_examples) * 100:.1f}%)")
    print(f"Total failed: {len(failed_examples)} ({len(failed_examples) / len(raw_examples) * 100:.1f}%)")
    print()

    # Rejection counts by filter
    rejection_counts = defaultdict(int)
    for ex in failed_examples:
        filter_name = ex["filter_failed_at"]
        rejection_counts[filter_name] += 1

    print("Rejections by filter:")
    for filter_name in ["format", "citations", "citation_validity", "reasoning_length", "answer_format", "gold_answer_agreement"]:
        count = rejection_counts[filter_name]
        print(f"  {filter_name}: {count}")
    print()

    # Pass rate by mode
    easy_passing = [ex for ex in passing_examples if ex["mode"] == "easy"]
    easy_total = [ex for ex in raw_examples if ex["mode"] == "easy"]
    hard_passing = [ex for ex in passing_examples if ex["mode"] == "hard"]
    hard_total = [ex for ex in raw_examples if ex["mode"] == "hard"]

    print("Pass rate by mode:")
    print(f"  easy: {len(easy_passing)}/{len(easy_total)} ({len(easy_passing) / len(easy_total) * 100:.1f}%)")
    print(f"  hard: {len(hard_passing)}/{len(hard_total)} ({len(hard_passing) / len(hard_total) * 100:.1f}%)")
    print()

    # Pass rate by question type
    factoid_passing = [ex for ex in passing_examples if ex["question_type"] == "factoid"]
    factoid_total = [ex for ex in raw_examples if ex["question_type"] == "factoid"]
    yesno_passing = [ex for ex in passing_examples if ex["question_type"] == "yesno"]
    yesno_total = [ex for ex in raw_examples if ex["question_type"] == "yesno"]

    print("Pass rate by question type:")
    print(f"  factoid: {len(factoid_passing)}/{len(factoid_total)} ({len(factoid_passing) / len(factoid_total) * 100:.1f}%)")
    print(f"  yesno: {len(yesno_passing)}/{len(yesno_total)} ({len(yesno_passing) / len(yesno_total) * 100:.1f}%)")
    print()

    print("=" * 80)
    print("M4b GENERATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Files saved:")
    print(f"  - {raw_output_file} ({len(raw_examples)} raw examples)")
    print(f"  - {filtered_output_file} ({len(passing_examples)} passing examples)")
    print()


if __name__ == "__main__":
    main()
