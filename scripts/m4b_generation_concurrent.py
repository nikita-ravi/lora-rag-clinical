"""M4b concurrent generation: ~4918 examples (2459 questions × 2 modes).

Uses ThreadPoolExecutor to parallelize retrieval + generation, bringing wall clock
from ~10 hours (sequential) to ~80-100 minutes (concurrent with 10 workers).

Resumes from existing progress in data/synthetic/lora_b_train_raw.jsonl.
"""

import json
import time
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data.bioasq import load_bioasq
from src.data.corpus import build_corpus, build_corpus_dict
from src.retrieval.index import load_index
from src.synthetic.build_examples import build_example_for_question
from src.synthetic.lora_b_generator import generate_lora_b_example
from src.synthetic.filters import apply_filters


# Global state with locks for thread-safe updates
file_lock = threading.Lock()
cost_lock = threading.Lock()
total_cost = 0.0
completed_count = 0

# Configuration
COST_CAP = 6.40  # Hard stop at $6.40 (under $6.50 budget)
MAX_WORKERS = 10  # Concurrent workers


def load_existing_progress(raw_file: Path) -> set[tuple[str, str]]:
    """Returns set of (question_id, mode) tuples already completed."""
    if not raw_file.exists():
        return set()

    completed = set()
    with open(raw_file, "r") as f:
        for line in f:
            try:
                ex = json.loads(line)
                completed.add((ex["question_id"], ex["mode"]))
            except json.JSONDecodeError:
                # Last line might be mid-write corrupted from sequential run
                # Skip it and continue
                continue

    return completed


def generate_one(question, mode, index, id_mapping, corpus_dict, raw_file):
    """Worker function: build example, generate, save.

    Returns:
        - ("success", result_dict, local_count, local_cost) on success
        - ("skip", question_id, reason) if insufficient distractors
        - ("error", question_id, error_str) on API/other failure
        - None if cost cap already hit
    """
    global total_cost, completed_count

    try:
        # Check cost cap before doing work
        with cost_lock:
            if total_cost >= COST_CAP:
                return None

        # Build example
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

        # Update cost atomically
        with cost_lock:
            total_cost += result["cost_estimate"]
            completed_count += 1
            local_count = completed_count
            local_cost = total_cost

        # Save atomically (thread-safe file write)
        with file_lock:
            with open(raw_file, "a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()

        return ("success", result, local_count, local_cost)

    except ValueError as e:
        # Insufficient distractors
        return ("skip", question["id"], str(e))

    except Exception as e:
        # API error, network error, any other failure
        return ("error", question["id"], str(e))


def main(test_limit=None):
    """Run M4b concurrent generation.

    Args:
        test_limit: If set, only generate this many new examples (for smoke testing)
    """
    global total_cost, completed_count

    print("=" * 80)
    print("M4b CONCURRENT GENERATION")
    if test_limit:
        print(f"TEST MODE: Limited to {test_limit} new generations")
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

    # Resume logic: load existing progress
    already_completed = load_existing_progress(raw_output_file)
    print(f"Found {len(already_completed)} already-completed (question_id, mode) pairs")
    print(f"Will skip these and resume from where the previous run left off")
    print()

    # Initialize global cost from existing progress
    if already_completed:
        # Recompute cost from existing examples for accurate cost cap tracking
        cost_so_far = 0.0
        with open(raw_output_file, "r") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    cost_so_far += ex["cost_estimate"]
                except:
                    pass
        total_cost = cost_so_far
        completed_count = len(already_completed)
        print(f"Cost so far from existing progress: ${total_cost:.4f}")
        print()

    # Build work queue
    work_items = []
    for question in train_questions:
        for mode in ["easy", "hard"]:
            if (question["id"], mode) not in already_completed:
                work_items.append((question, mode))

    print(f"Total work items to process: {len(work_items)}")
    if test_limit and test_limit < len(work_items):
        work_items = work_items[:test_limit]
        print(f"TEST MODE: Limiting to first {test_limit} items")
    print()

    # Run concurrent generation
    print("=" * 80)
    print("GENERATING EXAMPLES (CONCURRENT)")
    print("=" * 80)
    print()

    concurrent_start = time.time()
    skipped = []
    errored = []
    success_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all work
        futures = {
            executor.submit(generate_one, q, m, index, id_mapping, corpus_dict, raw_output_file): (q["id"], m)
            for (q, m) in work_items
        }

        # Process results as they complete
        for future in as_completed(futures):
            result = future.result()

            if result is None:
                # Cost cap hit, worker returned early
                continue

            result_type = result[0]

            if result_type == "skip":
                _, qid, reason = result
                skipped.append((qid, reason))
                continue

            if result_type == "error":
                _, qid, error_str = result
                errored.append((qid, error_str))
                continue

            # Success
            _, gen_result, local_count, local_cost = result
            success_count += 1

            # Progress every 50 generations
            if local_count % 50 == 0:
                elapsed = time.time() - concurrent_start
                rate = success_count / elapsed * 60 if elapsed > 0 else 0
                remaining = len(work_items) - success_count
                eta_min = remaining / rate if rate > 0 else 0
                print(f"[Progress] Done: {local_count}/{len(train_questions)*2} "
                      f"| Cost: ${local_cost:.4f} "
                      f"| Rate: {rate:.1f}/min "
                      f"| ETA: {eta_min:.1f} min")

    concurrent_elapsed = time.time() - concurrent_start
    total_elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Wall clock (total): {total_elapsed / 60:.1f} minutes")
    print(f"Wall clock (concurrent generation): {concurrent_elapsed / 60:.1f} minutes")
    print(f"Total completed: {completed_count}")
    print(f"New completions this run: {success_count}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Average cost per generation: ${total_cost / completed_count:.6f}")
    print(f"Skipped (insufficient distractors): {len(skipped)}")
    print(f"Errored (API/other failures): {len(errored)}")
    print()

    if skipped:
        print("Sample skipped questions (first 5):")
        for qid, reason in skipped[:5]:
            print(f"  - {qid}: {reason[:80]}...")
        if len(skipped) > 5:
            print(f"  ... ({len(skipped) - 5} more)")
        print()

    if errored:
        print("Sample errors (first 5):")
        for qid, error_str in errored[:5]:
            print(f"  - {qid}: {error_str[:80]}...")
        if len(errored) > 5:
            print(f"  ... ({len(errored) - 5} more)")
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
            try:
                raw_examples.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip any malformed lines
                continue

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
    print("M4b CONCURRENT GENERATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Files saved:")
    print(f"  - {raw_output_file} ({len(raw_examples)} raw examples)")
    print(f"  - {filtered_output_file} ({len(passing_examples)} passing examples)")
    print()


if __name__ == "__main__":
    import sys
    # Check for test mode argument
    test_limit = None
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_limit = 20  # 20-example smoke test

    main(test_limit=test_limit)
