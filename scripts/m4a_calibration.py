"""M4a cost calibration: Generate 10 examples to verify pricing and quality.

Picks 5 questions from BioASQ train (seed 42), generates 1 easy + 1 hard example
for each (10 total), and reports token usage and costs.
"""

import random
from pathlib import Path

from src.data.bioasq import load_bioasq
from src.data.corpus import build_corpus, build_corpus_dict
from src.retrieval.index import load_index
from src.synthetic.build_examples import build_example_for_question
from src.synthetic.lora_b_generator import generate_lora_b_example


def main():
    print("=" * 80)
    print("M4a Calibration: 10-example generation")
    print("=" * 80)
    print()

    # Load BioASQ train
    print("Loading BioASQ train split...")
    train_questions = load_bioasq("train")
    print(f"Loaded {len(train_questions)} training questions")
    print()

    # Pick 5 questions with seed 42
    rng = random.Random(42)
    selected_questions = rng.sample(train_questions, 5)
    print(f"Selected 5 questions for calibration:")
    for i, q in enumerate(selected_questions, 1):
        print(f"  {i}. {q['id']} ({q['question_type']})")
    print()

    # Load corpus and index
    print("Loading corpus and index...")
    corpus_list = build_corpus()
    corpus_dict = build_corpus_dict(corpus_list)
    print(f"Corpus size: {len(corpus_dict)} passages")

    # Use existing pubmedqa index for calibration (good enough for distractor sampling)
    index_path = Path("data/indices/pubmedqa")
    index, id_mapping = load_index(index_path)
    print(f"Index loaded: {index.ntotal} vectors")
    print()

    # Generate examples
    print("=" * 80)
    print("Generating examples...")
    print("=" * 80)
    print()

    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    for i, question in enumerate(selected_questions, 1):
        for mode in ["easy", "hard"]:
            print(f"\n{'=' * 80}")
            print(f"Example {len(all_results) + 1}: {question['id']} ({mode} mode)")
            print(f"{'=' * 80}")

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

            # Track totals
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            total_cost += result["cost_estimate"]

            all_results.append(result)

            # Print details
            print(f"\nQuestion type: {result['question_type']}")
            print(f"Question: {result['question']}")
            print(f"Gold answer: {result['gold_answer']}")
            print()
            print("Input passages:")
            for p in result["passages"]:
                is_gold_flag = "gold=True" if p["is_gold"] else "gold=False"
                text_preview = p["text"][:200] + "..." if len(p["text"]) > 200 else p["text"]
                print(f"  [P{p['position']}] ({is_gold_flag}) {text_preview}")
            print()
            print("Haiku response (raw):")
            # Reconstruct the full response
            if result["generated_reasoning"] and result["generated_answer"]:
                full_response = f"Reasoning: {result['generated_reasoning']}\n\nAnswer: {result['generated_answer']}"
            else:
                full_response = "(Failed to parse response)"
            print(full_response)
            print()
            print(f"Tokens: input={result['input_tokens']}, output={result['output_tokens']}, cost=${result['cost_estimate']:.6f}")

    # Summary
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)
    print(f"Total examples generated: {len(all_results)}")
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Average cost per generation: ${total_cost / len(all_results):.6f}")
    print()

    # Cost variance analysis
    costs = [r["cost_estimate"] for r in all_results]
    min_cost = min(costs)
    max_cost = max(costs)
    print(f"Cost variance: min=${min_cost:.6f}, max=${max_cost:.6f}, range=${max_cost - min_cost:.6f}")
    print()

    # Projected M4b cost
    projected_m4b_cost = (total_cost / len(all_results)) * 5000
    print(f"Projected M4b cost (5,000 generations): ${projected_m4b_cost:.2f}")
    print()

    # Cost check
    if projected_m4b_cost < 4.50:
        print("✓ Projected M4b cost is under $4.50 - within budget!")
    elif 4.50 <= projected_m4b_cost <= 5.50:
        print("⚠ Projected M4b cost is between $4.50 and $5.50 - close to budget limit")
    else:
        print("✗ Projected M4b cost exceeds $5.50 - prompt too expensive, needs revision")
    print()

    # Check for "Insufficient evidence" in hard mode
    hard_mode_results = [r for r in all_results if r["mode"] == "hard"]
    insufficient_count = sum(
        1 for r in hard_mode_results
        if r["generated_answer"] and "insufficient" in r["generated_answer"].lower()
    )
    print(f"Hard mode examples with 'Insufficient evidence': {insufficient_count}/{len(hard_mode_results)}")
    if insufficient_count == 0:
        print("⚠ Warning: No hard-mode examples produced 'Insufficient evidence' - prompt may need revision")
    print()


if __name__ == "__main__":
    main()
