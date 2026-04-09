"""Inspect the 4 confident-answer-without-citations cases in detail."""

import json
import re
from pathlib import Path


def main():
    # Load pilot data
    pilot_file = Path("data/synthetic/lora_b_pilot.jsonl")
    examples = []
    with open(pilot_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    # Find the 4 specific cases (from diagnostic output)
    target_cases = [
        ("517901bc8ed59a060a00003b", "hard"),
        ("58bc5e2202b8c60953000002", "easy"),
        ("5e776db8835f4e4777000011", "hard"),
        ("5a896c26fcd1d6a10c000007", "hard"),
    ]

    print("=" * 80)
    print("INSPECTING 4 CONFIDENT-ANSWER-WITHOUT-CITATIONS CASES")
    print("=" * 80)
    print()

    for i, (qid, mode) in enumerate(target_cases, 1):
        # Find the exact example (question_id + mode)
        matching = [
            ex for ex in examples
            if ex["question_id"] == qid
            and ex["mode"] == mode
            and not ex["filter_passed"]
            and ex["filter_failed_at"] == "citations"
        ]

        if not matching:
            print(f"--- Case {i}: {qid} ({mode}) ---")
            print("NOT FOUND (may have passed or been rejected by different filter)")
            print()
            continue

        ex = matching[0]  # Should be exactly one match
        print(f"--- Case {i}: {qid} ({ex['mode']} mode) ---")
        print()
        print(f"Question: {ex['question']}")
        print(f"Question type: {ex['question_type']}")
        print(f"Gold answer: {ex['gold_answer']}")
        print(f"Generated answer: {ex['generated_answer']}")
        print()

        print("Input passages:")
        for p in ex['passages']:
            text_preview = p['text'][:200] + "..." if len(p['text']) > 200 else p['text']
            print(f"  [P{p['position']}] (gold={p['is_gold']}) {text_preview}")
        print()

        print("Generated reasoning (FULL TEXT):")
        reasoning = ex.get("generated_reasoning", "")
        print(f'"""{reasoning}"""')
        print()

        # Count citations using exact regex
        citations = re.findall(r'\[P[1-5]\]', reasoning)
        print(f"Citation marker count (exact regex r'\\[P[1-5]\\]'): {len(citations)}")
        if citations:
            print(f"  Found markers: {citations}")
        else:
            print(f"  No [P1]-[P5] markers found")

        # Check for alternative citation formats
        alt_formats = {
            "(P1)-(P5)": re.findall(r'\(P[1-5]\)', reasoning),
            "Comma-separated [P1, P2]": re.findall(r'\[P[1-5](?:,\s*P[1-5])+\]', reasoning),
            "P1-P5 without brackets": re.findall(r'(?<!\[)P[1-5](?!\])', reasoning),
            "passage 1-5": re.findall(r'passage [1-5]', reasoning, re.IGNORECASE),
        }

        print()
        print("Alternative citation formats found:")
        for format_name, matches in alt_formats.items():
            if matches:
                print(f"  {format_name}: {matches}")
        if not any(alt_formats.values()):
            print("  None")

        print()
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()
