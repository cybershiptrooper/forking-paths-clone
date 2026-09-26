"""Classify mask sentences using the Thought Anchors taxonomy (Bogdan et al., 2506.19143).

Adds ``function_tags`` and ``depends_on`` fields to each sentence dict
in a mask JSON file, replicating the Appendix E annotation methodology.

Usage:
    python expts/classify_sentences.py --mask results/circuit_discovery/test_global_2.json
    python expts/classify_sentences.py --mask results/circuit_discovery/test_global_2.json --model openai/gpt-4o
"""

import argparse
import json
import os
import re
import sys

VALID_TAGS = {
    "problem_setup",
    "plan_generation",
    "fact_retrieval",
    "active_computation",
    "uncertainty_management",
    "result_consolidation",
    "self_checking",
    "final_answer_emission",
    "unknown",
}

SYSTEM_PROMPT = """\
You are an expert annotator of chain-of-thought reasoning traces produced by large language models.

Your task is to classify each sentence in a reasoning trace into one or more functional categories, and to identify which earlier sentences each sentence depends on.

## Categories

1. **problem_setup** — Parsing or rephrasing the problem statement.
   Example: "I need to find the area of a circle with radius 5 cm."

2. **plan_generation** — Stating or deciding on a plan of action, meta-reasoning about approach.
   Example: "I'll solve this by applying the area formula."

3. **fact_retrieval** — Recalling facts, formulas, or problem details without performing computation.
   Example: "The formula for the area of a circle is A = pi * r^2."

4. **active_computation** — Performing algebra, arithmetic, calculations, or other manipulations toward the answer.
   Example: "Substituting r = 5: A = pi * 25 = 25pi."

5. **uncertainty_management** — Expressing confusion, re-evaluating, backtracking, or hedging.
   Example: "Wait, I made a mistake earlier. Let me reconsider..."

6. **result_consolidation** — Aggregating intermediate results, summarizing progress, or preparing for the final answer.
   Example: "So the area is 25pi square cm, which is approximately 78.54 square cm."

7. **self_checking** — Verifying previous steps, checking calculations, or re-confirming results.
   Example: "Let me verify: pi * r^2 = pi * 25 = 25pi. Correct."

8. **final_answer_emission** — Explicitly stating the final answer.
   Example: "Therefore, the answer is 25pi square centimeters."

9. **unknown** — Sentences that are purely stylistic, transitional, or do not fit any above category.

## Instructions

- Each sentence can have ONE OR MORE tags (e.g., a sentence that checks a computation might be both "self_checking" and "active_computation").
- For **depends_on**: list the indices (integers) of earlier sentences whose reasoning the current sentence directly uses.
  - Only mark a sentence as dependent on another if its reasoning **clearly uses** a previous sentence's result or idea.
  - Include both long-range and short-range dependencies.
  - Do NOT forget about long-range dependencies — a sentence may depend on a much earlier sentence.
  - Dependencies should ensure there is a path from earlier sentences to the final answer.
  - If no dependencies exist, use an empty list [].

## Output Format

Return ONLY a JSON object (no markdown fences, no explanation) mapping sentence index (as string) to its annotation:

{"0": {"function_tags": ["problem_setup"], "depends_on": []}, "1": {"function_tags": ["plan_generation"], "depends_on": [0]}, ...}
"""


def build_user_prompt(sentences):
    """Build the user prompt listing all sentences for classification."""
    lines = ["Here is the reasoning trace to annotate:\n"]
    for i, sent in enumerate(sentences):
        text = sent["text"].strip()
        lines.append(f"[{i}] {text}")
    lines.append(
        "\nClassify every sentence above. Return ONLY the JSON object."
    )
    return "\n".join(lines)


def parse_llm_response(response_text, num_sentences):
    """Parse and validate the LLM JSON response."""
    # Try to extract JSON from the response (handle markdown fences)
    text = response_text.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1).strip()

    result = json.loads(text)

    # Validate and normalize
    classifications = {}
    for idx in range(num_sentences):
        key = str(idx)
        if key not in result:
            print(f"  Warning: sentence {idx} missing from LLM response, defaulting to 'unknown'")
            classifications[idx] = {"function_tags": ["unknown"], "depends_on": []}
            continue

        entry = result[key]
        tags = entry.get("function_tags", ["unknown"])
        depends = entry.get("depends_on", [])

        # Validate tags
        validated_tags = [t for t in tags if t in VALID_TAGS]
        if not validated_tags:
            print(f"  Warning: sentence {idx} has no valid tags {tags}, defaulting to 'unknown'")
            validated_tags = ["unknown"]
        if len(validated_tags) != len(tags):
            invalid = set(tags) - VALID_TAGS
            print(f"  Warning: sentence {idx} had invalid tags removed: {invalid}")

        # Validate depends_on (must be ints, in range, and < current index)
        validated_deps = []
        for d in depends:
            d_int = int(d)
            if 0 <= d_int < idx:
                validated_deps.append(d_int)

        classifications[idx] = {
            "function_tags": validated_tags,
            "depends_on": validated_deps,
        }

    return classifications


def classify_sentences(mask_path, model, force=False):
    """Classify sentences in a mask JSON file and update it in-place."""
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("Error: OPENROUTER_API_KEY not found in environment.", file=sys.stderr)
        sys.exit(1)

    # Load raw JSON to preserve all fields
    with open(mask_path, "r") as f:
        data = json.load(f)

    sentences = data.get("sentences", [])
    if not sentences:
        print("No sentences found in mask file.")
        return

    # Check if already classified
    if not force and all("function_tags" in s for s in sentences):
        print("Sentences already classified. Use --force to re-classify.")
        return

    print(f"Classifying {len(sentences)} sentences with {model}...")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

    user_prompt = build_user_prompt(sentences)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    response_text = response.choices[0].message.content
    classifications = parse_llm_response(response_text, len(sentences))

    # Merge classifications into sentence dicts
    for idx, cls in classifications.items():
        sentences[idx]["function_tags"] = cls["function_tags"]
        sentences[idx]["depends_on"] = cls["depends_on"]

    # Write back in-place
    with open(mask_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Done. Updated {mask_path} with classifications.")

    # Print summary
    from collections import Counter
    tag_counts = Counter()
    for s in sentences:
        for tag in s.get("function_tags", []):
            tag_counts[tag] += 1
    print("\nTag distribution:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify mask sentences using Thought Anchors taxonomy."
    )
    parser.add_argument(
        "--mask", required=True,
        help="Path to mask JSON file",
    )
    parser.add_argument(
        "--model", default="google/gemini-2.5-flash",
        help="OpenRouter model for classification (default: google/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-classify even if sentences already have tags",
    )
    args = parser.parse_args()

    if not os.path.exists(args.mask):
        print(f"Error: {args.mask} not found.", file=sys.stderr)
        sys.exit(1)

    classify_sentences(args.mask, args.model, force=args.force)


if __name__ == "__main__":
    main()
