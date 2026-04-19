"""Analyse collected forking-paths data.

Judges model answers against ground truth, filters for ambiguous samples
(25-75% accuracy), performs answer-closeness analysis, and generates
per-sample plots.

Usage:
    uv run python -m expts.analyse_collected_data \
        --data data/collection/deepseek_llama_8b/math_open.json \
        --judge-model meta-llama/llama-3.1-8b-instruct
"""

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# OpenRouter cache
# ---------------------------------------------------------------------------

CACHE_DIR = Path("cache/openrouter")


def _cache_key(model: str, messages: list[dict], max_tokens: int) -> str:
    """Deterministic hash for a (model, messages, max_tokens) tuple."""
    blob = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def cached_chat(client: OpenAI, model: str, messages: list[dict], max_tokens: int, temperature: float = 0) -> str:
    """Call OpenRouter with a file-based cache. Returns the response content string."""
    key = _cache_key(model, messages, max_tokens)
    cache_path = CACHE_DIR / model.replace("/", "__") / f"{key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)["content"]

    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    content = resp.choices[0].message.content.strip()
    with open(cache_path, "w") as f:
        json.dump({"content": content}, f)
    return content


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> str | None:
    """Extract the content of the last \\boxed{...}, handling nested braces."""
    result = None
    for m in re.finditer(r"\\boxed\{", text):
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            result = text[start : i - 1]
    return result


def extract_after_think(text: str) -> str:
    """Return everything after </think>, or the full text if no tag found."""
    marker = "</think>"
    idx = text.find(marker)
    if idx >= 0:
        return text[idx + len(marker) :].strip()
    return text.strip()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
    return dp[m]


def try_parse_number(s: str) -> float | None:
    """Try to parse a string (possibly LaTeX) as a float."""
    if s is None:
        return None
    s = s.strip()
    # Direct float
    try:
        return float(s)
    except ValueError:
        pass
    # LaTeX \frac{a}{b} or \dfrac{a}{b}
    m = re.match(r"^-?\\?d?frac\{([^}]+)\}\{([^}]+)\}$", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2)) * (-1 if s.startswith("-") else 1)
        except (ValueError, ZeroDivisionError):
            pass
    # Simple fraction a/b
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except ZeroDivisionError:
            pass
    # Trailing period (e.g. "8.")
    try:
        return float(s.rstrip("."))
    except ValueError:
        pass
    # sqrt
    m = re.match(r"^\\?sqrt\{?(\d+)\}?$", s)
    if m:
        return float(m.group(1)) ** 0.5
    return None


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are a math answer checker. Given a math problem, the correct answer, and a model's answer, \
determine if the model's answer is mathematically equivalent to the correct answer. \
Equivalent means the same value even if written differently (e.g. 42/5 = 8.4, \\frac{{1}}{{2}} = 0.5).

Problem:
{question}

Correct answer: {ground_truth}

Model's answer: {model_answer}

Is the model's answer correct? Reply with only YES or NO."""


def judge_answer(client: OpenAI, model: str, question: str, ground_truth: str, model_answer: str) -> bool:
    messages = [{"role": "user", "content": JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, model_answer=model_answer
    )}]
    content = cached_chat(client, model, messages, max_tokens=5)
    return content.upper().startswith("YES")


# ---------------------------------------------------------------------------
# Error categorisation
# ---------------------------------------------------------------------------

ERROR_CAT_PROMPT = """\
You are analysing a math model's wrong answer. Given the problem, the correct answer, and the \
model's full solution (after its thinking phase), classify the error into exactly ONE category.

Categories:
- silly_mistake: A small arithmetic or algebraic slip near the end; the approach was correct but \
a minor calculation went wrong (e.g. 743 vs 744, sign error in last step).
- token_error: The answer itself is garbled, truncated, or a formatting issue (e.g. "\\frac" \
without numbers, cut-off output). The reasoning may have been fine but the answer wasn't \
extracted properly.
- logic_error: A fundamental mistake in reasoning or approach (wrong formula, wrong setup, \
misunderstood the problem).
- incomplete: The model did not finish solving or gave up partway through.

Problem:
{question}

Correct answer: {ground_truth}

Model's full solution:
{model_solution}

Model's extracted answer: {model_answer}

Reply with ONLY the category name (one of: silly_mistake, token_error, logic_error, incomplete)."""


def categorise_error(
    client: OpenAI, model: str, question: str, ground_truth: str, model_solution: str, model_answer: str
) -> str:
    messages = [{"role": "user", "content": ERROR_CAT_PROMPT.format(
        question=question, ground_truth=ground_truth,
        model_solution=model_solution[:3000],  # truncate very long solutions
        model_answer=model_answer,
    )}]
    content = cached_chat(client, model, messages, max_tokens=10)
    cat = content.strip().lower().replace(" ", "_")
    valid = {"silly_mistake", "token_error", "logic_error", "incomplete"}
    return cat if cat in valid else "logic_error"


# ---------------------------------------------------------------------------
# Per-record judging
# ---------------------------------------------------------------------------


def judge_record(client: OpenAI, model: str, record: dict) -> dict:
    """Judge all sampled answers for one record."""
    question = record["question"]
    gt_boxed = extract_boxed(record["correct_answer"]) or record["correct_answer"]
    all_answers = record["all_sampled_answers"]

    # Deduplicate to save API calls
    unique_answers = list(set(all_answers))
    verdicts = {}
    for ans in unique_answers:
        verdicts[ans] = judge_answer(client, model, question, gt_boxed, ans)

    num_correct = sum(verdicts[a] for a in all_answers)
    return {
        "prompt_id": record["prompt_id"],
        "question": record["question"][:120].replace("\n", " "),
        "ground_truth": gt_boxed,
        "base_answer": record["clean_answer"],
        "base_correct": verdicts.get(record["clean_answer"], False),
        "num_correct": num_correct,
        "num_paths": len(all_answers),
        "accuracy": num_correct / len(all_answers),
        "base_answer_rate": record["base_answer_rate"],
        "num_cut_short": record["num_random_cut_short"],
        "all_answers": all_answers,
        "verdicts": [verdicts[a] for a in all_answers],
    }


# ---------------------------------------------------------------------------
# Answer closeness analysis
# ---------------------------------------------------------------------------


def analyse_closeness(client: OpenAI, cat_model: str, record: dict, result: dict) -> dict:
    """Compute numeric distance and LLM error categories for wrong answers."""
    gt = result["ground_truth"]
    gt_num = try_parse_number(gt)
    all_answers = result["all_answers"]
    verdicts = result["verdicts"]

    # Collect full solutions for wrong answers
    # all_sampled_answers[0] = base (clean_answer), rest = alternate_answers
    all_solutions = [extract_after_think(record["output_text"])]
    all_solutions.extend(extract_after_think(t) for t in record["alternate_texts"])

    wrong_indices = [i for i, v in enumerate(verdicts) if not v]
    unique_wrong = {}  # answer -> first index
    for i in wrong_indices:
        ans = all_answers[i]
        if ans not in unique_wrong:
            unique_wrong[ans] = i

    # Numeric distances and edit distances
    distances = {}
    edit_dists = {}
    for ans, idx in unique_wrong.items():
        ans_num = try_parse_number(ans)
        if gt_num is not None and ans_num is not None:
            distances[ans] = abs(ans_num - gt_num)
        else:
            distances[ans] = None
        edit_dists[ans] = edit_distance(ans, gt)

    # Error categories via LLM
    categories = {}
    for ans, idx in unique_wrong.items():
        categories[ans] = categorise_error(
            client, cat_model,
            record["question"], gt,
            all_solutions[idx], ans,
        )

    return {
        "prompt_id": result["prompt_id"],
        "ground_truth": gt,
        "gt_numeric": gt_num,
        "wrong_answers": {
            ans: {
                "count": sum(1 for i in wrong_indices if all_answers[i] == ans),
                "numeric_distance": distances.get(ans),
                "edit_distance": edit_dists.get(ans),
                "category": categories.get(ans, "unknown"),
            }
            for ans in unique_wrong
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sample(result: dict, closeness: dict, output_dir: Path):
    """Generate plots for one filtered sample."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_answers = result["all_answers"]
    verdicts = result["verdicts"]
    gt = result["ground_truth"]

    # --- 1. Answer distribution histogram ---
    answer_counts: dict[str, int] = {}
    for ans in all_answers:
        # Truncate long answers for display
        label = ans[:30] + ("…" if len(ans) > 30 else "")
        answer_counts[label] = answer_counts.get(label, 0) + 1

    fig, ax = plt.subplots(figsize=(max(6, len(answer_counts) * 1.2), 4))
    labels = list(answer_counts.keys())
    counts = list(answer_counts.values())
    colors = []
    # Color correct answers green, wrong red
    verdict_by_label: dict[str, bool] = {}
    for ans, v in zip(all_answers, verdicts):
        label = ans[:30] + ("…" if len(ans) > 30 else "")
        verdict_by_label[label] = v
    colors = ["#4CAF50" if verdict_by_label.get(l, False) else "#F44336" for l in labels]

    ax.bar(range(len(labels)), counts, color=colors, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(f"Answer distribution (prompt_id={result['prompt_id']})\nGT: {gt[:50]}")
    ax.set_xlabel("Answer")
    plt.tight_layout()
    fig.savefig(output_dir / "answer_distribution.png", dpi=150)
    plt.close(fig)

    # --- 2. Numeric distance from correct answer ---
    wrong_info = closeness["wrong_answers"]
    parseable = {ans: info for ans, info in wrong_info.items() if info["numeric_distance"] is not None}
    if parseable:
        fig, ax = plt.subplots(figsize=(max(6, len(parseable) * 1.2), 4))
        labels_d = [a[:30] for a in parseable]
        dists = [parseable[a]["numeric_distance"] for a in parseable]
        ax.bar(range(len(labels_d)), dists, color="#FF9800", edgecolor="black")
        ax.set_xticks(range(len(labels_d)))
        ax.set_xticklabels(labels_d, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Absolute distance from GT")
        ax.set_title(f"Numeric distance from correct answer (prompt_id={result['prompt_id']})")
        ax.set_xlabel("Wrong answer")
        plt.tight_layout()
        fig.savefig(output_dir / "numeric_distance.png", dpi=150)
        plt.close(fig)

    # --- 3. Edit distance from correct answer ---
    if wrong_info:
        fig, ax = plt.subplots(figsize=(max(6, len(wrong_info) * 1.2), 4))
        labels_e = [a[:30] for a in wrong_info]
        edists = [wrong_info[a]["edit_distance"] for a in wrong_info]
        ax.bar(range(len(labels_e)), edists, color="#2196F3", edgecolor="black")
        ax.set_xticks(range(len(labels_e)))
        ax.set_xticklabels(labels_e, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Edit distance (chars)")
        ax.set_title(f"Edit distance from correct answer (prompt_id={result['prompt_id']})")
        ax.set_xlabel("Wrong answer")
        plt.tight_layout()
        fig.savefig(output_dir / "edit_distance.png", dpi=150)
        plt.close(fig)

    # --- 4. Error categories ---
    if wrong_info:
        cat_counts: dict[str, int] = {}
        for info in wrong_info.values():
            cat = info["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + info["count"]

        cat_colors = {
            "silly_mistake": "#FFC107",
            "token_error": "#9C27B0",
            "logic_error": "#F44336",
            "incomplete": "#607D8B",
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        cats = list(cat_counts.keys())
        cat_vals = [cat_counts[c] for c in cats]
        ax.bar(cats, cat_vals, color=[cat_colors.get(c, "#999") for c in cats], edgecolor="black")
        ax.set_ylabel("Count (branches)")
        ax.set_title(f"Error categories (prompt_id={result['prompt_id']})")
        ax.set_xlabel("Category")
        plt.tight_layout()
        fig.savefig(output_dir / "error_categories.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyse collected forking-paths data")
    parser.add_argument("--data", type=str, default="data/collection/deepseek_llama_8b/math_open.json")
    parser.add_argument("--judge-model", type=str, default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--cat-model", type=str, default="meta-llama/llama-3.1-8b-instruct",
                        help="Model for error categorisation")
    parser.add_argument("--output-dir", type=str, default="results/data_collection_analysis")
    parser.add_argument("--filtered-output", type=str, default=None,
                        help="Path for filtered JSON (default: sibling of --data)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set in environment / .env", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # --- Load data ---
    with open(args.data) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {args.data}")

    # Build prompt_id -> original JSON index mapping
    pid_to_idx = {r["prompt_id"]: i for i, r in enumerate(data)}

    # --- Stage 1: Judge all answers ---
    print(f"\n=== Judging with {args.judge_model} ===")
    judge_results = [None] * len(data)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_record, client, args.judge_model, rec): i for i, rec in enumerate(data)}
        for future in tqdm(as_completed(futures), total=len(data), desc="Judging"):
            i = futures[future]
            judge_results[i] = future.result()

    # --- Summary table ---
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("all_answers", "verdicts")}
        for r in judge_results
    ]).sort_values("prompt_id").reset_index(drop=True)

    print(f"\nBase answer accuracy  (greedy):  {df['base_correct'].mean():.2%}")
    print(f"Mean accuracy over {df['num_paths'].iloc[0]} paths:  {df['accuracy'].mean():.2%}")

    # --- Stage 2: Filter ambiguous samples (25-75% accuracy) ---
    filtered_mask = (df["accuracy"] > 0.25) & (df["accuracy"] < 0.75)
    filtered_df = df[filtered_mask]
    filtered_pids = set(filtered_df["prompt_id"].tolist())
    print(f"\nFiltered to {len(filtered_df)} samples with 25-75% accuracy:")
    print(filtered_df[["prompt_id", "ground_truth", "base_answer", "accuracy"]].to_string(index=False))

    # Save filtered records (complete original records), preserving original order
    filtered_records = [r for r in data if r["prompt_id"] in filtered_pids]
    # Build filtered_index -> prompt_id mapping (order matches the filtered JSON)
    filtered_index_to_pid = {i: r["prompt_id"] for i, r in enumerate(filtered_records)}
    pid_to_filtered_index = {pid: i for i, pid in filtered_index_to_pid.items()}

    if args.filtered_output is None:
        filtered_path = Path(args.data).parent / "math_filtered.json"
    else:
        filtered_path = Path(args.filtered_output)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filtered_path, "w") as f:
        json.dump(filtered_records, f, indent=2)
    print(f"Saved {len(filtered_records)} filtered records to {filtered_path}")
    for fi, pid in filtered_index_to_pid.items():
        print(f"  filtered_index={fi}  prompt_id={pid}  original_index={pid_to_idx[pid]}")

    # --- Stage 3: Answer closeness analysis on filtered samples ---
    print(f"\n=== Error analysis with {args.cat_model} ===")
    filtered_results = {r["prompt_id"]: r for r in judge_results if r["prompt_id"] in filtered_pids}
    closeness_results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for pid in filtered_pids:
            rec = data[pid_to_idx[pid]]
            res = filtered_results[pid]
            futures[pool.submit(analyse_closeness, client, args.cat_model, rec, res)] = pid
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analysing"):
            pid = futures[future]
            closeness_results[pid] = future.result()

    # --- Stage 4: Per-sample plots + metadata ---
    print("\n=== Generating plots ===")
    output_base = Path(args.output_dir)
    for pid in sorted(filtered_pids):
        fi = pid_to_filtered_index[pid]
        sample_dir = output_base / str(fi)
        plot_sample(filtered_results[pid], closeness_results[pid], sample_dir)

        # Write per-sample metadata
        rec = data[pid_to_idx[pid]]
        # Extract complete final answers from full output text for each path
        all_full_texts = [rec["output_text"]] + rec["alternate_texts"]
        finish_reasons = [rec["finish_reason"]] + rec["alternate_finish_reasons"]
        complete_final_answers = []
        for text, fr in zip(all_full_texts, finish_reasons):
            boxed = extract_boxed(text)
            if boxed is not None:
                complete_final_answers.append(boxed)
            elif "</think>" in text:
                # Model finished thinking but didn't use \boxed{}
                complete_final_answers.append(extract_after_think(text)[:500])
            else:
                # Output was truncated before finishing thinking
                complete_final_answers.append(None)

        metadata = {
            "filtered_index": fi,
            "prompt_id": pid,
            "original_index": pid_to_idx[pid],
            "filtered_data_path": str(filtered_path),
            "original_data_path": args.data,
            "question": rec["question"],
            "ground_truth": closeness_results[pid]["ground_truth"],
            "accuracy": filtered_results[pid]["accuracy"],
            "num_paths": filtered_results[pid]["num_paths"],
            "base_answer": filtered_results[pid]["base_answer"],
            "base_correct": filtered_results[pid]["base_correct"],
            "dataset_name": rec.get("dataset_name"),
            "parsed_answers": filtered_results[pid]["all_answers"],
            "verdicts": filtered_results[pid]["verdicts"],
            "complete_final_answers": complete_final_answers,
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  filtered_index={fi}  prompt_id={pid} -> {sample_dir}/")

    # --- Stage 5: Print closeness report ---
    print("\n=== Answer Closeness Report ===")
    for pid in sorted(filtered_pids):
        fi = pid_to_filtered_index[pid]
        c = closeness_results[pid]
        print(f"\nfiltered_index={fi}  prompt_id={pid}  GT={c['ground_truth']}")
        for ans, info in c["wrong_answers"].items():
            dist_str = f"{info['numeric_distance']:.4g}" if info["numeric_distance"] is not None else "N/A"
            edit_str = str(info.get("edit_distance", "N/A"))
            print(f"  '{ans}' x{info['count']}  dist={dist_str}  edit_dist={edit_str}  category={info['category']}")

    # Save full report as JSON
    report_path = output_base / "report.json"
    output_base.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": {
            "total_records": len(data),
            "base_accuracy": float(df["base_correct"].mean()),
            "mean_path_accuracy": float(df["accuracy"].mean()),
            "num_filtered": len(filtered_pids),
        },
        "filtered_samples": {
            str(pid): closeness_results[pid] for pid in sorted(filtered_pids)
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
