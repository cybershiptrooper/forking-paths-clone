"""Build the candidate answer bank for open-ended (e.g. MATH) prompts.

For each prompt, fixes the probe context — the analysis-sentence prefix,
the k stored continuation sentences, and the probe suffix
`" </think> I think the answer is"` — then:

1. Samples N boxed answers from the clean model at that context
   (temperature 1.0, answer tokens only).
2. Extracts the answer string from each sample (``\\boxed{...}`` first,
   first-line fallback), normalizes lightly, deduplicates keeping counts.
3. Grades each distinct string: exact match against the gold answer,
   then an LLM judge (OpenRouter) for non-matches.  All strings judged
   correct share cluster 0 (the gold cluster); each distinct wrong
   string is its own cluster.
4. Force-includes the gold answer string (count 0) if sampling missed it.
5. Renders every candidate as the canonical continuation
   ``" \\boxed{<answer>}"`` after the probe suffix, tokenizes
   (suffix + continuation) jointly, and verifies the suffix tokens are
   unchanged by the join.
6. Records the clean model's teacher-forced sequence log-probability of
   each candidate continuation.

Output: one JSON per prompt in --output_dir (refuses to overwrite).

Usage:
    uv run python -m expts.direct_answer_circuit_discovery.build_answer_bank \
        --data_path data/collection/qwen3_8b/math_merged_filtered.json \
        --prompt_indices 0 3 5 9 12 \
        --output_dir results/math_reward_gap/answer_banks
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from expts.direct_answer_circuit_discovery.learn import _build_prefix
from expts.direct_answer_circuit_discovery.probe import DEFAULT_SUFFIX
from utils.rewards import extract_boxed, judge_answer
from utils.utils import set_seed, clear_cuda


def normalize_answer(text: str) -> str:
    """Light normalization for dedup/exact-match: strip outer whitespace,
    trailing periods, and collapse internal whitespace runs."""
    s = " ".join(text.split())
    return s.rstrip(".").strip()


def extract_answer_from_generation(text: str) -> str | None:
    """Answer string from a generation that continues ``" \\boxed{"``.

    The sampling context ends with a forced open brace, so *text* is the
    inside of the box (plus whatever follows).  Cut at the matching close
    brace; None if the box never closes (sample dropped as unparsed).
    Sampling with the box forced guarantees clean answer expressions —
    the free-form fallback used in an earlier version contaminated
    answers with trailing reasoning text ("... Wait, let me check").
    """
    depth = 1
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                inner = text[:i]
                return normalize_answer(inner) if inner.strip() else None
    return None


def tokenize_continuation(tokenizer, suffix: str, continuation_text: str):
    """Tokenize suffix + continuation jointly; verify the suffix tokens are
    unchanged by the join (same guarantee build_answer_probe enforces for
    letters).  Returns the full token id list (suffix + answer tokens)."""
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    full_ids = tokenizer.encode(suffix + continuation_text, add_special_tokens=False)
    if full_ids[: len(suffix_ids)] != suffix_ids:
        raise ValueError(
            f"Suffix tokenization changed by continuation {continuation_text!r}: "
            f"suffix={suffix_ids} full_prefix={full_ids[:len(suffix_ids)]}"
        )
    if len(full_ids) == len(suffix_ids):
        raise ValueError(f"Continuation {continuation_text!r} added no tokens")
    return full_ids


@torch.no_grad()
def clean_seq_logprob(model, context_ids, continuation_ids, device):
    """Teacher-forced sum of log-probs of continuation_ids after context_ids."""
    full = torch.cat(
        [context_ids, torch.tensor([continuation_ids], device=device)], dim=-1
    )
    ctx_len = context_ids.shape[-1]
    logits = model(full).logits.float()
    # Position i predicts token i+1: rows ctx_len-1 .. end-2 predict the continuation.
    rows = logits[0, ctx_len - 1 : -1, :]
    logprobs = F.log_softmax(rows, dim=-1)
    targets = full[0, ctx_len:]
    return float(logprobs[torch.arange(len(targets)), targets].sum().item())


def build_bank_for_prompt(
    *,
    model,
    tokenizer,
    data_path,
    prompt_index,
    analysis_sentence_step,
    sentences_after_prefix,
    n_samples,
    temperature,
    max_new_tokens,
    judge_client,
    judge_model,
    confirm_judge_model,
    seed,
    device,
):
    set_seed(seed)
    prefix_ids, _sents, prompt, gold_answer, _fmt, _npr = _build_prefix(
        tokenizer=tokenizer,
        prompt=None,
        data_path=data_path,
        prompt_index=prompt_index,
        base_answer_type="stored",
        analysis_timestep=None,
        analysis_sentence_step=analysis_sentence_step,
        sentences_after_prefix=sentences_after_prefix,
        min_sentence_length=10,
        sentence_chunk=1,
    )
    if gold_answer is None:
        raise ValueError(f"No gold answer for prompt_index={prompt_index}")
    gold_norm = normalize_answer(gold_answer)

    suffix_ids = tokenizer.encode(DEFAULT_SUFFIX, add_special_tokens=False)
    # Force the boxed format: sampling continues from " \boxed{" so every
    # sample is the inside of the box (clean answer expression).
    boxed_open_ids = tokenizer.encode(
        DEFAULT_SUFFIX + " \\boxed{", add_special_tokens=False
    )
    if boxed_open_ids[: len(suffix_ids)] != suffix_ids:
        raise ValueError("suffix tokenization changed by ' \\boxed{'")
    context = torch.cat(
        [prefix_ids, torch.tensor([boxed_open_ids])], dim=-1
    ).to(device)
    print(f"  Context: {context.shape[-1]} tokens "
          f"(prefix {prefix_ids.shape[-1]} + suffix {len(suffix_ids)} "
          f"+ forced ' \\boxed{{' {len(boxed_open_ids) - len(suffix_ids)})")

    # ----- Step 1: sample answers from the clean model -----
    out = model.generate(
        input_ids=context.expand(n_samples, -1),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen_texts = [
        tokenizer.decode(out[i, context.shape[-1]:], skip_special_tokens=True)
        for i in range(n_samples)
    ]

    # ----- Step 2: extract + dedupe -----
    counts: dict[str, int] = {}
    unparsed = 0
    for t in gen_texts:
        ans = extract_answer_from_generation(t)
        if ans is None or ans == "":
            unparsed += 1
            continue
        counts[ans] = counts.get(ans, 0) + 1

    # ----- Step 3: grade (double-judged) -----
    # A single 8B judge misgrades occasionally (observed: a non-solution
    # vector judged equivalent to the gold vector).  Rule: gold cluster
    # requires YES from both judges; distractor requires NO from both;
    # disagreement drops the string as ambiguous.
    def grade(ans):
        if ans == gold_norm:
            return ("exact", True)
        v1 = judge_answer(
            question=prompt, ground_truth=gold_answer, model_answer=ans,
            client=judge_client, model=judge_model,
        )
        v2 = judge_answer(
            question=prompt, ground_truth=gold_answer, model_answer=ans,
            client=judge_client, model=confirm_judge_model,
        )
        if v1 == v2:
            return ("double_judge", v1)
        return ("ambiguous", None)

    graded = {}
    ambiguous = []
    for ans in list(counts):
        method, verdict = grade(ans)
        if verdict is None:
            ambiguous.append(ans)
            del counts[ans]
            continue
        graded[ans] = (method, verdict)

    # ----- Step 4: force-include gold -----
    if gold_norm not in counts:
        counts[gold_norm] = 0
        graded[gold_norm] = ("forced_gold", True)

    # ----- Step 4b: augment with wrong answers from the collection rollouts -----
    # The clean model at the probe context sometimes produces no wrong
    # answers at all (or none matching the full-rollout failure modes);
    # the reward gap needs wrong clusters to compete against.  Source:
    # boxed answers from the prompt's original 16 full rollouts, kept only
    # if both judges grade them wrong.
    sources = {ans: "probe_sample" for ans in counts}
    if counts.get(gold_norm, 1) == 0:
        sources[gold_norm] = "forced_gold"
    with open(data_path) as f:
        record = json.load(f)[prompt_index]
    collection_answers = set()
    for alt in record.get("all_sampled_answers") or []:
        boxed = extract_boxed(alt)
        if boxed is None:
            continue
        s = normalize_answer(boxed)
        if not s or len(s) > 80:
            continue
        ascii_frac = sum(1 for ch in s if ord(ch) < 128) / len(s)
        has_content = any(ch.isalnum() or ch == "\\" for ch in s)
        if ascii_frac < 0.8 or not has_content:
            continue
        collection_answers.add(s)
    n_collection_added = 0
    for s in collection_answers - set(counts):
        method, verdict = grade(s)
        if verdict is False:
            counts[s] = 0
            graded[s] = (method, False)
            sources[s] = "collection_rollout"
            n_collection_added += 1
        elif verdict is None:
            ambiguous.append(s)

    # ----- Step 5+6: canonical continuations, clusters, clean logprobs -----
    # Cluster 0 = every string graded correct; each wrong string is its own
    # cluster (equivalent-but-differently-written wrong answers stay
    # separate; noted as a limitation in the report).
    candidates = []
    next_wrong_cluster = 1
    for ans, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        method, is_correct = graded[ans]
        if is_correct:
            cluster = 0
        else:
            cluster = next_wrong_cluster
            next_wrong_cluster += 1
        continuation_text = " \\boxed{" + ans + "}"
        token_ids = tokenize_continuation(tokenizer, DEFAULT_SUFFIX, continuation_text)
        # Second tokenization path: forced-brace split (the BPE merges
        # '{' with a following '-', '\\', '(' etc. in the canonical join;
        # generation, having already emitted '{', cannot take that path).
        forced_ids = boxed_open_ids + tokenizer.encode(
            ans + "}", add_special_tokens=False
        )
        variants = [token_ids] + ([forced_ids] if forced_ids != token_ids else [])
        lp = clean_seq_logprob(model, prefix_ids.to(device), token_ids, device)
        candidates.append({
            "answer_text": ans,
            "continuation_text": continuation_text,
            "continuation_token_ids": token_ids,  # suffix + answer tokens
            "continuation_token_ids_variants": variants,
            "count": count,
            "cluster_id": cluster,
            "is_correct": bool(is_correct),
            "grade_method": method,
            "source": sources.get(ans, "probe_sample"),
            "clean_seq_logprob": lp,
        })

    n_sampled = sum(c["count"] for c in candidates)
    n_correct_samples = sum(c["count"] for c in candidates if c["is_correct"])
    clean_fraction_correct = n_correct_samples / max(n_sampled, 1)
    no_wrong_candidates = next_wrong_cluster == 1
    if no_wrong_candidates:
        print(f"  WARNING: no wrong candidates for prompt_index={prompt_index} "
              f"— the reward gap over this bank is degenerate.")

    return {
        "data_path": data_path,
        "prompt_index": prompt_index,
        "analysis_sentence_step": analysis_sentence_step,
        "sentences_after_prefix": sentences_after_prefix,
        "probe_suffix": DEFAULT_SUFFIX,
        "probe_suffix_token_ids": suffix_ids,
        "question": prompt,
        "gold_answer": gold_answer,
        "gold_answer_normalized": gold_norm,
        "sampling": {
            "n_samples": n_samples,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "n_unparsed": unparsed,
        },
        "judge_model": judge_model,
        "confirm_judge_model": confirm_judge_model,
        "ambiguous_answers": ambiguous,
        "n_collection_distractors": n_collection_added,
        "no_wrong_candidates": no_wrong_candidates,
        "candidates": candidates,
        "num_clusters": next_wrong_cluster,
        "target_cluster": 0,
        "clean_fraction_correct": clean_fraction_correct,
        "sample_texts": gen_texts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prompt_indices", type=int, nargs="+", required=True)
    parser.add_argument("--analysis_sentence_step", type=int, default=50)
    parser.add_argument("--sentences_after_prefix", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--judge_model", default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument(
        "--confirm_judge_model", default="meta-llama/llama-3.3-70b-instruct",
        help="Second judge; gold cluster requires YES from both, "
        "distractors NO from both (disagreement drops the string).",
    )
    parser.add_argument("--output_dir", default="results/math_reward_gap/answer_banks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (checked .env)")
    judge_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()

    for pidx in args.prompt_indices:
        out_path = os.path.join(args.output_dir, f"p{pidx:02d}.json")
        if os.path.exists(out_path):
            print(f"[p{pidx:02d}] {out_path} exists; skipping (no overwrite).")
            continue
        print(f"[p{pidx:02d}] building bank...")
        bank = build_bank_for_prompt(
            model=model,
            tokenizer=tokenizer,
            data_path=args.data_path,
            prompt_index=pidx,
            analysis_sentence_step=args.analysis_sentence_step,
            sentences_after_prefix=args.sentences_after_prefix,
            n_samples=args.n_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            judge_client=judge_client,
            judge_model=args.judge_model,
            confirm_judge_model=args.confirm_judge_model,
            seed=args.seed,
            device=args.device,
        )
        with open(out_path, "w") as f:
            json.dump(bank, f, indent=2)
        n_c = len(bank["candidates"])
        print(f"[p{pidx:02d}] {n_c} candidates, "
              f"{bank['num_clusters']} clusters, "
              f"clean fraction correct = {bank['clean_fraction_correct']:.3f}, "
              f"gold={bank['gold_answer_normalized']!r} -> {out_path}")
        clear_cuda()


if __name__ == "__main__":
    main()
