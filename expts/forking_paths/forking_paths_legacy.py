import argparse
import bisect
import json
import os
import random
from typing import List, Optional
from vllm import LLM, SamplingParams

from utils.answer_utils import parse_answer
from utils.utils import MODEL_METADATA, clear_cuda, set_seed


_SENTENCE_ENDERS = frozenset('.!?')
_SUB_SENTENCE_DELIMITERS = frozenset(',;:')
_REASONING_MARKERS = ['</think>', '<think>']


def _find_sentence_boundaries(text, include_sub_sentences=False):
    """Find character positions of sentence (and optionally sub-sentence) delimiters.

    Returns a sorted list of unique character positions where a sentence or
    clause ends.  Each position is the index of the delimiter character
    itself (e.g. the '.' in "Hello. World" is at the returned position).

    Handles:
    - Sentence-ending punctuation (. ! ?) followed by whitespace or end of text
    - Decimal numbers: "0.8" is NOT treated as a sentence boundary
    - Ellipsis: "..." creates a single boundary (not three)
    - Newlines (single \\n and paragraph breaks \\n\\n)
    - Reasoning markers (<think>, </think>)
    - Optional sub-sentence delimiters (, ; :) followed by whitespace
    - Digit-separated commas (1,000) are NOT sub-sentence boundaries
    """
    boundaries = []
    i = 0
    n = len(text)

    while i < n:
        # ── Reasoning markers (<think>, </think>) ────────────────────
        matched_marker = False
        for marker in _REASONING_MARKERS:
            if text[i:i + len(marker)] == marker:
                boundaries.append(i + len(marker) - 1)
                i += len(marker)
                matched_marker = True
                break
        if matched_marker:
            continue

        char = text[i]

        # ── Newlines ─────────────────────────────────────────────────
        if char == '\n':
            # Consume consecutive newlines; emit one boundary at the last
            end = i + 1
            while end < n and text[end] == '\n':
                end += 1
            boundaries.append(end - 1)
            i = end
            continue

        # ── Sentence-ending punctuation (. ! ?) ──────────────────────
        if char in _SENTENCE_ENDERS:
            if char == '.':
                # Consume consecutive dots (ellipsis)
                end = i + 1
                while end < n and text[end] == '.':
                    end += 1

                # Decimal-number check (single dot between digits)
                if end == i + 1:
                    prev_is_digit = i > 0 and text[i - 1].isdigit()
                    next_is_digit = end < n and text[end].isdigit()
                    if prev_is_digit and next_is_digit:
                        i = end
                        continue

                # Boundary only if followed by whitespace or end of text
                if end >= n or text[end] in ' \t\r\n':
                    boundaries.append(end - 1)
                i = end
                continue
            else:
                # '!' or '?'
                if i + 1 >= n or text[i + 1] in ' \t\r\n':
                    boundaries.append(i)
                i += 1
                continue

        # ── Sub-sentence delimiters (, ; :) ───────────────────────────
        if include_sub_sentences and char in _SUB_SENTENCE_DELIMITERS:
            # Skip commas inside numbers (e.g. 1,000)
            if char == ',':
                if i > 0 and text[i - 1].isdigit() and i + 1 < n and text[i + 1].isdigit():
                    i += 1
                    continue
            if i + 1 >= n or text[i + 1] in ' \t\r\n':
                boundaries.append(i)

        i += 1

    return sorted(set(boundaries))


def collect_stumps(
    llm : LLM,
    prompt_token_ids : List[int],
    output_token_ids : List[int],
    include_sub_sentences : bool = False
):
    """
    Collect all starts of branches (i.e., stumps) at which we're forking.

    Splits the base reasoning path at sentence boundaries, ensuring every
    split aligns exactly with a token boundary.  Sentence boundaries are
    detected by looking for punctuation (. ! ?) followed by whitespace,
    newlines, and reasoning markers (<think>, </think>).

    llm : vllm.LLM
        vLLM model (used for its tokenizer).
    prompt_token_ids : list[int]
        Tokenized prompt (shared across all paths).
    output_token_ids : list[int]
        Tokenized base path after prompt.
    include_sub_sentences : bool
        If True, also split on comma, colon, and semicolon delimiters.

    Returns
    list[dict]
        Collection of stumps (places in base path at which we want to branch off).
        stump = {stump_token_ids, prompt_and_stump_token_ids, t}
    """
    tokenizer = llm.get_tokenizer()

    # Decode each token individually and build the full text with a
    # cumulative character-end mapping so we can snap text-level boundaries
    # back to exact token positions.
    token_texts = [tokenizer.decode([tid]) for tid in output_token_ids]
    full_text = ''.join(token_texts)

    token_char_ends = []
    pos = 0
    for text in token_texts:
        pos += len(text)
        token_char_ends.append(pos)

    # Detect sentence (and optionally sub-sentence) boundaries in the text
    boundary_chars = _find_sentence_boundaries(full_text, include_sub_sentences)

    # Map each character-level boundary to the token that contains it
    boundary_token_indices = set()
    for char_pos in boundary_chars:
        token_idx = bisect.bisect_right(token_char_ends, char_pos)
        if 0 <= token_idx < len(output_token_ids):
            boundary_token_indices.add(token_idx)

    # Assemble stumps in token order
    stumps = []
    for token_idx in sorted(boundary_token_indices):
        stump_token_ids = output_token_ids[:token_idx + 1]
        stumps.append({
            "stump_token_ids": stump_token_ids,
            "prompt_and_stump_token_ids": prompt_token_ids + stump_token_ids,
            "t": token_idx + 1
        })

    return stumps

def generate_branches(
    llm : LLM,
    stumps : List,
    num_branches : int,
    max_new_tokens : int,
    temperature : float
):
    """
    Starting at each stump in our collection, sample continuations (i.e., branches).

    llm : vllm.LLM
        vLLM model for generation.
    stumps : list[dict]
        List of stumps, each of which is a list of token ids.
        stump = {token_ids: ..., t: ..., token_id : ..., token_prob: ...} 
    num_branches : int
        Number of new branches to generate for each stump.
    max_new_tokens : int
        Maximum length of each branch (not counting the stump).

    Returns
    list[dict]
        Records for each branch generated. Includes information about where we branched, the generated text,
        the stopping reason, and the probability that the branch was sampled.
    """
    # generate branches!
    sampling_params = SamplingParams(
        n=num_branches, # output num_branches paths for each stump
        temperature=temperature, # random sampling
        logprobs=0, # return logprobs for sampled branch
        max_tokens=max_new_tokens
    )
    # put in vLLM input format (generate from full path, so include prompt ids)
    llm_inputs = [{'prompt_token_ids': stump["prompt_and_stump_token_ids"]} for stump in stumps]
    branch_outputs = llm.generate(llm_inputs, sampling_params)

    # post-process into ans_df
    branch_results = []
    for i in range(len(branch_outputs)):
        stump = stumps[i]
        for branch_output in branch_outputs[i].outputs: # go through generated outputs
            output_token_ids = stump['stump_token_ids'] + branch_output.token_ids # put together stump + branch (everything after prompt)
            output_text = llm.get_tokenizer().decode(output_token_ids, skip_special_tokens=True)
            branch_results.append({
                # stump data
                't': stump['t'], # fork token index
                # branch data
                'output_text': output_text, # full text (after prompt)
                'post_stump_output_text': branch_output.text, # branched text (after stump)
                'finish_reason': branch_output.finish_reason,
                'output_length': len(branch_output.token_ids),
                'cumulative_logprob': branch_output.cumulative_logprob,
                'norm_cumulative_logprob': branch_output.cumulative_logprob * (1 / max(1, len(branch_output.token_ids))),
                # answer data added later!
            })

    return branch_results

def main(
    model_name : str = "gpt2",
    dataset_name : str = "AQuA",
    # forking paths parameters
    num_branches : int = 30,
    max_new_tokens : int = 10000,
    temperature : float = 0.7,
    # control parameters
    seed : int = 42,
    # script paramaters
    start_index : Optional[int] = None,
    end_index : Optional[int] = None,
    only_parse_answers : bool = False,
    recompute_forking_paths : bool = False,
    include_sub_sentences : bool = False,
    # base-path generation (used when collection data doesn't exist)
    num_examples : int = 100,
    shuffle : bool = True,
    # streamlit compression
    compress_to_streamlit : bool = False
):
    set_seed(seed)

    with open('config.json') as f:
        config = json.load(f)
        dataset_metadata_filename = config["save_locations"]["dataset_metadata_file"]
        data_dir = config["save_locations"]["collection_folder"] # input
        forking_paths_dir = config["save_locations"]["forking_paths_folder"] # output
        answer_model_name = config["experiment_parameters"]["answer_model"]

    # load input
    model_nickname = MODEL_METADATA[model_name]['nickname']
    collection_path = f'{data_dir}/{model_nickname}/{dataset_name.lower()}.json'

    # ── Ensure base paths exist ──────────────────────────────────────
    base_llm = None
    base_paths_to_parse = None

    if os.path.exists(collection_path):
        with open(collection_path) as f:
            dataset = json.load(f)
    else:
        print(f"Collection data not found at {collection_path}")
        print(f"Generating base paths for {dataset_name} ({num_examples} examples, shuffle={shuffle})...")

        from data_collection import load_data, generate_base_paths

        with open(dataset_metadata_filename) as f:
            datasets_metadata = json.load(f)

        base_llm = LLM(model=model_name, dtype="bfloat16")
        raw_dataset = load_data(
            base_llm.get_tokenizer(),
            datasets_metadata[dataset_name],
            n=num_examples,
            shuffle=shuffle
        )
        dataset = generate_base_paths(
            base_llm, 
            raw_dataset,
            max_new_tokens=max_new_tokens,
            return_logprobs=True
        )
        # Normalize token-id types to lists (vLLM may return tuples which
        # don't survive list+tuple concatenation; JSON round-trips fix this
        # in the normal pipeline, but we skip that here).
        for entry in dataset:
            entry['prompt_token_ids'] = list(entry['prompt_token_ids'])
            entry['output_token_ids'] = list(entry['output_token_ids'])

        base_paths_to_parse = dataset

    start_index = 0 if start_index is None else start_index
    end_index = len(dataset) if end_index is None else min(end_index, len(dataset))

    # create output dir
    output_dir = f'{forking_paths_dir}/{model_nickname}/{dataset_name.lower()}'
    os.makedirs(output_dir, exist_ok=True)

    # ── Branch generation ────────────────────────────────────────────
    if not only_parse_answers:
        if base_llm is None:
            base_llm = LLM(model=model_name, dtype="bfloat16")

        for prompt_index in range(start_index, end_index):
            # skip if already processed
            result_path = os.path.join(output_dir, f'{prompt_index:04d}.json')
            if os.path.exists(result_path) and not recompute_forking_paths:
                print(f"Results for prompt #{prompt_index} already exist, skipping")
                continue

            if dataset[prompt_index]["finish_reason"] != "stop":
                print(f"Base #{prompt_index} exceeded length, skipping")
                continue

            print("Question:")
            print(dataset[prompt_index]["question"])
            print("Base path:")
            print(dataset[prompt_index]["output_text"])

            # collect stumps for prompt
            stumps = collect_stumps(
                base_llm,
                dataset[prompt_index]['prompt_token_ids'],
                dataset[prompt_index]['output_token_ids'],
                include_sub_sentences=include_sub_sentences,
            )

            print("Done collecting stumps.")
            print(f"Number of stumps: {len(stumps)}")
            print("-" * 30)
            
            random_stump = random.choice(stumps)
            print("-" * 30)
            print(f"Random stump (t = {random_stump['t']}):")
            print(base_llm.get_tokenizer().decode(random_stump["stump_token_ids"], skip_special_tokens=True))

            # generate branches for prompt
            branches = generate_branches(
                base_llm,
                stumps,
                num_branches=num_branches,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            # save results
            with open(result_path, "w+") as f:
                json.dump(branches, f, indent=2)
            print(f"Saving results to {result_path}")

        # clear cache
        del base_llm
        clear_cuda()
    else:
        if base_llm is not None:
            del base_llm
            clear_cuda()

    # ── Answer parsing ───────────────────────────────────────────────
    print("Parsing final answers")
    if dataset_name.lower() == "game-of-24": # game of 24 parses manually!
        answer_llm = None
    else:
        answer_llm = LLM(model=answer_model_name, dtype="bfloat16")

    # Parse base-path answers if we just generated them (no prior collection data)
    if base_paths_to_parse is not None:
        dataset = parse_answer(answer_llm, base_paths_to_parse)
        os.makedirs(os.path.dirname(collection_path), exist_ok=True)
        with open(collection_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved collection data to {collection_path}")

    # parse results
    for prompt_index in range(start_index, end_index):
        # not the prettiest, but re-read the results we just generated
        print(f"Parsing answers for prompt #{prompt_index}")
        result_path = os.path.join(output_dir, f'{prompt_index:04d}.json')
        if not os.path.exists(result_path):
            print(f"Skipping prompt #{prompt_index}, no forking paths results found.")
            continue

        with open(result_path) as f:
            branches = json.load(f)
        
        # copy relevant information from datapoint
        datapoint = dataset[prompt_index]
        branch_dataset = [{
            "dataset_type": datapoint["dataset_type"],
            "dataset_name": datapoint["dataset_name"],
            "question": datapoint["question"],
            "all_answers": datapoint["all_answers"],
            "all_letters": datapoint["all_letters"],
            **branch # output_text comes from branch!
        } for branch in branches]

        # feed generated answers (ext_full) into answer extraction prompt template
        parse_results = parse_answer(
            answer_llm,
            branch_dataset
        )

        # save results to same path (overwrite)
        # NOTE: we're copying info from base answer; this is redundant! can remove info if taking up too much space
        with open(result_path, "w+") as f:
            json.dump(parse_results, f, indent=2)
        print(f"Saving parsed results to {result_path}")

    if answer_llm is not None:
        del answer_llm
        clear_cuda()

    # ── Compress to streamlit ────────────────────────────────────────
    if compress_to_streamlit:
        from compress_to_streamlit import compress_forking_paths, compress_base_data

        streamlit_folder = config['save_locations']['streamlit_folder']
        print("Compressing results for Streamlit...")
        compress_forking_paths(forking_paths_dir, streamlit_folder, model_nickname, dataset_name.lower())
        compress_base_data(data_dir, streamlit_folder, model_nickname, dataset_name.lower())
        print("Done compressing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate forking paths with vLLM")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="AQuA", help="Dataset name")
    parser.add_argument("--num_branches", type=int, default=30, help="Number of branches to generate")
    parser.add_argument("--max_new_tokens", type=int, default=3000, help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start_index", type=int, default=None, help="Start index for data processing (0 by default)")
    parser.add_argument("--end_index", type=int, default=None, help="End index for data processing (length of dataset by default)")
    parser.add_argument("--only_parse_answers", action='store_true', help="Only parse answers from existing forking paths data")
    parser.add_argument("--recompute_forking_paths", action='store_true', help="Recompute forking paths (overwrite saved files)")
    parser.add_argument("--include_sub_sentences", action='store_true', help="Also split on comma, colon, and semicolon delimiters")
    # base-path generation (used when collection data doesn't already exist)
    parser.add_argument("--num_examples", type=int, default=100, help="Number of dataset examples for base-path generation")
    parser.add_argument("--shuffle", action='store_true', default=True, help="Shuffle dataset before sampling (--shuffle / --no-shuffle)")
    # streamlit compression
    parser.add_argument("--compress_to_streamlit", action='store_true', help="Compress results for Streamlit after forking paths analysis")

    args = parser.parse_args()
    main(**vars(args))