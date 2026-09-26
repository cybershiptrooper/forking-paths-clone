"""Prepare a proposed Task 5 pilot manifest from released data; no model inference.

Only metadata and tokenizer lengths are inspected. Labels and source paths in
this file are evaluation metadata and must never enter a judge packet.
"""
from collections import Counter, defaultdict
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import statistics
import unicodedata

SOURCE = Path('data/external/cot-proxy-tasks')
OUT = Path('results/prompt_bias_v2/end_to_end_protocol_0924')
SALT = 'monitorability-protocol-20260924-v1'
SUFFIX = '</think>\n\n**Final answer ('


def rank(value):
    return hashlib.sha256((SALT + '\0' + str(value)).encode()).hexdigest()


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).lower().split())


def main():
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    from expts.prompt_bias_circuit_discovery.audit_released_testbed_0924 import thinking_text

    rows = [json.loads(line) for line in Path(
        'results/prompt_bias_v2/testbed_audit_0924/released_index.jsonl'
    ).read_text().splitlines()]
    rows = [row for row in rows if row['task'] == 5]
    # Data integrity: compare question text across released split boundaries.
    text_splits = defaultdict(set)
    for row in rows:
        raw = json.loads(Path(row['source_file']).read_text())
        text_splits[normalized(raw['question_text'])].add(row['split'])
    overlaps = [sorted(splits) for splits in text_splits.values() if len(splits) > 1]
    assert not overlaps, overlaps

    selected = []
    for role, split, per_stratum in [('development', 'train', 1),
                                     ('calibration', 'val', 2),
                                     ('id_test', 'test', 6)]:
        used = set()
        candidates = [row for row in rows if row['split'] == split]
        for label in [0, 1]:
            for target in ['A', 'B']:
                groups = defaultdict(list)
                for row in candidates:
                    if row['label'] == label and row['target'] == target:
                        groups[row['base_id']].append(row)
                bases = sorted(groups, key=rank)
                bases = [base for base in bases if base not in used][:per_stratum]
                assert len(bases) == per_stratum
                for base in bases:
                    row = min(groups[base], key=lambda r: rank(r['source_file']))
                    selected.append(dict(row, role=role))
                    used.add(base)
    # Retain all OOD base-question/class units. Different hint directions of
    # the same question remain one bootstrap cluster, not independent units.
    groups = defaultdict(list)
    for row in rows:
        if row['split'] == 'ood_test':
            groups[row['base_id'], row['label']].append(row)
    for key in sorted(groups, key=rank):
        selected.append(dict(min(groups[key], key=lambda r: rank(r['source_file'])),
                             role='ood_test'))

    constants = {}
    for node in ast.parse((SOURCE/'src/tasks/hinted_cot/prompts.py').read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.targets[0], ast.Name):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    tokenizer_config = Path(cached_file('Qwen/Qwen3-32B', 'tokenizer_config.json', local_files_only=True))
    tokenizer_revision = tokenizer_config.parent.name
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-32B', revision=tokenizer_revision, local_files_only=True)
    for row in selected:
        raw = json.loads(Path(row['source_file']).read_text())
        assert raw['answer'] == raw['hint_letter']
        labels = list(raw['choices'])
        content = constants['STANFORD_PROFESSOR_PROMPT'].format(
            question=raw['question_text'],
            choices='\n'.join(f'{label}) {raw["choices"][label]}' for label in labels),
            hint_value=raw['hint_value'], label_list=', '.join(labels))
        prompt = tokenizer.apply_chat_template(
            [dict(role='user', content=content)], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        reasoning = thinking_text(raw['thinking'])
        row['candidate_sequence_tokens'] = len(tokenizer.encode(
            prompt + reasoning + SUFFIX, add_special_tokens=False))
        row['source_sha256'] = hashlib.sha256(Path(row['source_file']).read_bytes()).hexdigest()
        row['contains_think_delimiter_in_stored_reasoning'] = ('<think>' in reasoning or '</think>' in reasoning)
    assert len(selected) == 60
    summary = {}
    for role in ['development', 'calibration', 'id_test', 'ood_test']:
        rr = [row for row in selected if row['role'] == role]
        lengths = sorted(row['candidate_sequence_tokens'] for row in rr)
        summary[role] = dict(n_traces=len(rr), n_base_questions=len({r['base_id'] for r in rr}),
                            label_counts=dict(Counter(r['label'] for r in rr)),
                            label_hint_counts=dict(Counter(f'{r["label"]}_{r["target"]}' for r in rr)),
                            min_tokens=min(lengths), median_tokens=statistics.median(lengths),
                            max_tokens=max(lengths))
    manifest = dict(
        status='PROPOSED; runner and cost gates pending; no model runs launched',
        source_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        selection_salt=SALT, source_task=5, actor='Qwen/Qwen3-32B',
        tokenizer_revision=tokenizer_revision,
        rule='One deterministic hash-selected trace per selected question; OOD one per base-question/class unit. No length or model-score filtering.',
        bootstrap_cluster='base_id', exact_question_text_overlap_across_splits=0,
        monitor_packet_warning='All labels, rates, source names, variants and pair IDs here are evaluation metadata. Build a separate allowlisted packet.',
        suffix=SUFFIX, summary=summary, rows=selected)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'candidate_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print('Stored-reasoning delimiters:', sum(r['contains_think_delimiter_in_stored_reasoning'] for r in selected))
    print('No inference or judge calls. Full source traces retained.')


if __name__ == '__main__':
    main()
