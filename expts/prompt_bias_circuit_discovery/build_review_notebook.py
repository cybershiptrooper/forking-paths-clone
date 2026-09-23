"""Generate and execute ``bias_experiment_review.ipynb`` (repository root).

The notebook shows, end to end, the exact prompts and chains of thought
of the positives and negatives in (A) the judge sets used by the three
reports of 2026-09-18 and (B) the sanitised sets of this revision, answers
the question "prompt change or same prompt?", displays the prompt
chunking chosen per dataset on random prompts with the confound check,
renders the judge prompts, and tabulates the bias rates (Qwen3-8B and
Qwen3-32B) and the judge AUROCs where those results exist.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.build_review_notebook --execute
"""

from __future__ import annotations

import argparse
import os
import subprocess

import nbformat as nbf

SETUP = r'''
import json, glob, os, random, textwrap, collections
import numpy as np, pandas as pd
from IPython.display import display, Markdown, HTML
pd.set_option("display.max_colwidth", 200); pd.set_option("display.width", 250)
ROOT = os.getcwd()
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
def esc(s):
    return s.replace("<", "&lt;").replace(">", "&gt;")
def block(title, body, open_=False):
    display(HTML(f"<details{' open' if open_ else ''}><summary><b>{esc(title)}</b></summary><pre style='white-space:pre-wrap;font-size:12px;background:#f7f7f7;padding:8px;border-radius:6px'>{esc(body)}</pre></details>"))
def show_trace(rec, title, open_=True):
    txt = rec["output_text"]
    reasoning, _, after = txt.partition("</think>")
    reasoning = reasoning.replace("<think>", "").strip()
    meta = {k: rec.get(k) for k in ("uid","source_uid","case","family","is_positive","pushed_letter","reference_letter","biased_letter","clean_answer","verbalizes_cue","cue_mentions","delta","p_value","axis","value","fill_type","explicit_inserted") if rec.get(k) is not None}
    display(Markdown(f"**{title}**  \n`{json.dumps(meta)}`"))
    block("PROMPT (question with choices)", rec["question_with_choices"], open_=open_)
    block(f"CHAIN OF THOUGHT ({len(reasoning)} chars, full)", reasoning, open_=open_)
    block("FINAL ANSWER (after </think>)", after.strip(), open_=open_)
'''

Q1_MD = r'''
## Q1. Do we compare rollouts where the *prompt* changed, or different rollouts of the *same* prompt? How does the chain of thought differ?

**As run (reports of 2026-09-18).** Both designs were used and mixed:

| judge set | positives | negatives | design |
|---|---|---|---|
| resume (neutral-prompt negatives) | cue-variant traces with the pushed answer | traces of the **neutral** prompt of the same profile with the reference answer | prompt change (the negative prompt has no attribute at all, so a judge can read the label off the prompt) |
| resume, same prompt | the same positives | traces of the **cue variant itself** with the reference answer | same prompt |
| discrim-eval | pushed-answer traces of the 40 largest-effect variants | reference-answer traces of the same variant (27); **6 of the 20 variants had none**, so 12 positives were judged against negatives from other prompts | same prompt (partly) |
| BBQ | stereotyped-person traces | "unknown" traces of the same item | same prompt |

**How the chain of thought differs in the same-prompt design: it does not, in any way tied to the attribute.** Both traces list the same facts, weigh them, and land on opposite sides of a borderline call; the attribute is either absent from the text (implicit fills, names) or mentioned once and disclaimed (explicit fills). The examples below show this end to end. In the prompt-change design the *prompt* differs (a name or a stated attribute) and the chains of thought are again indistinguishable, except that the trace of a stated attribute usually names it and says it should not matter.

**Two defects found in the "as run" data.** (1) The discrim-eval *implicit* variants are separate paraphrases per variant in the public dataset, not name substitutions: the Tawa Altaha variant of question 9 is a one-line summary that drops the credit score, the steady income and the down payment that the Tyler Lewis variant states. Seven of the 20 selected variants, including all three with a full swing of 1.0, share less than half their text with their baseline; those "bias" effects are paraphrase effects. (2) With 8 rollouts per cell, the selection threshold of 0.25 is two rollouts, and no test was applied.

**Sanitised design (this revision).** Every family is a *minimal pair*: the resume prompts differ in exactly one sentence; the new discrim-eval prompts are built from one base text per (question, gender) and differ only in the race word (explicit) or the name (implicit), verified by a word-level diff. Each variant is sampled 16 times (screen), tested against its baseline with Fisher's exact test, and every screened variant is re-sampled 64 times with a fresh seed (confirmation) before it enters a judge set. The two cases are then built and evaluated separately: **prompt_change** (positives from the cue variant, negatives from an attribute-bearing prompt whose answer did not move) and **same_prompt** (both classes from the cue variant), plus an **explicit_control** in which one sentence naming the attribute as the reason is inserted, to check that the judge catches explicit bias.
'''


def make_nb():
    nb = nbf.v4.new_notebook()
    C = []
    C.append(nbf.v4.new_markdown_cell("# Review of the prompt-bias judge experiments: exact prompts, chains of thought, chunking and sanitised sets\n\n"
                                       "Companion to `notes/new_method_comparisons/{judge_bias_detection,prompt_to_trace_cue_masks,demographic_cue_behaviour}.md` "
                                       "and to the sanitised pipeline in `expts/prompt_bias_circuit_discovery/` (`build_discrim_minimal_pairs.py`, "
                                       "`analyze_bias_rates.py`, `build_judge_set_v2.py`, `judge_bias_v2.py`, `prompt_chunking.py`). "
                                       "Every trace below is printed in full (expand the blocks)."))
    C.append(nbf.v4.new_code_cell(SETUP))
    C.append(nbf.v4.new_markdown_cell(Q1_MD))
    # ---- A. old sets
    C.append(nbf.v4.new_markdown_cell("## A. The judge sets as run (2026-09-18): one positive and one matched negative per dataset, end to end"))
    C.append(nbf.v4.new_code_cell(r'''
OLD = {"resume (neutral-prompt negatives)": "results/prompt_bias/judge_resume_selection.json",
       "resume, same prompt": ("results/prompt_bias/judge_resume_selection.json", "results/prompt_bias/judge_resume_same_selection.json"),
       "discrim-eval (public fills)": "results/prompt_bias/judge_discrim_selection.json",
       "BBQ": "results/prompt_bias/judge_bbq_selection.json"}
for name, sp in OLD.items():
    display(Markdown(f"### A.{list(OLD).index(name)+1} {name}"))
    if isinstance(sp, tuple):
        selp, seln = load(sp[0]), load(sp[1])
        recp, recn = load(selp[0]["data_path"]), load(seln[0]["data_path"])
        p = [s for s in selp if s["is_positive"] and not s["verbalizes_cue"]][0]
        n = [s for s in seln if s["pair_uid"] == p["pair_uid"]][0]
        pairs = [(p, recp[p["prompt_index"]]), (n, recn[n["prompt_index"]])]
    else:
        sel = load(sp); recs = load(sel[0]["data_path"])
        pos = [s for s in sel if s["is_positive"]]
        pos_silent = [s for s in pos if not s["verbalizes_cue"]]
        p = (pos_silent or pos)[0]
        neg = [s for s in sel if not s["is_positive"] and s["pair_uid"] == p["pair_uid"]]
        if not neg:
            # discrim: the first positives have no same-prompt negative; pick a variant that has both
            for p2 in pos:
                neg = [s for s in sel if not s["is_positive"] and s["pair_uid"] == p2["pair_uid"]]
                if neg:
                    p = p2; break
        pairs = [(p, recs[p["prompt_index"]]), (neg[0], recs[neg[0]["prompt_index"]])]
    for s, r in pairs:
        r = dict(r); r.update({k: s.get(k) for k in ("tag","is_positive","biased_letter","reference_letter","verbalizes_cue","cue_mentions","axis","value","fill_type")})
        show_trace(r, ("POSITIVE (pushed answer)" if s["is_positive"] else "NEGATIVE (reference answer)") + f" - {s['tag']}, prompt uid {s['source_uid']}")
'''))
    C.append(nbf.v4.new_markdown_cell("### A.5 The paraphrase confound in the public discrim-eval implicit fills\n\nThe variants selected for the judge set and how much of their text they share with the white-man baseline of the same question and fill. Below 0.5 the two prompts are different paraphrases, not name substitutions."))
    C.append(nbf.v4.new_code_cell(r'''
import difflib
V = load("results/prompt_bias/discrim_rollouts_raw.json"); by = {v["uid"]: v for v in V}
sel = load("results/prompt_bias/judge_discrim_selection.json")
rows = []
for s in sel:
    if not s["is_positive"] or s["pair_uid"] in {r["variant"] for r in rows}: continue
    u = s["pair_uid"]; qid = u.split("_")[2]; b = by[f"discrim_{s['fill_type']}_{qid}_white_male"]
    rows.append(dict(variant=u, fill=s["fill_type"], delta=s["delta"], text_similarity_to_baseline=round(difflib.SequenceMatcher(None, by[u]["question"], b["question"]).ratio(), 2),
                     has_same_prompt_negative=any((not x["is_positive"]) and x["pair_uid"] == u for x in sel)))
display(pd.DataFrame(rows))
u = "discrim_implicit_q09_NativeAmerican_male"
block("q09 implicit, Native American male (the public dataset's own text)", by[u]["question"], open_=True)
block("q09 implicit, white male (the public dataset's own text)", by["discrim_implicit_q09_white_male"]["question"], open_=True)
'''))
    # ---- B. new sets
    C.append(nbf.v4.new_markdown_cell("## B. Sanitised judge sets: exact positives and negatives per family and case, end to end\n\n"
                                       "Built by `build_judge_set_v2.py` from confirmed cue effects (Fisher exact p < 0.05 and |delta| >= 0.25 on fresh 64-rollout samples where available; the screen sets are shown when the confirmation has not finished). "
                                       "Cases: **prompt_change** (negatives from an attribute-bearing prompt whose answer did not move), **same_prompt**, **explicit_control** (one sentence naming the attribute as the reason inserted before the decision)."))
    C.append(nbf.v4.new_code_cell(r'''
JS_DIR = "results/prompt_bias_v2/judge_sets" if glob.glob("results/prompt_bias_v2/judge_sets/*.json") else "results/prompt_bias_v2/judge_sets_screen"
display(Markdown(f"Judge sets in `{JS_DIR}`"))
summ = []
for f in sorted(glob.glob(f"{JS_DIR}/*_summary.json")):
    for k, v in load(f).items():
        summ.append(dict(model=os.path.basename(f).replace("_summary.json",""), set=k, **{kk: vv for kk, vv in v.items() if kk != "path"}))
display(pd.DataFrame(summ))
random.seed(0)
shown = set()
for f in sorted(glob.glob(f"{JS_DIR}/*.json")):
    if f.endswith("_summary.json"): continue
    recs = load(f); fam, case = recs[0]["family"], recs[0]["case"]
    if (recs[0]["model_tag"], fam, case) in shown: continue
    shown.add((recs[0]["model_tag"], fam, case))
    display(Markdown(f"### B. {recs[0]['model_tag']} / {fam} / {case}  ({sum(r['is_positive'] for r in recs)} pos, {sum(not r['is_positive'] for r in recs)} neg)"))
    pos = [r for r in recs if r["is_positive"]]; pos.sort(key=lambda r: r["verbalizes_cue"])
    p = pos[0]
    neg = [r for r in recs if not r["is_positive"] and (r["pair_uid"] == p["pair_uid"])]
    n = neg[0] if neg else [r for r in recs if not r["is_positive"]][0]
    show_trace(p, "POSITIVE: " + p["tag"], open_=(case != "explicit_control"))
    show_trace(n, "NEGATIVE: " + n["tag"], open_=(case == "same_prompt"))
'''))
    # ---- C. chunking
    C.append(nbf.v4.new_markdown_cell("## C. Prompt chunking: does the cue share a unit with other task information?\n\n"
                                       "Four chunkings of the question text (`prompt_chunking.py`): **sentence** (the pipeline's current unit), **llm** (an LLM splits the text into the smallest units of information, exact substrings, validated by reconstruction), "
                                       "**content** (one chunk per content word, spaCy POS, entities kept whole), **token**. The instruction line, each answer option and the chat-template tail are always their own chunks; the known cue span (name / attribute word / demographic sentence) is forced to be its own chunk under llm and content. "
                                       "A cue chunk is *clean* when the text of the chunk outside the cue contains no content word (role nouns such as 'candidate' are listed separately). The best method per family is the coarsest one that is clean on every prompt."))
    C.append(nbf.v4.new_code_cell(r'''
CS = load("results/prompt_bias_v2/analysis/confound_sweep.json")
if CS is None:
    display(Markdown("confound sweep not finished"))
else:
    rows = []
    for fam, ms in CS["families"].items():
        for m, s in ms.items():
            if m == "best": continue
            rows.append(dict(family=fam, method=m, n_prompts=s["n"], frac_cue_chunks_clean=round(s["frac_clean"], 3), mean_prompt_chunks=round(s["mean_chunks"], 1), llm_fallbacks=s.get("n_llm_fallback", 0), best=(m == ms["best"])))
    display(pd.DataFrame(rows))
    random.seed(1)
    for fam, det in CS["details"].items():
        best = CS["families"][fam]["best"]
        display(Markdown(f"### C. {fam}: best = **{best}**; three random prompts"))
        for d in random.sample(det, 3):
            m = d["methods"][best]
            cue = d["cue"]
            html = ["<div style='font-size:12px;line-height:1.7'>"]
            for kind, text in m["chunks"]:
                t = esc(text).replace("\n", "&#9166;")
                is_cue = any(c in text for c in cue)
                color = {"instruction": "#e8e8e8", "tail": "#e8e8e8", "choice": "#dfe9ff", "choices_header": "#dfe9ff"}.get(kind, "#fff3c4" if not is_cue else "#ffb3b3")
                html.append(f"<span style='background:{color};border:1px solid #bbb;border-radius:4px;padding:1px 3px;margin:1px;display:inline-block'>{t}</span>")
            html.append("</div>")
            display(Markdown(f"**{d['uid']}** cue = {cue}; {m['n_chunks']} chunks; cue chunks clean = {m['all_clean']}; cue chunks: {[(t, res, role) for t, res, role in m['cue_chunks']]}"))
            display(HTML("".join(html)))
'''))
    # ---- D. judge prompts
    C.append(nbf.v4.new_markdown_cell("## D. The judge prompts\n\nThe v2 prompt (`judge_bias_v2.py`) is the same text as the prompt used in the reports except for the answer format (a 0-100 probability instead of a 0-10 score, since the OpenRouter judges give no logprobs). The ceiling condition adds one line naming the attribute family. For a local judge the one-word yes/no variant is also asked and P(yes) is read from the logits. The attribution-table condition of the report is not part of v2 (its input, the teacher-forced single-read ablations, carried no signal)."))
    C.append(nbf.v4.new_code_cell(r'''
from expts.prompt_bias_circuit_discovery.judge_bias_v2 import build_messages, FORMAT_JSON, FORMAT_YESNO
JS_DIR = "results/prompt_bias_v2/judge_sets" if glob.glob("results/prompt_bias_v2/judge_sets/*.json") else "results/prompt_bias_v2/judge_sets_screen"
f = sorted(glob.glob(f"{JS_DIR}/*resume_same_prompt.json"))[0]; rec = [r for r in load(f) if r["is_positive"]][0]
rec2 = dict(rec); rec2["output_text"] = rec["output_text"][:1200] + " [... reasoning truncated for display; the judge sees it in full ...]</think>" + rec["output_text"].split("</think>")[-1][:300]
block("BASELINE judge prompt (JSON probability format)", build_messages(rec2, "baseline", FORMAT_JSON), open_=True)
block("CEILING judge prompt: differs from baseline only by the 'Focus on one characteristic' line", build_messages(rec2, "ceiling", FORMAT_JSON)[:900] + "\n[...]", open_=True)
block("Yes/no variant (local judge only): last line", FORMAT_YESNO, open_=True)
f = sorted(glob.glob(f"{JS_DIR}/*resume_explicit_control.json"))[0]; rec = [r for r in load(f) if r["is_positive"]][0]
r = rec["output_text"].split("</think>")[0]
block("EXPLICIT CONTROL: the inserted sentence (last 700 chars of the reasoning)", r[-700:], open_=True)
'''))
    # ---- E. bias rates
    C.append(nbf.v4.new_markdown_cell("## E. Bias rates per family: Qwen3-8B and Qwen3-32B, screen (16 rollouts) and confirmation (64 fresh rollouts)\n\n"
                                       "`screen` = |delta| >= 0.25 and Fisher p < 0.05 against the baseline of the same question (BBQ: stereotyped person chosen in >= half the rollouts and more often than the other person, binomial p < 0.05). `q` is the Benjamini-Hochberg adjusted p over the family."))
    C.append(nbf.v4.new_code_cell(r'''
rows = []
for f in sorted(glob.glob("results/prompt_bias_v2/analysis/bias_rates_*.json")):
    d = load(f)
    for fam, s in d["summary"].items():
        rows.append(dict(run=os.path.basename(f).replace("bias_rates_","").replace(".json",""), family=fam, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}))
display(pd.DataFrame(rows))
for img in ["bias_rate_by_family", "delta_distributions"]:
    display(Markdown(f"![{img}](notes/images/prompt_bias_v2/{img}.png)"))
'''))
    C.append(nbf.v4.new_markdown_cell("### E.2 Family-level effects in the papers' terms\n\n"
                                       "For each cue value: the mean over questions of P(pushed) - P(pushed | baseline) with a Wilcoxon signed-rank p over questions, and the paired **McNemar** test of Karvonen & Marks / Arcuschin et al. on the majority decision per prompt (discordant pairs: accept only under the variant vs only under the baseline), with a Bonferroni threshold over the family's cue values. Includes the three families built from the two papers' public datasets (`build_paper_prompts.py`)."))
    C.append(nbf.v4.new_code_cell(r'''
rows = []
for f in sorted(glob.glob("results/prompt_bias_v2/analysis/bias_rates_*screen.json")):
    d = load(f); run = os.path.basename(f).replace("bias_rates_","").replace(".json","")
    mc = {(g["family"], g["value"]): g for g in d.get("mcnemar_by_value", [])}
    for g in d.get("pooled_by_value", []):
        m = mc.get((g["family"], g["value"]), {})
        rows.append(dict(run=run, family=g["family"], value=g["value"], n_questions=g["n_questions"], mean_delta=round(g["mean_delta"], 3),
                         wilcoxon_p=f'{g["wilcoxon_p"]:.1e}', pooled_fisher_p=f'{g["pooled_fisher_p"]:.1e}',
                         mcnemar_discordant=f'+{m.get("discordant_variant_accepts","-")}/-{m.get("discordant_baseline_accepts","-")}',
                         mcnemar_p=(f'{m["mcnemar_p"]:.1e}' if m else None), mcnemar_bonferroni_sig=m.get("significant_bonferroni")))
df = pd.DataFrame(rows)
display(df[df.wilcoxon_p.astype(float) < 0.05].sort_values(["run","family","mean_delta"]) if len(df) else Markdown("none"))
display(Markdown("![family effects](notes/images/prompt_bias_v2/family_effects.png)"))
'''))
    C.append(nbf.v4.new_markdown_cell("## F. Judge AUROC per case, condition and judge (where run)"))
    C.append(nbf.v4.new_code_cell(r'''
rows = []
for f in sorted(glob.glob("results/prompt_bias_v2/judge/*.json")):
    if "cache" in f or os.path.basename(f).startswith("test_"): continue
    d = load(f)
    for k, v in d["summary"].items():
        case, cond, key = k.split("|")
        rows.append(dict(file=os.path.basename(f).replace(".json",""), judge=d["args"]["judge"], case=case, condition=cond, score=key, n_pos=v["n_pos"], n_neg=v["n_neg"],
                         auroc=round(v["auroc"], 3) if v["auroc"] is not None else None, ci_lo=round(v["ci"][0], 2) if v["ci"][0] is not None else None, ci_hi=round(v["ci"][1], 2) if v["ci"][1] is not None else None,
                         silent_auroc=round(v["silent_auroc"], 3) if v["silent_auroc"] is not None else None))
display(pd.DataFrame(rows) if rows else Markdown("no judge results yet"))
for img in ["judge_auroc_prompt_change", "judge_auroc_prompt_change_same_answer", "judge_auroc_same_prompt", "judge_auroc_explicit_control",
            "judge_tpr1fpr_prompt_change", "judge_tpr1fpr_prompt_change_same_answer", "judge_tpr1fpr_same_prompt", "judge_tpr1fpr_explicit_control", "judge_reasoning_increment"]:
    if os.path.exists(f"notes/images/prompt_bias_v2/{img}.png"):
        display(Markdown(f"![{img}](notes/images/prompt_bias_v2/{img}.png)"))
display(Markdown("Full table with the reasoning increment and TPR at 1% FPR: `uv run python -m expts.prompt_bias_circuit_discovery.summarize_judges`"))
'''))
    nb["cells"] = C
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bias_experiment_review.ipynb")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    nb = make_nb()
    nbf.write(nb, args.out)
    print("wrote", args.out)
    if args.execute:
        subprocess.run(["uv", "run", "python", "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=1800", args.out], check=True)
        print("executed", args.out)


if __name__ == "__main__":
    main()
