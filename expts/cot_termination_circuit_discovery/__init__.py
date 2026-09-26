"""Chain-of-thought termination circuit discovery.

Learns sentence-pair attention masks whose objective is to raise the
probability that the model's remaining chain of thought terminates
(emits ``</think>``) within a fixed token horizon *with the correct
answer*.  Candidates are sampled continuations of at most a few hundred
tokens, so the self-normalised importance-sampling weights over them do
not collapse (the failure mode documented for full-length chains in the
April 2026 notes).

Trainer files (``learn.py``, ``run.py``, ``probe.py``,
``answer_bank_utils.py``) are copies of their counterparts in
``expts/direct_answer_circuit_discovery/`` — deliberately copies, not
imports, since that folder changes between experiments.
"""
