"""Direct-answer circuit discovery.

After every analysis-timestep prefix, append a fixed suffix
(``" </think> I think the answer is "``) and read the model's logit
distribution over a discrete set of answer tokens (e.g. A/B/C/D) at the
position right after the suffix.  Replaces vLLM-sampled branches +
self-normalised importance sampling with a single deterministic forward
pass per (prefix, mask).

Two entry points:

- :mod:`learn` — drives existing :mod:`utils.circuit_discovery`
  algorithms (IG, activation patching, ...) with a synthetic single
  continuation and a one-position position mask.
- :mod:`suppress` — attention-suppression discovery (Thought Anchors)
  that measures KL on the answer-token logits instead of on prefix
  tokens.
"""
