"""Construct the deterministic answer probe.

A probe is a fixed token sequence appended to a prefix.  When the model
runs forward on ``prefix + suffix + placeholder``, the logits at the
position whose next token would be the placeholder give the model's
distribution over the answer tokens.

The position-mask plumbing in :mod:`utils.circuit_discovery` is reused
unchanged: every algorithm accepts ``position_mask_overrides`` (one per
continuation), and local objectives consult that mask to decide which
positions contribute to the loss.  We supply a mask that is True at
exactly one position — the one whose logits encode the answer
distribution.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch


DEFAULT_SUFFIX = " </think> I think the answer is"
DEFAULT_ANSWER_LETTERS = [" A", " B", " C", " D"]


@dataclass
class AnswerProbe:
    """Tokenized representation of a fixed answer probe.

    ``continuation_ids`` is a single ``(1, suffix_len + 1)`` tensor —
    suffix tokens followed by one *placeholder* token (we use the first
    answer letter; the placeholder's identity does not matter because
    the loss reads logits *before* it).

    ``position_mask`` is a ``(1, prefix_len + suffix_len + 1)`` float
    tensor with a single 1.0 at the position whose logits predict the
    answer letter (i.e. ``prefix_len + suffix_len - 1``, since logits at
    position *i* predict token *i+1*).  Caller may need to re-build
    ``position_mask`` if ``prefix_len`` changes.
    """

    suffix: str
    answer_letters: List[str]
    suffix_ids: torch.Tensor          # (suffix_len,) long
    answer_token_ids: torch.Tensor    # (num_answers,) long
    placeholder_id: int

    @property
    def suffix_len(self) -> int:
        return int(self.suffix_ids.shape[-1])

    @property
    def continuation_len(self) -> int:
        return self.suffix_len + 1

    def num_answers(self) -> int:
        return int(self.answer_token_ids.shape[-1])

    def make_continuation(self, device: torch.device) -> torch.Tensor:
        """Return ``(1, suffix_len + 1)`` continuation tensor on *device*."""
        cont = torch.cat(
            [
                self.suffix_ids,
                torch.tensor([self.placeholder_id], dtype=torch.long),
            ],
            dim=-1,
        ).unsqueeze(0)
        return cont.to(device)

    def make_position_mask(
        self, prefix_len: int, device: torch.device,
    ) -> torch.Tensor:
        """Single-position mask over ``prefix + continuation`` tokens.

        The mask is 1.0 at exactly the position whose logits predict the
        answer letter — i.e. the last suffix token's position.  Logits at
        position *i* predict token *i+1*, so for a continuation laid out
        as ``[suffix..., placeholder]`` starting at index ``prefix_len``,
        the answer-prediction position is ``prefix_len + suffix_len - 1``.
        """
        full_len = prefix_len + self.continuation_len
        mask = torch.zeros(1, full_len, device=device)
        answer_logit_pos = prefix_len + self.suffix_len - 1
        mask[0, answer_logit_pos] = 1.0
        return mask

    def answer_logit_position(self, prefix_len: int) -> int:
        return prefix_len + self.suffix_len - 1


def build_answer_probe(
    tokenizer,
    suffix: str = DEFAULT_SUFFIX,
    answer_letters: Optional[List[str]] = None,
) -> AnswerProbe:
    """Tokenize *suffix* and *answer_letters*, returning an :class:`AnswerProbe`.

    The full probe text ``suffix + letter`` is tokenized end-to-end for
    each letter; the *shared prefix* across all letters is taken as the
    suffix's token ids, and the *trailing token* is the letter's answer
    token id.  This handles BPE merge boundaries correctly — e.g. a
    Llama tokenizer merges a trailing space in *suffix* with the leading
    letter into one token, so a naïve ``encode(suffix) + encode(letter)``
    would mis-align.

    Raises ``ValueError`` if any letter does not encode to a single
    trailing token (i.e. the shared prefix is not consistent across
    letters), or if the prefix differs across letters.
    """
    if answer_letters is None:
        answer_letters = list(DEFAULT_ANSWER_LETTERS)

    full_encodings: List[List[int]] = []
    for letter in answer_letters:
        ids = tokenizer.encode(suffix + letter, add_special_tokens=False)
        if not ids:
            raise ValueError(
                f"Empty tokenization for suffix+letter={suffix + letter!r}"
            )
        full_encodings.append(ids)

    base_len = len(full_encodings[0]) - 1
    shared_prefix = full_encodings[0][:base_len]
    answer_ids: List[int] = []
    for letter, ids in zip(answer_letters, full_encodings):
        if len(ids) != base_len + 1:
            raise ValueError(
                f"suffix + {letter!r} tokenizes to {len(ids)} tokens; expected "
                f"{base_len + 1}. The letter must add exactly one token after "
                f"the suffix. Pick a different suffix or letter formatting "
                f"(e.g. include leading space in the letter)."
            )
        if ids[:base_len] != shared_prefix:
            raise ValueError(
                f"Inconsistent shared prefix across answer letters: letter "
                f"{letter!r} produced prefix {ids[:base_len]!r}, expected "
                f"{shared_prefix!r}. Letters must share an identical "
                f"suffix-tokenization."
            )
        answer_ids.append(ids[-1])

    return AnswerProbe(
        suffix=suffix,
        answer_letters=list(answer_letters),
        suffix_ids=torch.tensor(shared_prefix, dtype=torch.long),
        answer_token_ids=torch.tensor(answer_ids, dtype=torch.long),
        placeholder_id=int(answer_ids[0]),
    )


def answer_probs_from_logits(
    logits: torch.Tensor,
    probe: AnswerProbe,
    prefix_len: int,
    renormalize: bool = True,
) -> torch.Tensor:
    """Extract P(letter) for each answer letter from a logits tensor.

    Args:
        logits: ``(batch, full_len, vocab)`` — full-sequence logits from
            a forward pass on ``prefix + continuation``.
        probe: The :class:`AnswerProbe`.
        prefix_len: Length of the prefix (so we know which row to read).
        renormalize: If True (default), softmax over only the answer
            tokens — gives a proper distribution on {A, B, C, D}.  If
            False, returns full-vocab softmax probabilities for the
            answer tokens (they'll not sum to 1).

    Returns:
        ``(num_answers,)`` tensor of probabilities.
    """
    pos = probe.answer_logit_position(prefix_len)
    row = logits[0, pos].float()
    if renormalize:
        ans_logits = row[probe.answer_token_ids.to(row.device)]
        return torch.softmax(ans_logits, dim=-1)
    full_probs = torch.softmax(row, dim=-1)
    return full_probs[probe.answer_token_ids.to(row.device)]


def answer_logprobs_from_logits(
    logits: torch.Tensor,
    probe: AnswerProbe,
    prefix_len: int,
    renormalize: bool = True,
) -> torch.Tensor:
    """Like :func:`answer_probs_from_logits` but returns log-probabilities."""
    pos = probe.answer_logit_position(prefix_len)
    row = logits[0, pos].float()
    if renormalize:
        ans_logits = row[probe.answer_token_ids.to(row.device)]
        return torch.log_softmax(ans_logits, dim=-1)
    full_lp = torch.log_softmax(row, dim=-1)
    return full_lp[probe.answer_token_ids.to(row.device)]
