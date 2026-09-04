"""Shared helpers for candidate answer banks."""


def flatten_bank_candidates(bank):
    """Flatten bank candidates into per-continuation lists.

    Candidates may carry multiple tokenization variants of the same answer
    string (``continuation_token_ids_variants``: canonical join and
    forced-brace split — the byte-pair encoder merges ``{-`` etc., so a
    single path can misestimate the string's probability). All variants of
    a candidate share its cluster id, so cluster-level aggregation
    (softmax sum / logsumexp) marginalizes over tokenization paths.
    Sample counts attach to the first variant only (they describe sampled
    strings, not paths).

    Returns (token_id_lists, cluster_ids, counts) of equal length.
    """
    token_lists, cluster_ids, counts = [], [], []
    for c in bank["candidates"]:
        variants = c.get("continuation_token_ids_variants") or [
            c["continuation_token_ids"]
        ]
        for j, ids in enumerate(variants):
            token_lists.append(ids)
            cluster_ids.append(c["cluster_id"])
            counts.append(c["count"] if j == 0 else 0)
    return token_lists, cluster_ids, counts
