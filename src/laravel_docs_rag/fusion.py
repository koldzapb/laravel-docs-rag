"""Reciprocal Rank Fusion (RRF): merge several ranked lists into one.

Each list votes for its items with 1 / (k + rank). Only ranks are used, never raw scores,
so a cosine similarity (0..1) and a BM25 score (0..30+) can be combined without any
normalisation. k=60 is the value from the original paper (Cormack et al., 2009); it damps
the advantage of being #1 so that items ranked well by *both* lists win.
"""

from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of ids. Returns (id, score) pairs, highest score first.

    >>> reciprocal_rank_fusion([["a", "b"], ["b", "c"]])[0][0]
    'b'
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    # sorted() is stable, so ties keep first-seen order and results are deterministic.
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
