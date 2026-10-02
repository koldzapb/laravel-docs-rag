"""Keyword search with BM25 (via rank_bm25), built in memory from the stored chunks.

Vector search is good at meaning ("send work to the background"), BM25 is good at exact
terms (`ShouldBeUnique`, `queue:work`, `--tries`). Hybrid search uses both.
"""

import re

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9_]+")
# Words so common in questions/docs that they carry no signal ("laravel" is on every page).
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
        "how", "i", "if", "in", "is", "it", "laravel", "my", "of", "on", "or", "should", "so",
        "that", "the", "this", "to", "use", "using", "what", "when", "where", "which", "with",
        "you", "your",
    }
)  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Lowercase, split on anything that isn't a letter/digit/underscore, drop stopwords.

    "php artisan queue:work --tries=3" -> ["php", "artisan", "queue", "work", "tries", "3"]
    """
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Maps chunk ids to BM25 scores for a query."""

    def __init__(self, chunk_ids: list[str], texts: list[str]) -> None:
        self.chunk_ids = chunk_ids
        self.bm25 = BM25Okapi([tokenize(text) for text in texts])

    def search(self, query: str, n: int) -> list[str]:
        """Return the ids of the `n` best-scoring chunks (score > 0), best first."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunk_ids[i] for i in ranked[:n] if scores[i] > 0]
