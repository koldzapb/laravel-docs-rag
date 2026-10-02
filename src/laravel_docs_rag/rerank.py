"""Reranking with a local cross-encoder.

The embedder encodes question and chunk separately (fast, but approximate). A cross-encoder
reads the question and one chunk *together* and outputs a relevance score: much more
accurate, but too slow to run over thousands of chunks. So we use it only on the top
~30 candidates from hybrid search.
"""

from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

from laravel_docs_rag.chunking import Chunk


class Reranker:
    """Reorders candidate chunks by cross-encoder relevance to the query."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """Return the `top_n` most relevant chunks, best first."""
        if not chunks:
            return []
        scores = list(self.model.rerank(query, [c.embedding_text() for c in chunks]))
        ranked = sorted(zip(scores, chunks, strict=True), key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_n]]
