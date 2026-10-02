"""The retriever: vector search, BM25, hybrid (RRF) and hybrid + rerank behind one method."""

from dataclasses import dataclass
from enum import StrEnum

from laravel_docs_rag.bm25 import BM25Index
from laravel_docs_rag.chunking import Chunk
from laravel_docs_rag.config import Settings
from laravel_docs_rag.embeddings import Embedder
from laravel_docs_rag.fusion import reciprocal_rank_fusion
from laravel_docs_rag.rerank import Reranker
from laravel_docs_rag.vector_store import VectorStore


class Mode(StrEnum):
    """Retrieval strategies, compared against each other by the eval harness."""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class IndexNotReadyError(RuntimeError):
    """Raised when the vector store is empty, i.e. `ingest` has not been run yet."""


class RerankerDisabledError(RuntimeError):
    """Raised when hybrid_rerank is requested but the reranker was not loaded."""


@dataclass
class Retriever:
    """Holds the indexes and models needed to answer `search(query, mode, k)`."""

    store: VectorStore
    embedder: Embedder
    bm25: BM25Index
    chunks_by_id: dict[str, Chunk]
    reranker: Reranker | None
    candidates: int = 30

    @classmethod
    def from_settings(cls, settings: Settings, load_reranker: bool = True) -> "Retriever":
        """Open the persisted index and load the models. Takes a few seconds."""
        store = VectorStore(settings.chroma_dir)
        if store.count() == 0:
            raise IndexNotReadyError("The index is empty. Run `rag ingest` first.")
        chunks = store.all_chunks()
        return cls(
            store=store,
            embedder=Embedder(settings.embedding_model, settings.model_cache_dir),
            bm25=BM25Index([c.id for c in chunks], [c.embedding_text() for c in chunks]),
            chunks_by_id={c.id: c for c in chunks},
            reranker=Reranker(settings.rerank_model, settings.model_cache_dir)
            if load_reranker
            else None,
            candidates=settings.candidates,
        )

    def vector_search(self, query: str, n: int) -> list[str]:
        return self.store.query(self.embedder.embed_query(query), n)

    def bm25_search(self, query: str, n: int) -> list[str]:
        return self.bm25.search(query, n)

    def hybrid_search(self, query: str, n: int) -> list[str]:
        """Fuse the top `candidates` of each retriever with RRF and keep the best `n`."""
        fused = reciprocal_rank_fusion(
            [self.vector_search(query, self.candidates), self.bm25_search(query, self.candidates)]
        )
        return [chunk_id for chunk_id, _ in fused[:n]]

    def search(self, query: str, mode: Mode, k: int) -> list[Chunk]:
        """Return the `k` best chunks for `query` using the given strategy."""
        mode = Mode(mode)  # also accepts plain strings like "hybrid"
        if mode is Mode.VECTOR:
            ids = self.vector_search(query, k)
        elif mode is Mode.BM25:
            ids = self.bm25_search(query, k)
        elif mode is Mode.HYBRID:
            ids = self.hybrid_search(query, k)
        else:
            if self.reranker is None:
                raise RerankerDisabledError(
                    "Reranking is disabled. Set USE_RERANK=true to enable it."
                )
            candidates = [self.chunks_by_id[i] for i in self.hybrid_search(query, self.candidates)]
            return self.reranker.rerank(query, candidates, k)
        return [self.chunks_by_id[i] for i in ids]
