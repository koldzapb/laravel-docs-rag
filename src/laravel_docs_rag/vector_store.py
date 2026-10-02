"""Persistent vector index on Chroma.

Chroma stores, per chunk: the id, our precomputed embedding, the chunk text and its metadata.
It is also the single source of truth for chunks: the BM25 index is rebuilt from it at startup.
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from laravel_docs_rag.chunking import Chunk

COLLECTION_NAME = "laravel_docs"


def chunk_to_metadata(chunk: Chunk) -> dict[str, str]:
    """Chroma metadata values must be scalars, so the heading list is stored as JSON."""
    return {
        "file": chunk.file,
        "headings": json.dumps(chunk.headings),
        "anchor": chunk.anchor or "",
    }


def chunk_from_record(chunk_id: str, text: str, metadata: dict) -> Chunk:
    """Rebuild a Chunk from what Chroma returns."""
    return Chunk(
        id=chunk_id,
        file=metadata["file"],
        headings=json.loads(metadata["headings"]),
        anchor=metadata["anchor"] or None,
        text=text,
    )


class VectorStore:
    """A thin wrapper so the rest of the app talks in Chunks, not Chroma dictionaries."""

    def __init__(self, path: Path) -> None:
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self._get_collection()

    def _get_collection(self) -> chromadb.Collection:
        # Cosine distance matches how BGE embeddings are meant to be compared.
        return self.client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        """Delete everything (used by `ingest` to rebuild from scratch)."""
        if COLLECTION_NAME in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(COLLECTION_NAME)
        self.collection = self._get_collection()

    def add(
        self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 500
    ) -> None:
        """Insert chunks with their embeddings, in batches (Chroma limits request size)."""
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings[start : start + batch_size],
                documents=[c.text for c in batch],
                metadatas=[chunk_to_metadata(c) for c in batch],
            )

    def query(self, embedding: list[float], n: int) -> list[str]:
        """Return the ids of the `n` nearest chunks, best first."""
        result = self.collection.query(query_embeddings=[embedding], n_results=n, include=[])
        return result["ids"][0]

    def count(self) -> int:
        return self.collection.count()

    def all_chunks(self) -> list[Chunk]:
        """Load every stored chunk (a few thousand, fits easily in memory)."""
        result = self.collection.get(include=["documents", "metadatas"])
        return [
            chunk_from_record(chunk_id, text, metadata)
            for chunk_id, text, metadata in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        ]
