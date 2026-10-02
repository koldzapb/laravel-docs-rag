"""Ingest: download docs -> chunk -> embed -> store in Chroma."""

import time

from laravel_docs_rag.chunking import Chunk, chunk_markdown
from laravel_docs_rag.config import Settings
from laravel_docs_rag.docs_source import download_docs, list_doc_files
from laravel_docs_rag.embeddings import Embedder
from laravel_docs_rag.vector_store import VectorStore


def load_chunks(settings: Settings) -> list[Chunk]:
    """Chunk every downloaded page."""
    chunks: list[Chunk] = []
    for path in list_doc_files(settings.docs_dir):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    return chunks


def run_ingest(settings: Settings, refresh_docs: bool = False) -> int:
    """Build the index from scratch. Returns the number of chunks indexed."""
    if refresh_docs or not list_doc_files(settings.docs_dir):
        print(f"Downloading Laravel docs into {settings.docs_dir} ...")
        download_docs(settings.docs_dir)

    chunks = load_chunks(settings)
    pages = len({c.file for c in chunks})
    print(f"Chunked {pages} pages into {len(chunks)} chunks.")

    started = time.perf_counter()
    embedder = Embedder(settings.embedding_model, settings.model_cache_dir)
    embeddings = embedder.embed_documents([c.embedding_text() for c in chunks])
    print(f"Embedded {len(chunks)} chunks in {time.perf_counter() - started:.0f}s.")

    store = VectorStore(settings.chroma_dir)
    store.reset()
    store.add(chunks, embeddings)
    print(f"Stored {store.count()} chunks in Chroma at {settings.chroma_dir}.")
    return len(chunks)


def download_models(settings: Settings) -> None:
    """Pre-download the embedding and reranking models (used when building the Docker image)."""
    from laravel_docs_rag.rerank import Reranker

    Embedder(settings.embedding_model, settings.model_cache_dir)
    Reranker(settings.rerank_model, settings.model_cache_dir)
    print(f"Models cached in {settings.model_cache_dir}.")
