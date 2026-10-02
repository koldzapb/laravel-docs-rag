"""Local text embeddings with fastembed (ONNX on CPU, no API key, no PyTorch)."""

from pathlib import Path

from fastembed import TextEmbedding


class Embedder:
    """Wraps a fastembed model. BGE models expect a query prefix, which `query_embed` adds."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages to be stored in the index."""
        return [vector.tolist() for vector in self.model.passage_embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question."""
        return next(iter(self.model.query_embed(text))).tolist()
