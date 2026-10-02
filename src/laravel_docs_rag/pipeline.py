"""The RAG pipeline: retrieve -> build prompt -> generate -> attach sources."""

from dataclasses import dataclass
from typing import Protocol

from laravel_docs_rag.chunking import Chunk
from laravel_docs_rag.citations import Source, build_sources, normalize_citations
from laravel_docs_rag.generation import DECLINE_MESSAGE, build_messages, is_decline
from laravel_docs_rag.retrieval import Mode


class SearchesChunks(Protocol):
    """Anything with a `search` method (the real Retriever, or a fake in tests)."""

    def search(self, query: str, mode: Mode, k: int) -> list[Chunk]: ...


class CompletesChat(Protocol):
    """Anything with a `complete` method (the real LLMClient, or a fake in tests)."""

    def complete(self, messages: list[dict[str, str]]) -> str: ...


@dataclass
class Answer:
    """What the user gets back."""

    question: str
    answer: str
    sources: list[Source]
    declined: bool
    mode: Mode


@dataclass
class RagPipeline:
    retriever: SearchesChunks
    llm: CompletesChat
    top_k: int = 5
    default_mode: Mode = Mode.HYBRID

    def ask(self, question: str, mode: Mode | None = None, top_k: int | None = None) -> Answer:
        """Answer a question from the docs, with numbered sources matching the [n] citations."""
        mode = mode or self.default_mode
        chunks = self.retriever.search(question, mode, top_k or self.top_k)
        if not chunks:
            # Nothing retrieved: don't even ask the model, it could only guess.
            return Answer(question, DECLINE_MESSAGE, [], declined=True, mode=mode)

        text = normalize_citations(self.llm.complete(build_messages(question, chunks)))
        declined = is_decline(text)
        sources = [] if declined else build_sources(chunks, text)
        return Answer(question, text, sources, declined=declined, mode=mode)


def build_pipeline() -> RagPipeline:
    """Wire up the real components from environment settings."""
    from laravel_docs_rag.config import get_settings
    from laravel_docs_rag.generation import LLMClient
    from laravel_docs_rag.retrieval import Retriever

    settings = get_settings()
    return RagPipeline(
        retriever=Retriever.from_settings(settings, load_reranker=settings.use_rerank),
        llm=LLMClient(settings),
        top_k=settings.top_k,
        default_mode=Mode.HYBRID_RERANK if settings.use_rerank else Mode.HYBRID,
    )
