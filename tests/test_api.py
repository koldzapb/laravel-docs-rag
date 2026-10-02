"""API tests with a fake retriever and a fake LLM: no models, no network, no index needed."""

import pytest
from fastapi.testclient import TestClient

from laravel_docs_rag.api import app, get_pipeline
from laravel_docs_rag.chunking import Chunk
from laravel_docs_rag.generation import DECLINE_MESSAGE
from laravel_docs_rag.pipeline import RagPipeline
from laravel_docs_rag.retrieval import IndexNotReadyError, Mode

BATCH_CHUNK = Chunk(
    id="queues.md:1:0",
    file="queues.md",
    headings=["Queues", "Job Batching", "Dispatching Batches"],
    anchor="dispatching-batches",
    text="Use `Bus::batch([...])->then(...)->dispatch()`.",
)


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, Mode, int]] = []

    def search(self, query: str, mode: Mode, k: int) -> list[Chunk]:
        self.calls.append((query, mode, k))
        return self.chunks[:k]


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self.reply


@pytest.fixture
def client_with():
    """Returns a factory that installs a fake pipeline and gives back a TestClient."""

    def make(chunks: list[Chunk], reply: str) -> tuple[TestClient, RagPipeline]:
        pipeline = RagPipeline(FakeRetriever(chunks), FakeLLM(reply), top_k=5)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        return TestClient(app), pipeline

    yield make
    app.dependency_overrides.clear()


def test_health(client_with):
    client, _ = client_with([BATCH_CHUNK], "irrelevant")
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_cited_sources(client_with):
    client, pipeline = client_with([BATCH_CHUNK], "Call `Bus::batch` then `dispatch` [1].")
    response = client.post("/ask", json={"question": "How do I dispatch a batch?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Call `Bus::batch` then `dispatch` [1]."
    assert body["declined"] is False
    assert body["sources"] == [
        {
            "number": 1,
            "title": "Queues > Job Batching > Dispatching Batches",
            "url": "https://laravel.com/docs/13.x/queues#dispatching-batches",
            "cited": True,
        }
    ]
    # The retrieved chunk really went into the prompt, numbered [1].
    prompt = pipeline.llm.messages[-1]["content"]
    assert "[1] Queues > Job Batching > Dispatching Batches (queues.md)" in prompt
    assert "Question: How do I dispatch a batch?" in prompt


def test_ask_passes_mode_and_top_k(client_with):
    client, pipeline = client_with([BATCH_CHUNK], "ok [1]")
    client.post("/ask", json={"question": "batches?", "mode": "bm25", "top_k": 3})
    assert pipeline.retriever.calls == [("batches?", Mode.BM25, 3)]


def test_declined_answer_has_no_sources(client_with):
    client, _ = client_with([BATCH_CHUNK], DECLINE_MESSAGE)
    body = client.post("/ask", json={"question": "How do I configure Django?"}).json()
    assert body["declined"] is True
    assert body["sources"] == []


def test_no_retrieved_chunks_declines_without_calling_llm(client_with):
    client, pipeline = client_with([], "should not be used")
    body = client.post("/ask", json={"question": "anything at all"}).json()
    assert body == {"answer": DECLINE_MESSAGE, "declined": True, "mode": "hybrid", "sources": []}
    assert pipeline.llm.messages == []


def test_validation_rejects_empty_question(client_with):
    client, _ = client_with([BATCH_CHUNK], "x")
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_empty_index_returns_503():
    def not_ready() -> RagPipeline:
        raise IndexNotReadyError("The index is empty. Run `rag ingest` first.")

    app.dependency_overrides[get_pipeline] = not_ready
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "ingest" in response.json()["detail"]


def test_index_page_is_served(client_with):
    client, _ = client_with([], "x")
    response = client.get("/")
    assert response.status_code == 200
    assert "Ask the" in response.text
