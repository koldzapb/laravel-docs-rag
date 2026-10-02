"""Prompt construction and the LLM call (any OpenAI-compatible API: Groq, OpenAI, Ollama)."""

from openai import OpenAI

from laravel_docs_rag.chunking import Chunk
from laravel_docs_rag.config import DOCS_VERSION, Settings

# The exact sentence the model must use when the context does not contain the answer.
# Having a fixed sentence makes "did it decline?" a simple string check in the eval.
DECLINE_MESSAGE = f"I couldn't find this in the Laravel {DOCS_VERSION} documentation."

SYSTEM_PROMPT = f"""You answer questions about the Laravel {DOCS_VERSION} documentation.

Rules:
- Use ONLY the numbered context passages below. Do not use outside knowledge.
- After each claim, cite the passage(s) it comes from using square brackets with the passage
  number, e.g. "Batches are dispatched with Bus::batch [1][3]." Use no other citation format.
- Include short code examples from the context when they help.
- If the context does not contain the answer, reply with exactly this sentence and nothing \
else: "{DECLINE_MESSAGE}"
- Be concise and practical."""


# A handful of chunks are huge code samples (kept whole by the chunker). Capping what we send
# per chunk keeps the prompt within the provider's token limits.
MAX_CHARS_PER_PASSAGE = 4000


def build_context(chunks: list[Chunk]) -> str:
    """Format chunks as numbered passages. The numbers are what the model cites."""
    passages = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.text
        if len(text) > MAX_CHARS_PER_PASSAGE:
            text = text[:MAX_CHARS_PER_PASSAGE] + "\n[... passage truncated ...]"
        passages.append(f"[{i}] {chunk.title} ({chunk.file})\n{text}")
    return "\n\n".join(passages)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    """The chat messages sent to the LLM: rules in the system turn, context + question after."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n\n{build_context(chunks)}\n\nQuestion: {question}",
        },
    ]


def is_decline(answer: str) -> bool:
    """True if the model said the docs don't cover the question."""
    return DECLINE_MESSAGE.lower().rstrip(".") in answer.lower()


class LLMClient:
    """Minimal chat-completion client. The provider is chosen purely by base_url + key."""

    def __init__(self, settings: Settings) -> None:
        # max_retries makes the SDK back off and retry on rate limits (HTTP 429).
        self.client = OpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, max_retries=5
        )
        self.model = settings.llm_model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0,  # deterministic-ish answers; we want facts, not creativity
        )
        return (response.choices[0].message.content or "").strip()
