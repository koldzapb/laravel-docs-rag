"""Turning retrieved chunks into numbered, clickable sources."""

import re
from dataclasses import dataclass

from laravel_docs_rag.chunking import Chunk
from laravel_docs_rag.config import DOCS_VERSION

DOCS_BASE_URL = "https://laravel.com/docs"
CITATION_RE = re.compile(r"\[(\d+)\]")
# gpt-oss models sometimes cite in their training format: 【5†L1-L5】 or 【1】.
FANCY_CITATION_RE = re.compile(r"\s?【(\d+)(?:†[^】]*)?】")


@dataclass
class Source:
    """One numbered source shown to the user. `number` matches the [n] markers in the answer."""

    number: int
    title: str
    url: str
    file: str
    cited: bool  # did the answer actually reference this source?


def doc_url(file: str, anchor: str | None, version: str = DOCS_VERSION) -> str:
    """Build the laravel.com URL for a page section.

    >>> doc_url("queues.md", "job-batching")
    'https://laravel.com/docs/13.x/queues#job-batching'
    """
    url = f"{DOCS_BASE_URL}/{version}/{file.removesuffix('.md')}"
    return f"{url}#{anchor}" if anchor else url


def normalize_citations(answer: str) -> str:
    """Rewrite other citation styles to our [n] style: "works 【5†L1-L5】" -> "works [5]"."""
    return FANCY_CITATION_RE.sub(lambda m: f" [{m.group(1)}]", answer)


def cited_numbers(answer: str) -> set[int]:
    """Return the source numbers referenced in an answer, e.g. "see [1] and [3]" -> {1, 3}."""
    return {int(n) for n in CITATION_RE.findall(answer)}


def build_sources(chunks: list[Chunk], answer: str) -> list[Source]:
    """Number the context chunks 1..n (same order as in the prompt) and flag the cited ones."""
    cited = cited_numbers(answer)
    return [
        Source(
            number=i,
            title=chunk.title,
            url=doc_url(chunk.file, chunk.anchor),
            file=chunk.file,
            cited=i in cited,
        )
        for i, chunk in enumerate(chunks, start=1)
    ]
