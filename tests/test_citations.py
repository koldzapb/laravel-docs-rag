from laravel_docs_rag.chunking import Chunk
from laravel_docs_rag.citations import build_sources, cited_numbers, doc_url, normalize_citations


def make_chunk(file: str, anchor: str | None, *headings: str) -> Chunk:
    return Chunk(id=f"{file}:0:0", file=file, headings=list(headings), anchor=anchor, text="...")


def test_doc_url_with_anchor():
    assert doc_url("queues.md", "job-batching") == (
        "https://laravel.com/docs/13.x/queues#job-batching"
    )


def test_doc_url_without_anchor_points_to_page():
    assert doc_url("eloquent-relationships.md", None) == (
        "https://laravel.com/docs/13.x/eloquent-relationships"
    )


def test_cited_numbers():
    assert cited_numbers("Use Bus::batch [1][3]. See also [3].") == {1, 3}
    assert cited_numbers("No citations here.") == set()


def test_normalize_gpt_oss_style_citations():
    text = "Use --tries 【5†L1-L5】 and backoff【2】."
    assert normalize_citations(text) == "Use --tries [5] and backoff [2]."


def test_build_sources_numbers_in_prompt_order_and_flags_cited():
    chunks = [
        make_chunk("queues.md", "job-batching", "Queues", "Job Batching"),
        make_chunk("events.md", None, "Events"),
    ]
    sources = build_sources(chunks, "Batches are great [1].")
    assert [s.number for s in sources] == [1, 2]
    assert sources[0].title == "Queues > Job Batching"
    assert sources[0].url == "https://laravel.com/docs/13.x/queues#job-batching"
    assert [s.cited for s in sources] == [True, False]
