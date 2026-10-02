import pytest

from laravel_docs_rag.chunking import Chunk
from laravel_docs_rag.evaluation import (
    GoldRef,
    compute_metrics,
    first_hit_rank,
    is_hit,
    normalize_heading,
)


def chunk(file: str, *headings: str) -> Chunk:
    return Chunk(id="x", file=file, headings=list(headings), anchor=None, text="")


def test_normalize_heading():
    assert normalize_heading("  The `queue:work`   Command ") == "the queue:work command"


def test_hit_matches_any_heading_in_path():
    gold = [GoldRef("queues.md", "job batching")]
    assert is_hit(chunk("queues.md", "Queues", "Job Batching", "Dispatching Batches"), gold)
    assert not is_hit(chunk("events.md", "Events", "Job Batching"), gold)  # wrong file
    assert not is_hit(chunk("queues.md", "Queues", "Unique Jobs"), gold)


def test_first_hit_rank():
    gold = [GoldRef("queues.md", "Unique Jobs")]
    chunks = [chunk("queues.md", "Queues"), chunk("queues.md", "Queues", "Unique Jobs")]
    assert first_hit_rank(chunks, gold) == 2
    assert first_hit_rank(chunks[:1], gold) is None


def test_compute_metrics():
    metrics = compute_metrics([1, 3, None, 12])
    assert metrics.hit_at_1 == 0.25
    assert metrics.hit_at_5 == 0.5
    assert metrics.mrr_at_10 == pytest.approx((1 + 1 / 3) / 4)  # rank 12 is beyond @10
