import pytest

from laravel_docs_rag.fusion import reciprocal_rank_fusion


def test_item_ranked_well_by_both_lists_wins():
    vector = ["a", "b", "c"]
    bm25 = ["b", "d", "a"]
    fused = [item for item, _ in reciprocal_rank_fusion([vector, bm25])]
    assert fused[:2] == ["b", "a"]  # b: 1/62 + 1/61, a: 1/61 + 1/63
    assert set(fused) == {"a", "b", "c", "d"}


def test_scores_follow_the_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["a"]], k=60))
    assert fused["a"] == pytest.approx(2 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_ties_keep_first_seen_order():
    fused = reciprocal_rank_fusion([["x"], ["y"]])
    assert [item for item, _ in fused] == ["x", "y"]


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
