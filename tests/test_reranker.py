"""Reranking contracts use deterministic scores and never download model weights."""

import unittest

from app.kb_index import KnowledgeSearchResult
from app.reranker import rerank_candidates


class FakeReranker:
    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, passages: tuple[str, ...]) -> tuple[float, ...]:
        self.calls.append((query, passages))
        return self.scores


def hit(number: int, similarity: float = 0.8) -> KnowledgeSearchResult:
    return KnowledgeSearchResult(
        f"chunk-{number}", number, f"passage {number}", similarity,
        "Named paper", "zotero_pdf", f"第 {number} 页", None,
    )


class RerankTests(unittest.TestCase):
    def test_target_can_move_from_dense_rank_seven_to_first(self) -> None:
        candidates = tuple(hit(number) for number in range(1, 21))
        scorer = FakeReranker(tuple(10.0 if n == 7 else -float(n) for n in range(1, 21)))
        ranked = rerank_candidates("question", candidates, scorer, top_k=3)
        self.assertEqual([item.chunk_id for item in ranked], ["chunk-7", "chunk-1", "chunk-2"])
        self.assertEqual(ranked[0].locator, "第 7 页")
        self.assertEqual(ranked[0].similarity, 0.8)
        self.assertEqual(len(scorer.calls[0][1]), 20)

    def test_ties_keep_dense_order(self) -> None:
        self.assertEqual(
            [item.chunk_id for item in rerank_candidates(
                "question", (hit(1), hit(2)), FakeReranker((0.0, 0.0)), top_k=2,
            )],
            ["chunk-1", "chunk-2"],
        )

    def test_invalid_scores_fail_closed(self) -> None:
        for scores in ((1.0,), (float("nan"), 0.0), (float("inf"), 0.0)):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                rerank_candidates("question", (hit(1), hit(2)), FakeReranker(scores), top_k=2)

    def test_empty_candidates_do_not_call_model(self) -> None:
        scorer = FakeReranker(())
        self.assertEqual(rerank_candidates("question", (), scorer, top_k=3), ())
        self.assertEqual(scorer.calls, [])


if __name__ == "__main__":
    unittest.main()
