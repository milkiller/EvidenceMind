"""The document tool only releases bounded, traceable search hits."""

import unittest
from unittest.mock import patch

from app.agent_core import ToolRegistry
from app.agent_tools import register_knowledge_search
from app.kb_index import KnowledgeSearchResult


class FakeReranker:
    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, passages: tuple[str, ...]) -> tuple[float, ...]:
        self.calls.append((query, passages))
        return self.scores


def hit(number: int, similarity: float) -> KnowledgeSearchResult:
    return KnowledgeSearchResult(
        f"chunk-{number}", number, f"passage {number}", similarity,
        "Named paper", "zotero_pdf", f"第 {number} 页", None,
    )


class KnowledgeToolTests(unittest.TestCase):
    def test_paper_scope_ignores_model_generated_source_title(self) -> None:
        registry = ToolRegistry()
        with patch("app.agent_tools.search_knowledge_index", return_value=()) as search:
            register_knowledge_search(
                registry, object(), object(), scope="paper",
                user_question="当前论文中 AE32 如何表示气味条件？",
            )
            output = registry.execute_result(
                "search_knowledge",
                {"query": "AE32", "source_title": "模型臆测的论文完整标题"},
            )
        self.assertIn("证据不足", output.content)
        self.assertEqual(search.call_args.kwargs["scope"], "paper")
        self.assertIsNone(search.call_args.kwargs["source_title"])

    def test_optional_rerank_filters_threshold_before_scoring(self) -> None:
        registry = ToolRegistry()
        scorer = FakeReranker((1.0, 2.0))
        with patch("app.agent_tools.search_knowledge_index", return_value=(
            hit(1, 0.9), hit(2, 0.2), hit(3, 0.8),
        )) as search:
            register_knowledge_search(
                registry, object(), object(), top_k=1, candidate_k=20,
                min_similarity=0.5, reranker=scorer,
            )
            output = registry.execute_result(
                "search_knowledge", {"query": "question", "source_title": ""},
            )
        self.assertEqual(search.call_args.kwargs["top_k"], 20)
        self.assertEqual(scorer.calls[0][1], ("passage 1", "passage 3"))
        self.assertEqual(output.evidence_ids, ("chunk-3",))
        self.assertIn("相似度：0.8000", output.content)

    def test_rerank_needs_enough_candidates(self) -> None:
        with self.assertRaises(ValueError):
            register_knowledge_search(
                ToolRegistry(), object(), object(), top_k=3, candidate_k=2,
                reranker=FakeReranker(()),
            )

    def test_search_returns_bounded_source_and_location(self) -> None:
        result = KnowledgeSearchResult(
            "chunk-1", 0, "evidence " * 200, 0.8,
            "Current paper", "paper", "段落 12-14", "5.1",
        )
        registry = ToolRegistry()
        with patch("app.agent_tools.search_knowledge_index", return_value=(result,)) as search:
            register_knowledge_search(
                registry, object(), object(), scope="paper", top_k=2,
                min_similarity=0.5, max_evidence_chars=180,
            )
            tool_output = registry.execute_result(
                "search_knowledge", {"query": "odor condition", "source_title": ""}
            )
            observation = tool_output.content

        self.assertLessEqual(len(observation), 180)
        self.assertIn("Current paper", observation)
        self.assertIn("段落 12-14", observation)
        self.assertIn("chunk-1", observation)
        self.assertEqual(tool_output.evidence_ids, ("chunk-1",))
        self.assertEqual(search.call_args.kwargs["scope"], "paper")
        self.assertFalse(search.call_args.kwargs["include_history"])
        self.assertIsNone(search.call_args.kwargs["source_title"])

    def test_no_matching_evidence_tells_model_it_is_insufficient(self) -> None:
        result = KnowledgeSearchResult(
            "chunk-1", 0, "weak evidence", 0.2,
            "Paper", "paper", "段落 1", "5.1",
        )
        registry = ToolRegistry()
        with patch("app.agent_tools.search_knowledge_index", return_value=(result,)):
            register_knowledge_search(registry, object(), object(), min_similarity=0.5)
            observation = registry.execute("search_knowledge", {"query": "question", "source_title": ""})
        self.assertIn("证据不足", observation)

    def test_named_pdf_is_resolved_before_search(self) -> None:
        from app.kb_index import KnowledgeChunk, KnowledgeIndex

        chunk = KnowledgeChunk(
            "id-1", "source-1", "zotero_pdf", "DiGress: Discrete Denoising diffusion for graph generation",
            "local.pdf", None, False, "第 1 页", "content", (1.0,),
        )
        index = KnowledgeIndex("fake", 1, 800, 120, (chunk,))
        registry = ToolRegistry()
        with patch("app.agent_tools.search_knowledge_index", return_value=()) as search:
            register_knowledge_search(registry, index, object(), scope="literature_pdf")
            registry.execute("search_knowledge", {"query": "graph edits", "source_title": "DiGress"})
        self.assertEqual(search.call_args.kwargs["source_title"], chunk.title)

    def test_title_must_be_visible_in_user_question(self) -> None:
        from app.kb_index import KnowledgeChunk, KnowledgeIndex
        from app.agent_core import InvalidToolArgumentsError

        chunk = KnowledgeChunk(
            "id-1", "source-1", "zotero_pdf", "DiGress", "local.pdf", None,
            False, "第 1 页", "content", (1.0,),
        )
        registry = ToolRegistry()
        with patch("app.agent_tools.search_knowledge_index") as search:
            register_knowledge_search(
                registry, KnowledgeIndex("fake", 1, 800, 120, (chunk,)), object(),
                user_question="介绍这篇论文的方法",
            )
            with self.assertRaises(InvalidToolArgumentsError):
                registry.execute("search_knowledge", {"query": "method", "source_title": "DiGress"})
        search.assert_not_called()

    def test_ambiguous_or_unknown_title_does_not_search(self) -> None:
        from app.kb_index import KnowledgeChunk, KnowledgeIndex
        from app.agent_core import InvalidToolArgumentsError

        chunks = tuple(
            KnowledgeChunk(str(i), str(i), "zotero_pdf", title, "local.pdf", None,
                           False, "第 1 页", "content", (1.0,))
            for i, title in enumerate(("Diffusion A", "Diffusion B"))
        )
        registry = ToolRegistry()
        with patch("app.agent_tools.search_knowledge_index") as search:
            register_knowledge_search(
                registry, KnowledgeIndex("fake", 1, 800, 120, chunks), object(),
            )
            for title in ("Diffusion", "Missing"):
                with self.assertRaises(InvalidToolArgumentsError):
                    registry.execute("search_knowledge", {"query": "method", "source_title": title})
        search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
