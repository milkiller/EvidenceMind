from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.retriever import SearchResult
from evaluation.eval_dense import EvalCase, evaluate_cases, format_summary, load_eval_cases


class DenseEvaluationTests(unittest.TestCase):
    def test_load_eval_cases_rejects_empty_page_labels(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            eval_path = Path(temporary_directory) / "eval.json"
            eval_path.write_text(
                '{"schema_version": 1, "cases": [{"id": "q1", "question": "问题", "expected_pages": []}]}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "needs positive expected_pages"):
                load_eval_cases(eval_path)

    def test_evaluate_cases_computes_page_recall_and_mrr(self) -> None:
        cases = (
            EvalCase("q1", "问题一", frozenset({2, 5})),
            EvalCase("q2", "问题二", frozenset({7})),
        )
        results = {
            "问题一": (
                SearchResult("chunk-1", 1, "无关", 0.9),
                SearchResult("chunk-2", 2, "相关", 0.8),
                SearchResult("chunk-3", 2, "重复页", 0.7),
            ),
            "问题二": (SearchResult("chunk-4", 7, "相关", 0.8),),
        }

        summary = evaluate_cases(cases, lambda question: results[question])

        self.assertEqual(summary.scores[0].retrieved_pages, (1, 2, 2))
        self.assertTrue(summary.scores[0].hit_at_k)
        self.assertEqual(summary.scores[0].recall_at_k, 0.5)
        self.assertEqual(summary.scores[0].reciprocal_rank, 0.5)
        self.assertEqual(summary.hit_rate_at_k, 1.0)
        self.assertEqual(summary.recall_at_k, 0.75)
        self.assertEqual(summary.mean_reciprocal_rank, 0.75)

    def test_format_summary_shows_metrics_and_case_details(self) -> None:
        summary = evaluate_cases(
            (EvalCase("q1", "问题", frozenset({3})),),
            lambda _: (SearchResult("chunk-3", 3, "相关", 0.8),),
        )

        rendered = format_summary(summary, top_k=3)

        self.assertIn("Hit@3：1.000", rendered)
        self.assertIn("Recall@3：1.000", rendered)
        self.assertIn("q1 | 期望页 [3] | 检索页 [3]", rendered)
