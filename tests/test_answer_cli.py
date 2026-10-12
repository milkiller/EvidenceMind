from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.answer_cli import format_answer, main
from app.answering import Answer, Citation
from app.retriever import SearchResult


class RecordingGenerator:
    """Fake generator used to prove the CLI sends retrieved evidence onward."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[SearchResult, ...]]] = []

    def generate(self, question: str, evidence: tuple[SearchResult, ...]) -> str:
        self.calls.append((question, evidence))
        return "基于证据的回答"


class AnswerCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = SimpleNamespace(
            config=SimpleNamespace(model_name="test-model", normalize_embeddings=True)
        )

    def test_main_refuses_without_evidence_and_skips_gateway(self) -> None:
        generator_factory_calls: list[str] = []
        output = StringIO()

        with (
            patch("app.answer_cli.load_vector_index", return_value=self.index),
            patch("app.answer_cli.search_vector_index", return_value=()),
            redirect_stdout(output),
        ):
            exit_code = main(
                ["index.json", "没有证据的问题", "--offline"],
                embedding_factory=lambda **_: object(),
                generator_factory=lambda config: generator_factory_calls.append(config),
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().strip(), "证据不足，无法基于当前文档回答。")
        self.assertEqual(generator_factory_calls, [])

    def test_main_calls_gateway_and_prints_deterministic_citations(self) -> None:
        evidence = (SearchResult("chunk-5", 5, "第五页证据", 0.8),)
        generator = RecordingGenerator()
        output = StringIO()

        with (
            patch("app.answer_cli.load_vector_index", return_value=self.index),
            patch("app.answer_cli.search_vector_index", return_value=evidence),
            redirect_stdout(output),
        ):
            exit_code = main(
                ["index.json", "请总结", "--config", "custom.yaml"],
                embedding_factory=lambda **_: object(),
                generator_factory=lambda config: generator if config == "custom.yaml" else None,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(generator.calls, [("请总结", evidence)])
        rendered = output.getvalue()
        self.assertIn("基于证据的回答", rendered)
        self.assertIn("第 5 页 | Chunk：chunk-5", rendered)

    def test_format_answer_omits_empty_citation_heading(self) -> None:
        self.assertEqual(
            format_answer(Answer("证据不足", (), False)),
            "证据不足",
        )
        self.assertIn(
            "第 3 页 | Chunk：chunk-3",
            format_answer(Answer("回答", (Citation("chunk-3", 3, "摘要"),), True)),
        )


if __name__ == "__main__":
    unittest.main()
