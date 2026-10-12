import unittest
from collections.abc import Sequence

from app.answering import Answer, Citation, answer_question
from app.retriever import SearchResult


class RecordingGenerator:
    """A test double that records a generation request."""

    def __init__(self, answer_text: str = "基于证据的回答") -> None:
        self.answer_text = answer_text
        self.calls: list[tuple[str, Sequence[SearchResult]]] = []

    def generate(self, question: str, evidence: Sequence[SearchResult]) -> str:
        self.calls.append((question, evidence))
        return self.answer_text


class AnsweringTests(unittest.TestCase):
    def test_empty_evidence_refuses_without_calling_generator(self) -> None:
        generator = RecordingGenerator()

        answer = answer_question("这个问题怎么回答？", (), generator)

        self.assertEqual(
            answer,
            Answer("证据不足，无法基于当前文档回答。", (), False),
        )
        self.assertEqual(generator.calls, [])

    def test_answer_keeps_citation_metadata_and_evidence_order(self) -> None:
        evidence = (
            SearchResult("chunk-2", 2, "第二页证据", 0.8),
            SearchResult("chunk-7", 7, "第七页证据", 0.7),
        )
        generator = RecordingGenerator("有依据的回答")

        answer = answer_question("请总结", evidence, generator)

        self.assertTrue(answer.is_grounded)
        self.assertEqual(answer.text, "有依据的回答")
        self.assertEqual(
            answer.citations,
            (
                Citation("chunk-2", 2, "第二页证据"),
                Citation("chunk-7", 7, "第七页证据"),
            ),
        )
        self.assertEqual(generator.calls, [("请总结", evidence)])


if __name__ == "__main__":
    unittest.main()
