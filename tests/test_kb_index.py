import tempfile
import unittest
from pathlib import Path

from app.kb_index import (
    MULTILINGUAL_MODEL_NAME, build_knowledge_index, load_knowledge_index, save_knowledge_index,
    search_knowledge_index,
)
from app.kb_sources import KnowledgeSource, TextSection


class FakeEmbedding:
    def embed(self, texts):
        return tuple((1.0, 0.0) if "current" in text else (0.0, 1.0) for text in texts)


class RecordingEmbedding:
    def __init__(self) -> None:
        self.calls = []

    def embed(self, texts):
        self.calls.append(tuple(texts))
        return tuple((1.0, 0.0) for _ in texts)


class NearScoreEmbedding:
    def embed(self, texts):
        return tuple(
            (0.96, 0.28) if "paper text" in text else (1.0, 0.0)
            for text in texts
        )


class KnowledgeIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = (
            KnowledgeSource("old", "paper", "旧论文", "old.docx", "4.0", False,
                            (TextSection("段落 1-2", "old method"),)),
            KnowledgeSource("new", "paper", "当前论文", "new.docx", "5.1", True,
                            (TextSection("段落 3-4", "current method"),)),
            KnowledgeSource("note", "obsidian", "笔记", "note.md", None, False,
                            (TextSection("行 1-2", "other observation"),)),
            KnowledgeSource("research-note", "obsidian", "odor/方法.md", "method.md",
                            None, False, (TextSection("行 1-3", "current note"),)),
        )

    def test_current_paper_only_by_default_and_history_opt_in(self) -> None:
        index = build_knowledge_index(self.sources, FakeEmbedding(), model_name="fake")
        current = search_knowledge_index("current question", index, FakeEmbedding(), top_k=3)
        self.assertNotIn("旧论文", [result.source_title for result in current])
        self.assertEqual(current[0].source_title, "当前论文")
        self.assertEqual(current[0].locator, "段落 3-4")
        self.assertEqual(current[0].page_number, 0)
        self.assertNotIn("笔记", [result.source_title for result in current])
        all_sources = search_knowledge_index(
            "current question", index, FakeEmbedding(), top_k=4, scope="all"
        )
        self.assertIn("笔记", [result.source_title for result in all_sources])

        history = search_knowledge_index(
            "old question", index, FakeEmbedding(), top_k=2, include_history=True
        )
        self.assertIn("旧论文", [result.source_title for result in history])
        self.assertIn("当前论文", [result.source_title for result in history])
        only_current = search_knowledge_index(
            "old question", index, FakeEmbedding(), top_k=3,
            scope="paper", paper_versions=("5.1",),
        )
        self.assertEqual({result.version for result in only_current}, {"5.1"})

    def test_roundtrip_preserves_provenance(self) -> None:
        index = build_knowledge_index(self.sources, FakeEmbedding(), model_name="fake")
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_knowledge_index(save_knowledge_index(index, Path(directory) / "kb.json"))
        self.assertEqual(index, loaded)

    def test_rejects_invalid_chunk_settings(self) -> None:
        with self.assertRaises(ValueError):
            build_knowledge_index(
                self.sources, FakeEmbedding(), model_name="fake", chunk_size=100, overlap=100
            )

    def test_multilingual_e5_uses_passage_and_query_prefixes(self) -> None:
        model = RecordingEmbedding()
        index = build_knowledge_index(
            self.sources[:1], model, model_name=MULTILINGUAL_MODEL_NAME
        )
        search_knowledge_index("问题", index, model, include_history=True)
        self.assertTrue(model.calls[0][0].startswith("passage: "))
        self.assertEqual(model.calls[-1], ("query: 问题",))

    def test_research_results_keep_relevant_current_paper(self) -> None:
        sources = (
            KnowledgeSource("note-a", "obsidian", "odor/a.md", "a.md", None, False,
                            (TextSection("行 1", "note a"),)),
            KnowledgeSource("note-b", "obsidian", "odor/b.md", "b.md", None, False,
                            (TextSection("行 1", "note b"),)),
            KnowledgeSource("paper", "paper", "Current", "paper.docx", "5.1", True,
                            (TextSection("段落 1", "paper text"),)),
        )
        index = build_knowledge_index(sources, NearScoreEmbedding(), model_name="fake")
        results = search_knowledge_index("query", index, NearScoreEmbedding(), top_k=2)
        self.assertIn("Current", [result.source_title for result in results])

    def test_duplicate_zotero_records_do_not_repeat_identical_evidence(self) -> None:
        sources = (
            KnowledgeSource("z1", "zotero", "DiGress", "zotero:1", None, False,
                            (TextSection("摘要", "same abstract"),)),
            KnowledgeSource("z2", "zotero", "DiGress", "zotero:2", None, False,
                            (TextSection("摘要", "same abstract"),)),
        )
        index = build_knowledge_index(sources, RecordingEmbedding(), model_name="fake")
        results = search_knowledge_index(
            "query", index, RecordingEmbedding(), scope="literature", top_k=3
        )
        self.assertEqual(len(results), 1)

    def test_can_limit_literature_to_one_pdf_title(self) -> None:
        sources = (
            KnowledgeSource("metadata", "zotero", "DiGress", "zotero:1", None,
                            False, (TextSection("摘要", "summary"),)),
            KnowledgeSource("pdf", "zotero_pdf", "DiGress", "paper.pdf", None,
                            False, (TextSection("第 2 页", "full text"),)),
            KnowledgeSource("other", "zotero_pdf", "Other", "other.pdf", None,
                            False, (TextSection("第 4 页", "full text"),)),
        )
        index = build_knowledge_index(sources, RecordingEmbedding(), model_name="fake")
        results = search_knowledge_index(
            "query", index, RecordingEmbedding(), scope="literature_pdf",
            source_title="digress", top_k=3,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_number, 2)


if __name__ == "__main__":
    unittest.main()
