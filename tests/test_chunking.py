import unittest
from pathlib import Path

from app.chunking import (
    ChunkConfig,
    ChunkStrategy,
    _build_chunk_id,
    chunk_document,
    split_chars_with_overlap,
    split_fixed_chars,
    split_fixed_tokens,
    split_paragraphs,
)
from app.pdf_reader import PDFDocument, PageText


class ChunkingPracticeTests(unittest.TestCase):
    def test_chunk_document_preserves_page_metadata_and_order(self) -> None:
        document = PDFDocument(
            path=Path("paper.pdf"),
            filename="paper.pdf",
            page_count=3,
            pages=(
                PageText(page_number=1, text="abcdef"),
                PageText(page_number=2, text="   "),
                PageText(page_number=3, text="xy"),
            ),
        )
        config = ChunkConfig(
            strategy=ChunkStrategy.FIXED_CHAR,
            chunk_size=3,
        )

        chunks = chunk_document(document, config)

        self.assertEqual([chunk.text for chunk in chunks], ["abc", "def", "xy"])
        self.assertEqual([chunk.page_number for chunk in chunks], [1, 1, 3])
        self.assertEqual([chunk.position for chunk in chunks], [0, 1, 0])
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [
                "paper.pdf-page1-pos0",
                "paper.pdf-page1-pos1",
                "paper.pdf-page3-pos0",
            ],
        )
        self.assertTrue(
            all(chunk.strategy is ChunkStrategy.FIXED_CHAR for chunk in chunks)
        )

    # @unittest.skip("TODO: implement split_fixed_chars, then remove this decorator")
    def test_fixed_chars_keeps_short_final_chunk(self) -> None:
        self.assertEqual(split_fixed_chars("abcdefg", 3), ("abc", "def", "g"))

    # @unittest.skip("TODO: implement split_chars_with_overlap, then remove this decorator")
    def test_char_overlap_repeats_boundary_text(self) -> None:
        self.assertEqual(
            split_chars_with_overlap("abcdefgh", chunk_size=4, overlap=2),
            ("abcd", "cdef", "efgh"),
        )

    def test_chunk_size_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            split_chars_with_overlap(
                "abcdefgh",
                chunk_size=0,
                overlap=1,
            )

    # @unittest.skip("TODO: implement split_fixed_tokens, then remove this decorator")
    def test_fixed_tokens_uses_supplied_tokenizer(self) -> None:
        encode = lambda text: [ord(character) for character in text]
        decode = lambda tokens: "".join(chr(token) for token in tokens)

        chunks = split_fixed_tokens(
            "abcdefg",
            chunk_size=3,
            encode=encode,
            decode=decode,
        )

        self.assertEqual(chunks, ("abc", "def", "g"))

    # @unittest.skip("TODO: implement split_paragraphs, then remove this decorator")
    def test_paragraphs_group_text_within_budget(self) -> None:
        text = "第一段。\n\n第二段。\n\n第三段更长。"

        chunks = split_paragraphs(text, chunk_size=10)

        self.assertEqual(chunks, ("第一段。\n\n第二段。", "第三段更长。"))

    def test_paragraph_separator_counts_toward_budget(self) -> None:
        for budget, expected in (
            (4, ("ab", "cd", "ef")),
            (5, ("ab", "cd", "ef")),
            (6, ("ab\n\ncd", "ef")),
        ):
            with self.subTest(budget=budget):
                chunks = split_paragraphs("ab\n\ncd\n\nef", budget)
                self.assertEqual(chunks, expected)
                self.assertTrue(all(len(chunk) <= budget for chunk in chunks))

    def test_paragraphs_preserve_one_oversized_paragraph(self) -> None:
        self.assertEqual(
            split_paragraphs("ab\n\nabcdefg\n\ncd", 4),
            ("ab", "abcdefg", "cd"),
        )

    def test_paragraph_document_keeps_metadata_after_budget_split(self) -> None:
        document = PDFDocument(
            path=Path("paper.pdf"),
            filename="paper.pdf",
            page_count=1,
            pages=(PageText(page_number=1, text="ab\n\ncd"),),
        )
        chunks = chunk_document(
            document, ChunkConfig(strategy=ChunkStrategy.PARAGRAPH, chunk_size=5)
        )
        self.assertEqual([chunk.text for chunk in chunks], ["ab", "cd"])
        self.assertEqual([chunk.page_number for chunk in chunks], [1, 1])
        self.assertEqual([chunk.position for chunk in chunks], [0, 1])
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            ["paper.pdf-page1-pos0", "paper.pdf-page1-pos1"],
        )

    # @unittest.skip("TODO: validate overlap, then remove this decorator")
    def test_overlap_must_be_smaller_than_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            split_chars_with_overlap("abc", chunk_size=3, overlap=3)
    def test_build_chunk_id_is_traceable(self) -> None:
        self.assertEqual(
            _build_chunk_id("paper.pdf", page_number=3, position=2),
            "paper.pdf-page3-pos2",
        )

    def test_build_chunk_id_rejects_blank_filename(self) -> None:
        with self.assertRaises(ValueError):
            _build_chunk_id("   ", page_number=1, position=0)

    def test_build_chunk_id_rejects_invalid_page_number(self) -> None:
        with self.assertRaises(ValueError):
            _build_chunk_id("paper.pdf", page_number=0, position=0)

    def test_build_chunk_id_rejects_negative_position(self) -> None:
        with self.assertRaises(ValueError):
            _build_chunk_id("paper.pdf", page_number=1, position=-1)


if __name__ == "__main__":
    unittest.main()
