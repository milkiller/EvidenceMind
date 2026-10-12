import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from app.chunking import ChunkStrategy
from app.retrieval_cli import build_parser, format_search_results, main
from app.retriever import SearchResult
from app.vector_index import VectorIndex, VectorIndexConfig


class RetrievalCliTests(unittest.TestCase):
    def test_parser_accepts_a_similarity_threshold_in_cosine_range(self) -> None:
        args = build_parser().parse_args(
            ("index.json", "测试问题", "--min-similarity", "0.5")
        )

        self.assertEqual(args.min_similarity, 0.5)

        for invalid_value in ("-1.01", "1.01"):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(
                        (
                            "index.json",
                            "测试问题",
                            "--min-similarity",
                            invalid_value,
                        )
                    )

    def test_format_search_results_keeps_rank_and_evidence_metadata(self) -> None:
        results = (
            SearchResult("c1", 7, "证据原文", 0.87654),
            SearchResult("c2", 3, "第二段", 0.5),
        )

        output = format_search_results(results)

        self.assertIn("[1] 页码：7 | 相似度：0.8765 | Chunk：c1", output)
        self.assertIn("证据原文", output)
        self.assertIn("[2] 页码：3 | 相似度：0.5000 | Chunk：c2", output)

    @patch("app.retrieval_cli.search_vector_index")
    @patch("app.retrieval_cli.SentenceTransformerEmbedding")
    @patch("app.retrieval_cli.load_vector_index")
    def test_main_uses_the_model_configuration_saved_in_the_index(
        self,
        load_index,
        embedding_type,
        search_index,
    ) -> None:
        index = VectorIndex(
            config=VectorIndexConfig(
                source_filename="paper.pdf",
                model_name="saved-model",
                normalize_embeddings=True,
                embedding_dimension=2,
                chunk_strategy=ChunkStrategy.CHAR_OVERLAP,
                chunk_size=800,
                overlap=120,
            ),
            chunks=(),
        )
        result = SearchResult("c1", 4, "命中原文", 0.75)
        load_index.return_value = index
        search_index.return_value = (result,)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                ("index.json", "测试问题", "--top-k", "1", "--offline")
            )

        self.assertEqual(exit_code, 0)
        embedding_type.assert_called_once_with(
            model_name="saved-model",
            device="cpu",
            normalize_embeddings=True,
            local_files_only=True,
        )
        search_index.assert_called_once_with(
            "测试问题",
            index,
            embedding_type.return_value,
            top_k=1,
        )
        self.assertIn("模型：saved-model", stdout.getvalue())
        self.assertIn("页码：4", stdout.getvalue())

    @patch("app.retrieval_cli.search_vector_index")
    @patch("app.retrieval_cli.SentenceTransformerEmbedding")
    @patch("app.retrieval_cli.load_vector_index")
    def test_main_filters_results_and_keeps_the_threshold_boundary(
        self,
        load_index,
        embedding_type,
        search_index,
    ) -> None:
        load_index.return_value = VectorIndex(
            config=VectorIndexConfig(
                source_filename="paper.pdf",
                model_name="saved-model",
                normalize_embeddings=True,
                embedding_dimension=2,
                chunk_strategy=ChunkStrategy.CHAR_OVERLAP,
                chunk_size=800,
                overlap=120,
            ),
            chunks=(),
        )
        search_index.return_value = (
            SearchResult("high", 1, "高分证据", 0.8),
            SearchResult("boundary", 2, "边界证据", 0.75),
            SearchResult("low", 3, "低分证据", 0.4),
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            main(
                (
                    "index.json",
                    "测试问题",
                    "--min-similarity",
                    "0.75",
                )
            )

        output = stdout.getvalue()
        self.assertIn("高分证据", output)
        self.assertIn("边界证据", output)
        self.assertNotIn("低分证据", output)

    @patch("app.retrieval_cli.search_vector_index")
    @patch("app.retrieval_cli.SentenceTransformerEmbedding")
    @patch("app.retrieval_cli.load_vector_index")
    def test_main_explains_when_no_result_reaches_the_threshold(
        self,
        load_index,
        embedding_type,
        search_index,
    ) -> None:
        load_index.return_value = VectorIndex(
            config=VectorIndexConfig(
                source_filename="paper.pdf",
                model_name="saved-model",
                normalize_embeddings=True,
                embedding_dimension=2,
                chunk_strategy=ChunkStrategy.CHAR_OVERLAP,
                chunk_size=800,
                overlap=120,
            ),
            chunks=(),
        )
        search_index.return_value = (SearchResult("low", 3, "低分证据", 0.4),)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            main(
                (
                    "index.json",
                    "测试问题",
                    "--min-similarity",
                    "0.9",
                )
            )

        output = stdout.getvalue()
        self.assertIn("没有达到相似度阈值的检索结果", output)
        self.assertNotIn("低分证据", output)


if __name__ == "__main__":
    unittest.main()
