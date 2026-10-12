import importlib
import importlib.util
import inspect
import unittest
from collections.abc import Sequence

from app.chunking import Chunk, ChunkStrategy
from app.embedding import EmbeddingModel


class RetrieverContractTests(unittest.TestCase):
    def test_retriever_exposes_result_model_and_function_signatures(self) -> None:
        module_spec = importlib.util.find_spec("app.retriever")
        self.assertIsNotNone(module_spec, "app.retriever must define the API")

        retriever = importlib.import_module("app.retriever")

        self.assertEqual(
            tuple(retriever.SearchResult.__dataclass_fields__),
            ("chunk_id", "page_number", "text", "similarity"),
        )
        self.assertEqual(
            tuple(inspect.signature(retriever.cosine_similarity).parameters),
            ("left", "right"),
        )
        self.assertEqual(
            tuple(inspect.signature(retriever.retrieve_top_k).parameters),
            ("query", "chunks", "embedding_model", "top_k"),
        )


class CosineSimilarityTests(unittest.TestCase):
    def test_cosine_similarity_for_same_direction_is_one(self) -> None:
        from app.retriever import cosine_similarity

        self.assertAlmostEqual(cosine_similarity((1.0, 2.0), (2.0, 4.0)), 1.0)

    def test_cosine_similarity_rejects_zero_vector(self) -> None:
        from app.retriever import cosine_similarity

        with self.assertRaises(ValueError):
            cosine_similarity((0.0, 0.0), (1.0, 0.0))

    def test_cosine_similarity_rejects_different_dimensions(self) -> None:
        from app.retriever import cosine_similarity

        with self.assertRaises(ValueError):
            cosine_similarity((1.0, 2.0), (1.0,))

class TopKRetrieverTests(unittest.TestCase):
    def test_retrieve_top_k_preserves_chunk_metadata_and_orders_scores(self) -> None:
        from app.retriever import retrieve_top_k

        class FakeEmbedding:
            def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
                vectors = {
                    "query": (1.0, 0.0),
                    "best": (1.0, 0.0),
                    "second": (0.0, 1.0),
                }
                return tuple(vectors[text] for text in texts)

        chunks = (
            Chunk("c1", "second", 2, ChunkStrategy.FIXED_CHAR, 0),
            Chunk("c2", "best", 1, ChunkStrategy.FIXED_CHAR, 0),
        )
        embedding_model: EmbeddingModel = FakeEmbedding()

        results = retrieve_top_k("query", chunks, embedding_model, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c2")
        self.assertEqual(results[0].page_number, 1)
        self.assertEqual(results[0].text, "best")
        self.assertAlmostEqual(results[0].similarity, 1.0)

    def test_retrieve_top_k_rejects_non_positive_top_k(self) -> None:
        from app.retriever import retrieve_top_k

        with self.assertRaises(ValueError):
            retrieve_top_k("query", (), _UnusedEmbedding(), top_k=0)

    def test_retrieve_top_k_rejects_blank_query(self) -> None:
        from app.retriever import retrieve_top_k

        with self.assertRaises(ValueError):
            retrieve_top_k("   ", (), _UnusedEmbedding(), top_k=1)

    def test_retrieve_top_k_preserves_original_order_when_scores_tie(self) -> None:
        """Equal scores should keep the input Chunk order."""
        from app.retriever import retrieve_top_k

        class TiedEmbedding:
            def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
                return tuple((1.0, 0.0) for _ in texts)

        chunks = (
            Chunk("c1", "first", 1, ChunkStrategy.FIXED_CHAR, 0),
            Chunk("c2", "second", 2, ChunkStrategy.FIXED_CHAR, 0),
        )

        results = retrieve_top_k("query", chunks, TiedEmbedding(), top_k=2)

        self.assertEqual(
            [result.chunk_id for result in results],
            ["c1", "c2"],
        )

    def test_retrieve_top_k_rejects_embedding_count_mismatch(self) -> None:
        """The model must return one vector for every input text."""
        from app.retriever import retrieve_top_k

        class MissingVectorEmbedding:
            def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
                return ((1.0, 0.0),)

        chunks = (
            Chunk("c1", "only chunk", 1, ChunkStrategy.FIXED_CHAR, 0),
        )

        with self.assertRaisesRegex(ValueError, "one vector per input text"):
            retrieve_top_k(
                "query",
                chunks,
                MissingVectorEmbedding(),
                top_k=1,
            )

    def test_retrieve_top_k_returns_all_chunks_when_top_k_is_larger(self) -> None:
        """Requesting more results than available should return every Chunk."""
        from app.retriever import retrieve_top_k

        class FakeEmbedding:
            def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
                vectors = {
                    "query": (1.0, 0.0),
                    "best": (1.0, 0.0),
                    "second": (0.0, 1.0),
                }
                return tuple(vectors[text] for text in texts)

        chunks = (
            Chunk("c1", "second", 2, ChunkStrategy.FIXED_CHAR, 0),
            Chunk("c2", "best", 1, ChunkStrategy.FIXED_CHAR, 0),
        )

        results = retrieve_top_k("query", chunks, FakeEmbedding(), top_k=5)

        self.assertEqual(len(results), 2)
        self.assertEqual(
            [result.chunk_id for result in results],
            ["c2", "c1"],
        )


class _UnusedEmbedding:
    """Embedding dependency for validation paths that should return early."""

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        raise AssertionError("embedding must not run before input validation")


if __name__ == "__main__":
    unittest.main()
