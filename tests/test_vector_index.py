import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from app.chunking import Chunk, ChunkConfig, ChunkStrategy
from app.vector_index import (
    build_vector_index,
    load_vector_index,
    save_vector_index,
    search_vector_index,
)


class FakeEmbedding:
    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors = {
            "query": (1.0, 0.0),
            "best": (1.0, 0.0),
            "second": (0.0, 1.0),
        }
        return tuple(vectors[text] for text in texts)


class VectorIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = (
            Chunk("c1", "second", 2, ChunkStrategy.CHAR_OVERLAP, 0),
            Chunk("c2", "best", 1, ChunkStrategy.CHAR_OVERLAP, 0),
        )
        self.chunk_config = ChunkConfig(
            strategy=ChunkStrategy.CHAR_OVERLAP,
            chunk_size=800,
            overlap=120,
        )

    def test_build_save_and_load_preserve_vectors_and_config(self) -> None:
        index = build_vector_index(
            self.chunks,
            FakeEmbedding(),
            source_filename="paper.pdf",
            model_name="fake-model",
            normalize_embeddings=True,
            chunk_config=self.chunk_config,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_vector_index(index, Path(directory) / "index.json")
            loaded = load_vector_index(path)

        self.assertEqual(loaded, index)
        self.assertEqual(loaded.config.embedding_dimension, 2)

    def test_search_uses_precomputed_chunk_vectors(self) -> None:
        index = build_vector_index(
            self.chunks,
            FakeEmbedding(),
            source_filename="paper.pdf",
            model_name="fake-model",
            normalize_embeddings=True,
            chunk_config=self.chunk_config,
        )

        results = search_vector_index("query", index, FakeEmbedding(), top_k=2)

        self.assertEqual([result.chunk_id for result in results], ["c2", "c1"])
        self.assertEqual([result.page_number for result in results], [1, 2])


if __name__ == "__main__":
    unittest.main()
