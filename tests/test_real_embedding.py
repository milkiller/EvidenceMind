import unittest
from collections.abc import Sequence

from app.embedding import EmbeddingModel, SentenceTransformerEmbedding


class FakeSentenceEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def get_embedding_dimension(self) -> int:
        return 2

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Sequence[Sequence[float]]:
        self.calls += 1
        if not normalize_embeddings or not convert_to_numpy or show_progress_bar:
            raise AssertionError("adapter must request normalized NumPy vectors")
        return [(float(len(text)), 1.0) for text in sentences]


class SentenceTransformerEmbeddingTests(unittest.TestCase):
    def test_adapter_satisfies_protocol_and_preserves_batch_order(self) -> None:
        backend = FakeSentenceEncoder()
        model = SentenceTransformerEmbedding("fake-model", backend=backend)

        self.assertIsInstance(model, EmbeddingModel)
        self.assertEqual(model.embed(("a", "bbb")), ((1.0, 1.0), (3.0, 1.0)))
        self.assertEqual(model.dimension, 2)
        self.assertEqual(backend.calls, 1)

    def test_empty_batch_does_not_call_backend(self) -> None:
        backend = FakeSentenceEncoder()
        model = SentenceTransformerEmbedding("fake-model", backend=backend)

        self.assertEqual(model.embed(()), ())
        self.assertEqual(backend.calls, 0)


if __name__ == "__main__":
    unittest.main()
