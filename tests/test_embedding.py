import importlib.util
import unittest
from collections.abc import Sequence


class DeterministicFakeEmbedding:
    """Offline fake whose vectors depend only on the input text."""

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (float(len(text)), float(sum(ord(character) for character in text)))
            for text in texts
        )


class EmbeddingInterfaceTests(unittest.TestCase):
    def test_fake_embedding_satisfies_protocol_and_is_deterministic(self) -> None:
        module_spec = importlib.util.find_spec("app.embedding")
        self.assertIsNotNone(module_spec, "app.embedding must define the interface")

        from app.embedding import EmbeddingModel

        model = DeterministicFakeEmbedding()

        self.assertIsInstance(model, EmbeddingModel)
        self.assertEqual(
            model.embed(("a", "bb")),
            ((1.0, 97.0), (2.0, 196.0)),
        )


# TODO(user): add a test proving that embed(()) returns an empty tuple.
    def test_fake_embedding_returns_empty_tuple_for_empty_input(self) -> None:
        model = DeterministicFakeEmbedding()
        result = model.embed(())
        self.assertEqual(result, ())

if __name__ == "__main__":
    unittest.main()
