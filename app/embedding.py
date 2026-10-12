"""Injectable embedding contracts and a local Sentence Transformers adapter."""

from collections.abc import Sequence
from typing import Protocol, TypeAlias, runtime_checkable


EmbeddingVector: TypeAlias = tuple[float, ...]
EmbeddingBatch: TypeAlias = tuple[EmbeddingVector, ...]


@runtime_checkable
class EmbeddingModel(Protocol):
    """Convert an ordered text batch into an equally ordered vector batch."""

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Return one embedding vector for each input text, preserving order."""
        ...


class SentenceEncoderBackend(Protocol):
    """Subset of SentenceTransformer used by the production adapter."""

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Sequence[Sequence[float]]:
        """Encode an ordered batch of sentences."""
        ...

    def get_embedding_dimension(self) -> int | None:
        """Return the model output dimension when known."""
        ...


DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


class SentenceTransformerEmbedding:
    """Run a local Sentence Transformers model behind the project protocol."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        normalize_embeddings: bool = True,
        local_files_only: bool = False,
        backend: SentenceEncoderBackend | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-blank")

        if backend is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for the local embedding model"
                ) from exc
            backend = SentenceTransformer(
                model_name,
                device=device,
                local_files_only=local_files_only,
            )

        dimension = backend.get_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise ValueError("embedding backend must expose a positive dimension")

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.dimension = dimension
        self._backend = backend

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Encode texts locally and validate the project's batch contract."""
        ordered_texts = list(texts)
        if not ordered_texts:
            return ()

        raw_vectors = self._backend.encode(
            ordered_texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = tuple(
            tuple(float(value) for value in vector) for vector in raw_vectors
        )
        if len(vectors) != len(ordered_texts):
            raise ValueError("embedding backend must return one vector per input text")
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError("embedding backend returned an unexpected vector dimension")
        return vectors
