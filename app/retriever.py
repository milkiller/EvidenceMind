"""In-memory similarity retrieval contracts for EvidenceMind."""

from collections.abc import Sequence
from dataclasses import dataclass
import math

from .chunking import Chunk
from .embedding import EmbeddingModel, EmbeddingVector


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved Chunk plus its query similarity score."""

    chunk_id: str
    page_number: int
    text: str
    similarity: float


def cosine_similarity(
    left: EmbeddingVector,  #一般是用户问题的向量
    right: EmbeddingVector, #某个chunk的向量
) -> float:
    """Return the cosine similarity between two equal-dimensional vectors.

    TODO(user):
    1. Reject vectors with different lengths.
    2. Compute the dot product and the two Euclidean norms.
    3. Reject a zero norm before dividing.
    4. Return dot(left, right) / (norm(left) * norm(right)).
    """
    if len(left) != len(right):
        raise ValueError("Vectors must have the same length")
    if not left or not right:
        raise ValueError("Vectors must not be empty")
    dot_product = sum(
        l * r for l, r in zip(left, right)
    )
    left_norm = math.sqrt(sum(l ** 2 for l in left))
    right_norm = math.sqrt(sum(r ** 2 for r in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Vectors must not be zero vectors")
    return dot_product / (left_norm * right_norm)
    raise NotImplementedError("TODO(user): implement cosine similarity")


def retrieve_top_k(
    query: str,
    chunks: Sequence[Chunk],
    embedding_model: EmbeddingModel,
    top_k: int,
) -> tuple[SearchResult, ...]:
    """Return the highest-scoring chunks for a query.

    TODO(user):
    1. Validate non-blank query and top_k > 0.
    2. Embed the query once and all Chunk.text values as one batch.
    3. Pair each score with the original Chunk metadata.
    4. Sort by similarity descending and keep the first top_k results.
    5. Preserve original order when scores tie.
    """
    if not isinstance(chunks, Sequence):
        raise ValueError("Chunks must be a sequence")
    if not query.strip():
        raise ValueError("Query must be non-blank")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if not chunks:
        return ()

    texts = [query] +[chunk.text for chunk in chunks]
    vectors = embedding_model.embed(texts)
    if len(vectors) != len(texts):
        raise ValueError("Embedding model must return one vector per input text")
    query_vector = vectors[0]
    chunk_vectors = vectors[1:]
    results = []
    for chunk , chunk_vector in zip(chunks , chunk_vectors):
        similarity = cosine_similarity(query_vector, chunk_vector)
        results.append((chunk, similarity))
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:top_k]
    return tuple(
        SearchResult(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            text=chunk.text,
            similarity=similarity,
        )
        for chunk, similarity in top_results
    )
    raise NotImplementedError("TODO(user): implement Top-K retrieval")
