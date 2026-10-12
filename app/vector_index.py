"""Build, persist, load, and search a transparent single-document vector index."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .chunking import Chunk, ChunkConfig, ChunkStrategy
from .embedding import EmbeddingModel, EmbeddingVector
from .retriever import SearchResult, cosine_similarity


@dataclass(frozen=True, slots=True)
class VectorIndexConfig:
    """Configuration required to reproduce a single-document index."""

    source_filename: str
    model_name: str
    normalize_embeddings: bool
    embedding_dimension: int
    chunk_strategy: ChunkStrategy
    chunk_size: int
    overlap: int
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """Chunk metadata paired with its precomputed embedding."""

    chunk_id: str
    page_number: int
    position: int
    text: str
    vector: EmbeddingVector


@dataclass(frozen=True, slots=True)
class VectorIndex:
    """A serializable vector index for one source document."""

    config: VectorIndexConfig
    chunks: tuple[IndexedChunk, ...]


def build_vector_index(
    chunks: Sequence[Chunk],
    embedding_model: EmbeddingModel,
    *,
    source_filename: str,
    model_name: str,
    normalize_embeddings: bool,
    chunk_config: ChunkConfig,
) -> VectorIndex:
    """Embed every Chunk once and return a reproducible in-memory index."""
    if not source_filename.strip():
        raise ValueError("source_filename must be non-blank")
    if not model_name.strip():
        raise ValueError("model_name must be non-blank")
    if not chunks:
        raise ValueError("at least one Chunk is required to build an index")

    vectors = embedding_model.embed(tuple(chunk.text for chunk in chunks))
    if len(vectors) != len(chunks):
        raise ValueError("embedding model must return one vector per Chunk")

    dimension = len(vectors[0])
    if dimension <= 0:
        raise ValueError("embedding vectors must not be empty")
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding vectors must share one dimension")

    indexed_chunks = tuple(
        IndexedChunk(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            position=chunk.position,
            text=chunk.text,
            vector=vector,
        )
        for chunk, vector in zip(chunks, vectors)
    )
    config = VectorIndexConfig(
        source_filename=source_filename,
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        embedding_dimension=dimension,
        chunk_strategy=chunk_config.strategy,
        chunk_size=chunk_config.chunk_size,
        overlap=chunk_config.overlap,
    )
    return VectorIndex(config=config, chunks=indexed_chunks)


def save_vector_index(index: VectorIndex, output_path: str | Path) -> Path:
    """Persist index vectors and reproducibility metadata as readable JSON."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    config_data = asdict(index.config)
    config_data["chunk_strategy"] = index.config.chunk_strategy.value
    payload = {
        "config": config_data,
        "chunks": [asdict(chunk) for chunk in index.chunks],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_vector_index(index_path: str | Path) -> VectorIndex:
    """Load an index written by save_vector_index and validate its schema."""
    path = Path(index_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    config_data = payload["config"]
    if config_data.get("schema_version") != 1:
        raise ValueError("unsupported vector index schema version")

    config = VectorIndexConfig(
        source_filename=config_data["source_filename"],
        model_name=config_data["model_name"],
        normalize_embeddings=config_data["normalize_embeddings"],
        embedding_dimension=config_data["embedding_dimension"],
        chunk_strategy=ChunkStrategy(config_data["chunk_strategy"]),
        chunk_size=config_data["chunk_size"],
        overlap=config_data["overlap"],
        schema_version=config_data["schema_version"],
    )
    chunks = tuple(
        IndexedChunk(
            chunk_id=item["chunk_id"],
            page_number=item["page_number"],
            position=item["position"],
            text=item["text"],
            vector=tuple(float(value) for value in item["vector"]),
        )
        for item in payload["chunks"]
    )
    if any(len(chunk.vector) != config.embedding_dimension for chunk in chunks):
        raise ValueError("stored vector dimension does not match index config")
    return VectorIndex(config=config, chunks=chunks)


def search_vector_index(
    query: str,
    index: VectorIndex,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> tuple[SearchResult, ...]:
    """Embed one query and rank the precomputed document vectors."""
    if not query.strip():
        raise ValueError("query must be non-blank")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if not index.chunks:
        return ()

    query_vectors = embedding_model.embed((query,))
    if len(query_vectors) != 1:
        raise ValueError("embedding model must return one query vector")
    query_vector = query_vectors[0]
    if len(query_vector) != index.config.embedding_dimension:
        raise ValueError("query vector dimension does not match the index")

    scored_chunks = [
        (chunk, cosine_similarity(query_vector, chunk.vector))
        for chunk in index.chunks
    ]
    scored_chunks.sort(key=lambda item: item[1], reverse=True)
    return tuple(
        SearchResult(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            text=chunk.text,
            similarity=similarity,
        )
        for chunk, similarity in scored_chunks[:top_k]
    )
