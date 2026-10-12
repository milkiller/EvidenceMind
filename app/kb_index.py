"""Build and search a source-aware local vector index."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from .chunking import split_chars_with_overlap
from .embedding import EmbeddingModel, EmbeddingVector
from .kb_sources import KnowledgeSource
from .retriever import SearchResult, cosine_similarity


MULTILINGUAL_MODEL_NAME = "intfloat/multilingual-e5-small"


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    chunk_id: str
    source_id: str
    kind: str
    title: str
    origin: str
    version: str | None
    is_current: bool
    locator: str
    text: str
    vector: EmbeddingVector


@dataclass(frozen=True, slots=True)
class KnowledgeIndex:
    model_name: str
    embedding_dimension: int
    chunk_size: int
    overlap: int
    chunks: tuple[KnowledgeChunk, ...]
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class KnowledgeSearchResult(SearchResult):
    source_title: str
    source_kind: str
    locator: str
    version: str | None


def build_knowledge_index(
    sources: Sequence[KnowledgeSource],
    embedding_model: EmbeddingModel,
    *,
    model_name: str,
    chunk_size: int = 800,
    overlap: int = 120,
    batch_size: int = 32,
) -> KnowledgeIndex:
    """Split source sections and embed them in bounded batches."""
    if not sources or not model_name.strip():
        raise ValueError("sources and model_name must not be empty")
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size or batch_size <= 0:
        raise ValueError("invalid chunk or batch size")

    pending: list[tuple[KnowledgeSource, str, str, str]] = []
    for source in sources:
        for section in source.sections:
            pieces = split_chars_with_overlap(section.text, chunk_size, overlap)
            for number, piece in enumerate(pieces, start=1):
                if not piece.strip():
                    continue
                locator = section.locator
                if len(pieces) > 1:
                    locator += f" · 片段 {number}"
                digest = sha256(
                    f"{source.source_id}\0{locator}\0{piece}".encode("utf-8")
                ).hexdigest()[:16]
                pending.append((source, locator, piece, f"{source.source_id}:{digest}"))
    if not pending:
        raise ValueError("no extractable text was found")

    chunks: list[KnowledgeChunk] = []
    dimension = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedding_model.embed(
            tuple(_model_input(item[2], model_name, "passage") for item in batch)
        )
        if len(vectors) != len(batch):
            raise ValueError("embedding model returned the wrong batch size")
        for (source, locator, piece, chunk_id), vector in zip(batch, vectors):
            if not vector or (dimension and len(vector) != dimension):
                raise ValueError("embedding model returned inconsistent dimensions")
            dimension = len(vector)
            chunks.append(KnowledgeChunk(
                chunk_id, source.source_id, source.kind, source.title, source.origin,
                source.version, source.is_current, locator, piece, vector,
            ))
    return KnowledgeIndex(model_name, dimension, chunk_size, overlap, tuple(chunks))


def save_knowledge_index(index: KnowledgeIndex, path: str | Path) -> Path:
    """Save an index containing private source text to a local-only path."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(index), ensure_ascii=False), encoding="utf-8")
    return target


def load_knowledge_index(path: str | Path) -> KnowledgeIndex:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported knowledge index schema version")
    chunks = tuple(
        KnowledgeChunk(
            item["chunk_id"], item["source_id"], item["kind"], item["title"],
            item["origin"], item["version"], item["is_current"], item["locator"],
            item["text"], tuple(float(value) for value in item["vector"]),
        )
        for item in payload["chunks"]
    )
    dimension = payload["embedding_dimension"]
    if dimension <= 0 or any(len(chunk.vector) != dimension for chunk in chunks):
        raise ValueError("stored vector dimension does not match index")
    return KnowledgeIndex(
        payload["model_name"], dimension, payload["chunk_size"],
        payload["overlap"], chunks, payload["schema_version"],
    )


def search_knowledge_index(
    query: str,
    index: KnowledgeIndex,
    embedding_model: EmbeddingModel,
    *,
    top_k: int = 3,
    include_history: bool = False,
    scope: str = "research",
    paper_versions: Sequence[str] | None = None,
    source_title: str | None = None,
) -> tuple[KnowledgeSearchResult, ...]:
    """Rank eligible chunks; old paper revisions require explicit opt-in."""
    if not query.strip() or top_k <= 0:
        raise ValueError("query must be non-blank and top_k positive")
    if scope not in {"research", "paper", "literature", "literature_pdf", "all"}:
        raise ValueError("invalid scope")
    versions = set(paper_versions) if paper_versions is not None else None
    if versions is not None and (not versions or any(not version.strip() for version in versions)):
        raise ValueError("paper_versions must contain non-blank versions")
    include_history = include_history or versions is not None
    query_vectors = embedding_model.embed((_model_input(query, index.model_name, "query"),))
    if len(query_vectors) != 1 or len(query_vectors[0]) != index.embedding_dimension:
        raise ValueError("query vector dimension does not match index")
    eligible = (
        chunk for chunk in index.chunks
        if (include_history or chunk.kind != "paper" or chunk.is_current)
        and _in_scope(chunk, scope)
        and (versions is None or chunk.kind != "paper" or chunk.version in versions)
        and (source_title is None or source_title.casefold() in chunk.title.casefold())
    )
    ranked = sorted(
        ((chunk, cosine_similarity(query_vectors[0], chunk.vector)) for chunk in eligible),
        key=lambda item: item[1], reverse=True,
    )
    scored: list[tuple[KnowledgeChunk, float]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for chunk, score in ranked:
        signature = (chunk.kind, chunk.title.casefold(), chunk.locator, chunk.text.strip())
        if signature not in seen:
            seen.add(signature)
            scored.append((chunk, score))
    selected = scored[:top_k]
    if scope == "research" and not include_history and top_k >= 2 and scored:
        current = _representative_paper(scored, current=True)
        if current is not None and current[1] >= scored[0][1] - 0.08:
            if all(item[0].chunk_id != current[0].chunk_id for item in selected):
                selected = selected[:-1] + [current]
                selected.sort(key=lambda item: item[1], reverse=True)
    if include_history and top_k >= 2 and scope in {"paper", "research", "all"}:
        # A revision question needs both sides of the comparison in context.
        current = _representative_paper(scored, current=True)
        old = _representative_paper(scored, current=False)
        if current is not None and old is not None:
            reserved = [current, old]
            reserved_ids = {item[0].chunk_id for item in reserved}
            selected = reserved + [
                item for item in scored if item[0].chunk_id not in reserved_ids
            ][:top_k - 2]
            selected.sort(key=lambda item: item[1], reverse=True)
    return tuple(
        KnowledgeSearchResult(
            chunk_id=chunk.chunk_id,
            page_number=_page_number(chunk.locator),
            text=chunk.text,
            similarity=score,
            source_title=chunk.title,
            source_kind=chunk.kind,
            locator=chunk.locator,
            version=chunk.version,
        )
        for chunk, score in selected
    )


def _page_number(locator: str) -> int:
    if locator.startswith("第 ") and " 页" in locator:
        return int(locator.split()[1])
    return 0


def _in_scope(chunk: KnowledgeChunk, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "paper":
        return chunk.kind == "paper"
    if scope == "literature":
        return chunk.kind in {"zotero", "zotero_pdf"}
    if scope == "literature_pdf":
        return chunk.kind == "zotero_pdf"
    if chunk.kind != "obsidian":
        return chunk.kind != "obsidian_pdf"
    normalized = chunk.title.replace("\\", "/")
    return normalized.startswith(("odor/", "中期检查/"))


def _model_input(text: str, model_name: str, role: str) -> str:
    """Use the retrieval prefixes required by multilingual E5."""
    if model_name == MULTILINGUAL_MODEL_NAME:
        return f"{role}: {text}"
    return text


def _representative_paper(
    scored: Sequence[tuple[KnowledgeChunk, float]], *, current: bool
) -> tuple[KnowledgeChunk, float] | None:
    matches = [
        item for item in scored
        if item[0].kind == "paper" and item[0].is_current == current
    ]
    return next((item for item in matches if len(item[0].text) >= 200), None) or (
        matches[0] if matches else None
    )
