"""Optional second-stage ranking for a bounded set of retrieved passages."""

from collections.abc import Sequence
import math
from typing import Protocol

from .kb_index import KnowledgeSearchResult


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class PassageReranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        """Return one relevance score per passage, in input order."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, *, offline: bool = False) -> None:
        """Load a public reranker locally; offline mode must never fetch weights."""
        if not model_name.strip():
            raise ValueError("reranker model name must not be blank")
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                model_name, device="cpu", local_files_only=offline, trust_remote_code=False,
            )
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"无法加载重排模型 {model_name}；检查本地缓存，或取消 --offline 下载公开模型。"
            ) from exc

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        if not passages:
            return ()
        pairs = [(query, passage) for passage in passages]
        return tuple(float(value) for value in self._model.predict(pairs, batch_size=16))


def rerank_candidates(
    query: str,
    candidates: Sequence[KnowledgeSearchResult],
    reranker: PassageReranker,
    *,
    top_k: int,
) -> tuple[KnowledgeSearchResult, ...]:
    """Sort dense candidates by cross-encoder score without changing cosine metadata."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not candidates:
        return ()
    scores = tuple(reranker.score(query, tuple(item.text for item in candidates)))
    if len(scores) != len(candidates) or any(not math.isfinite(score) for score in scores):
        raise ValueError("reranker returned invalid scores")
    ranked = sorted(
        enumerate(candidates), key=lambda item: (-scores[item[0]], item[0]),
    )
    return tuple(item for _, item in ranked[:top_k])
