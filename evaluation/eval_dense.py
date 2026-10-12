"""Evaluate Dense Retrieval against manually labeled relevant pages."""

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from app.embedding import EmbeddingModel, SentenceTransformerEmbedding
from app.retriever import SearchResult
from app.vector_index import load_vector_index, search_vector_index


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One question and the source pages that can directly answer it."""

    case_id: str
    question: str
    expected_pages: frozenset[int]


@dataclass(frozen=True, slots=True)
class CaseScore:
    """Retrieval result and page-level metrics for one evaluation question."""

    case_id: str
    expected_pages: tuple[int, ...]
    retrieved_pages: tuple[int, ...]
    hit_at_k: bool
    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate metrics for one fixed index and evaluation set."""

    scores: tuple[CaseScore, ...]
    hit_rate_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float


Retriever = Callable[[str], Sequence[SearchResult]]


def load_eval_cases(eval_path: str | Path) -> tuple[EvalCase, ...]:
    """Load only fully labeled evaluation cases and validate their page labels."""
    path = Path(eval_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported evaluation schema version")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("evaluation file must contain at least one case")

    cases: list[EvalCase] = []
    for item in raw_cases:
        if not isinstance(item, Mapping):
            raise ValueError("every evaluation case must be an object")
        case_id = item.get("id")
        question = item.get("question")
        expected_pages = item.get("expected_pages")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("evaluation case id must be non-blank")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"evaluation question must be non-blank: {case_id}")
        if (
            not isinstance(expected_pages, list)
            or not expected_pages
            or not all(isinstance(page, int) and page > 0 for page in expected_pages)
        ):
            raise ValueError(f"evaluation case needs positive expected_pages: {case_id}")
        cases.append(
            EvalCase(
                case_id=case_id,
                question=question,
                expected_pages=frozenset(expected_pages),
            )
        )

    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("evaluation case ids must be unique")
    return tuple(cases)


def evaluate_cases(cases: Sequence[EvalCase], retrieve: Retriever) -> EvaluationSummary:
    """Score page recall, hit rate, and first relevant result rank for each case."""
    if not cases:
        raise ValueError("at least one evaluation case is required")

    scores: list[CaseScore] = []
    for case in cases:
        results = tuple(retrieve(case.question))
        retrieved_pages = tuple(result.page_number for result in results)
        retrieved_page_set = set(retrieved_pages)
        matching_ranks = tuple(
            rank
            for rank, page_number in enumerate(retrieved_pages, start=1)
            if page_number in case.expected_pages
        )
        relevant_page_count = len(retrieved_page_set & case.expected_pages)
        scores.append(
            CaseScore(
                case_id=case.case_id,
                expected_pages=tuple(sorted(case.expected_pages)),
                retrieved_pages=retrieved_pages,
                hit_at_k=bool(matching_ranks),
                recall_at_k=relevant_page_count / len(case.expected_pages),
                reciprocal_rank=1 / matching_ranks[0] if matching_ranks else 0.0,
            )
        )

    case_count = len(scores)
    return EvaluationSummary(
        scores=tuple(scores),
        hit_rate_at_k=sum(score.hit_at_k for score in scores) / case_count,
        recall_at_k=sum(score.recall_at_k for score in scores) / case_count,
        mean_reciprocal_rank=sum(score.reciprocal_rank for score in scores) / case_count,
    )


def format_summary(summary: EvaluationSummary, top_k: int) -> str:
    """Render inspectable per-case results and aggregate Dense Retrieval metrics."""
    lines = [
        f"评测题数：{len(summary.scores)}",
        f"Hit@{top_k}：{summary.hit_rate_at_k:.3f}",
        f"Recall@{top_k}：{summary.recall_at_k:.3f}",
        f"MRR：{summary.mean_reciprocal_rank:.3f}",
        "",
        "逐题结果：",
    ]
    for score in summary.scores:
        lines.append(
            " | ".join(
                (
                    score.case_id,
                    f"期望页 {list(score.expected_pages)}",
                    f"检索页 {list(score.retrieved_pages)}",
                    f"命中 {score.hit_at_k}",
                    f"Recall {score.recall_at_k:.3f}",
                    f"RR {score.reciprocal_rank:.3f}",
                )
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract for one local Dense Retrieval evaluation."""
    parser = argparse.ArgumentParser(description="运行带页码标签的 Dense Retrieval 离线评测。")
    parser.add_argument("--eval-path", default="evaluation/dense_eval.json")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load one index once and evaluate all labeled questions without an LLM call."""
    args = build_parser().parse_args(argv)
    cases = load_eval_cases(args.eval_path)
    eval_payload = json.loads(Path(args.eval_path).read_text(encoding="utf-8"))
    index_path = args.index_path or eval_payload.get("index_path")
    if not isinstance(index_path, str) or not index_path.strip():
        raise SystemExit("provide --index-path or index_path in the evaluation file")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than 0")

    try:
        index = load_vector_index(index_path)
        embedding_model = SentenceTransformerEmbedding(
            model_name=index.config.model_name,
            device=args.device,
            normalize_embeddings=index.config.normalize_embeddings,
            local_files_only=args.offline,
        )
        summary = evaluate_cases(
            cases,
            lambda question: search_vector_index(question, index, embedding_model, args.top_k),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(format_summary(summary, args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
