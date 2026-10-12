"""Run a fixed, source-aware local KB evaluation without changing the index."""

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from app.answering import answer_question
from app.agent_tools import resolve_zotero_pdf_title
from app.embedding import SentenceTransformerEmbedding
from app.gateway_generator import GatewayChatCompletionGenerator
from app.kb_index import load_knowledge_index, search_knowledge_index
from app.reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL, rerank_candidates


def main(argv: Sequence[str] | None = None) -> int:
    """Record retrieval and optional generated answers for ten reviewed questions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="data/kb-eval-10.json")
    parser.add_argument("--output", default="data/kb-eval-results.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.5)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--title-aware", action="store_true",
                        help="Use only a title visibly present in the question")
    parser.add_argument("--rerank", action="store_true", help="Rerank a larger dense candidate pool")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    args = parser.parse_args(argv)
    if args.top_k < 1 or not -1 <= args.min_similarity <= 1:
        parser.error("invalid retrieval limits")
    if args.rerank and args.candidate_k < args.top_k:
        parser.error("candidate-k must be at least top-k")
    payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = payload["cases"]
    if len(cases) != 10:
        parser.error("evaluation must contain exactly ten cases")
    index = load_knowledge_index(payload["index_path"])
    chunks_by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
    for case in cases:
        reference_ids = case.get("reference_chunk_ids")
        if not isinstance(reference_ids, list) or not reference_ids:
            raise ValueError(f"missing reference chunks: {case['id']}")
        references = [chunks_by_id.get(chunk_id) for chunk_id in reference_ids]
        if any(
            chunk is None or chunk.kind != case["expected_kind"]
            or case["expected_title"].casefold() not in chunk.title.casefold()
            for chunk in references
        ) or not any(
            case["reference_locator"] in chunk.locator for chunk in references
        ):
            raise ValueError(f"reference chunks do not match source: {case['id']}")
    embedder = SentenceTransformerEmbedding(
        model_name=index.model_name, local_files_only=True, device="cpu"
    )
    generator = GatewayChatCompletionGenerator.from_config() if args.online else None
    reranker = CrossEncoderReranker(args.reranker_model, offline=True) if args.rerank else None
    records = []
    for case in cases:
        query_title = case.get("query_title")
        if query_title is not None and (
            not isinstance(query_title, str)
            or query_title.casefold() not in case["question"].casefold()
        ):
            raise ValueError(f"query_title is not visible in the question: {case['id']}")
        results = search_knowledge_index(
            case["question"], index, embedder,
            top_k=args.candidate_k if reranker is not None else args.top_k,
            scope=case["scope"],
            source_title=(
                resolve_zotero_pdf_title(index, query_title)
                if args.title_aware and query_title else None
            ),
        )
        eligible = tuple(r for r in results if r.similarity >= args.min_similarity)
        evidence = (
            rerank_candidates(case["question"], eligible, reranker, top_k=args.top_k)
            if reranker is not None else eligible
        )
        candidate_reference = [
            r for r in eligible
            if r.source_kind == case["expected_kind"]
            and case["expected_title"].casefold() in r.source_title.casefold()
            and case["reference_locator"] in r.locator
        ]
        source_matches = [
            r for r in evidence
            if r.source_kind == case["expected_kind"]
            and case["expected_title"].casefold() in r.source_title.casefold()
        ]
        expected = [
            r for r in source_matches
            if case["reference_locator"] in r.locator
        ]
        record = {
            "id": case["id"], "question": case["question"],
            "reference_answer": case["reference_answer"],
            "candidate_evidence_rank": (
                eligible.index(candidate_reference[0]) + 1
            ) if candidate_reference else None,
            "expected_source_rank": (
                evidence.index(source_matches[0]) + 1
            ) if source_matches else None,
            "expected_evidence_rank": (evidence.index(expected[0]) + 1) if expected else None,
            "retrieved": [
                {"source_kind": r.source_kind, "source_title": r.source_title,
                 "locator": r.locator, "similarity": round(r.similarity, 4),
                 "chunk_id": r.chunk_id}
                for r in evidence
            ],
            "generated_answer": None,
            "error": None,
        }
        if args.online:
            try:
                record["generated_answer"] = answer_question(
                    case["question"], evidence, generator
                ).text
            except (RuntimeError, ValueError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        print(f"{case['id']}: expected evidence rank={record['expected_evidence_rank']}; "
              f"retrieved={len(evidence)}; answer={'yes' if record['generated_answer'] else 'no'}",
              flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "config": {"index": payload["index_path"], "top_k": args.top_k,
                   "min_similarity": args.min_similarity, "online": args.online,
                   "title_aware": args.title_aware, "rerank": args.rerank,
                   "candidate_k": args.candidate_k if args.rerank else None,
                   "reranker_model": args.reranker_model if args.rerank else None},
        "cases": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
