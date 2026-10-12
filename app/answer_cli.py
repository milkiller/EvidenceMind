"""Run evidence retrieval and gateway answering from one command."""

import argparse
from collections.abc import Callable, Sequence
import sys

from .answering import Answer, AnswerGenerator, answer_question
from .embedding import EmbeddingModel, SentenceTransformerEmbedding
from .gateway_generator import GatewayChatCompletionGenerator
from .retrieval_cli import parse_min_similarity
from .retriever import SearchResult
from .vector_index import VectorIndex, load_vector_index, search_vector_index


EmbeddingFactory = Callable[..., EmbeddingModel]
GeneratorFactory = Callable[[str], AnswerGenerator]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract for evidence-constrained answering."""
    parser = argparse.ArgumentParser(
        description="检索本地索引，并通过中转站生成带页码引用的回答。"
    )
    parser.add_argument("index_path", help="由 app.index_cli 生成的索引 JSON 路径")
    parser.add_argument("query", help="要回答的问题")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=parse_min_similarity, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--config", default="config.yaml", help="非敏感运行配置路径")
    return parser


def format_answer(answer: Answer) -> str:
    """Render answer text followed by deterministic citations, when present."""
    if not answer.citations:
        return answer.text
    citations = "\n".join(
        (
            f"- {citation.source_title}"
            + (f"（版本 {citation.version}）" if citation.version else "")
            + f" | {citation.locator} | Chunk：{citation.chunk_id}"
            if citation.source_title and citation.locator else
            f"- 第 {citation.page_number} 页 | Chunk：{citation.chunk_id}"
        )
        for citation in answer.citations
    )
    return f"回答：\n{answer.text}\n\n引用：\n{citations}"


def ensure_utf8_stdout() -> None:
    """Preserve model output containing symbols unavailable in Windows GBK."""
    output = sys.stdout
    if getattr(output, "encoding", None) and output.encoding.lower() != "utf-8":
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")


def main(
    argv: Sequence[str] | None = None,
    *,
    embedding_factory: EmbeddingFactory = SentenceTransformerEmbedding,
    generator_factory: GeneratorFactory = GatewayChatCompletionGenerator.from_config,
) -> int:
    """Retrieve evidence first; call the gateway only when evidence survives filtering."""
    ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        index = load_vector_index(args.index_path)
        embedding_model = embedding_factory(
            model_name=index.config.model_name,
            device=args.device,
            normalize_embeddings=index.config.normalize_embeddings,
            local_files_only=args.offline,
        )
        results = search_vector_index(args.query, index, embedding_model, args.top_k)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    evidence = tuple(
        result for result in results if result.similarity >= args.min_similarity
    )
    generator = generator_factory(args.config) if evidence else None
    answer = answer_question(args.query, evidence, generator)
    print(format_answer(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
