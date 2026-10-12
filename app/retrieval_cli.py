"""Query a saved vector index and print traceable Top-K evidence."""

import argparse
from collections.abc import Sequence

from .embedding import SentenceTransformerEmbedding
from .retriever import SearchResult
from .vector_index import load_vector_index, search_vector_index

def parse_min_similarity(value: str) -> float:
    """Parse a string as a float and validate it is in the range [-1, 1]."""
    try:
        value_float = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{value}' 不是一个有效的浮点数") from exc
    if not -1.0 <= value_float <= 1.0:
        raise argparse.ArgumentTypeError(f"'{value}' 必须在 [-1, 1] 范围内")
    return value_float


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract for querying one saved index."""
    
    parser = argparse.ArgumentParser(
        description="查询本地向量索引，返回带页码和相似度的 Top-K 原文。"
    )
    parser.add_argument("index_path", help="由 app.index_cli 生成的索引 JSON 路径")
    parser.add_argument("query", help="要检索的问题")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="只从本地缓存加载模型，不访问 Hugging Face Hub",
    )
    parser.add_argument(
        "--min-similarity",
        type=parse_min_similarity,
        default=0.0,
        help="过滤掉相似度低于此阈值的结果",
    )
    return parser


def format_search_results(results: Sequence[SearchResult]) -> str:
    """Format ranked evidence without dropping its source metadata."""
    blocks = []
    for rank, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                (
                    f"[{rank}] 页码：{result.page_number} | "
                    f"相似度：{result.similarity:.4f} | "
                    f"Chunk：{result.chunk_id}",
                    result.text.strip(),
                )
            )
        )
    return "\n\n".join(blocks)


def main(argv: Sequence[str] | None = None) -> int:
    """Load the index model configuration and execute one Top-K query."""
    args = build_parser().parse_args(argv)
    try:
        index = load_vector_index(args.index_path)
        embedding_model = SentenceTransformerEmbedding(
            model_name=index.config.model_name,
            device=args.device,
            normalize_embeddings=index.config.normalize_embeddings,
            local_files_only=args.offline,
        )
        results = search_vector_index(
            args.query,
            index,
            embedding_model,
            top_k=args.top_k,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    filtered_results = tuple(
        result
        for result in results
        if result.similarity >= args.min_similarity
    )

    print(f"模型：{index.config.model_name}")
    print(f"问题：{args.query}")
    if not filtered_results:
        print("没有达到相似度阈值的检索结果")
    else:
        print(format_search_results(filtered_results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
