"""Build and ask a local, source-aware research knowledge base."""

import argparse
from collections import Counter
from collections.abc import Sequence

from .answer_cli import ensure_utf8_stdout, format_answer
from .answering import answer_question
from .embedding import SentenceTransformerEmbedding
from .gateway_generator import GatewayChatCompletionGenerator, GatewayRequestError
from .kb_index import (
    MULTILINGUAL_MODEL_NAME, build_knowledge_index, load_knowledge_index, save_knowledge_index,
    search_knowledge_index,
)
from .kb_sources import collect_sources
from .retrieval_cli import parse_min_similarity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地多来源知识库：建库与证据问答")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="只读源文件，生成本地索引")
    build.add_argument("index_path")
    build.add_argument("--papers-root", required=True)
    build.add_argument("--obsidian-root", required=True)
    build.add_argument("--zotero-db", required=True)
    build.add_argument("--zotero-collection", default="odor")
    build.add_argument("--exclude-zotero-collection", default="科研绘图")
    build.add_argument("--current-version", default="5.1")
    build.add_argument("--paper-glob", default="*.docx")
    build.add_argument("--no-zotero-pdfs", action="store_true")
    build.add_argument("--model-name", default=MULTILINGUAL_MODEL_NAME)
    build.add_argument("--chunk-size", type=int, default=800)
    build.add_argument("--overlap", type=int, default=120)
    build.add_argument("--offline", action="store_true")

    ask = commands.add_parser("ask", help="检索证据并按需调用中转站")
    ask.add_argument("index_path")
    ask.add_argument("query")
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--min-similarity", type=parse_min_similarity, default=0.0)
    ask.add_argument("--history", action="store_true", help="允许检索论文旧版")
    ask.add_argument("--paper-versions", help="只检索指定论文版本，如 5.0,5.1")
    ask.add_argument("--scope", choices=("research", "paper", "literature", "literature_pdf", "all"),
                     default="research", help="默认只检索研究相关来源；all 包含整个 vault")
    ask.add_argument("--source-title", help="只检索标题含指定文本的来源")
    ask.add_argument("--retrieval-only", action="store_true", help="仅列来源，不发送片段")
    ask.add_argument("--config", default="config.yaml")
    ask.add_argument("--offline", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            collection = collect_sources(
                args.papers_root, args.obsidian_root, args.zotero_db,
                current_version=args.current_version,
                paper_glob=args.paper_glob,
                zotero_collection=args.zotero_collection,
                excluded_collection=args.exclude_zotero_collection,
                include_zotero_pdfs=not args.no_zotero_pdfs,
            )
            model = SentenceTransformerEmbedding(
                model_name=args.model_name, local_files_only=args.offline, device="cpu"
            )
            index = build_knowledge_index(
                collection.sources, model, model_name=model.model_name,
                chunk_size=args.chunk_size, overlap=args.overlap,
            )
            target = save_knowledge_index(index, args.index_path)
            counts = Counter(source.kind for source in collection.sources)
            print(f"本地索引：{target}")
            print(f"来源：{dict(sorted(counts.items()))}；片段：{len(index.chunks)}")
            for warning in collection.warnings:
                print(f"跳过/提醒：{warning}")
            return 0

        index = load_knowledge_index(args.index_path)
        model = SentenceTransformerEmbedding(
            model_name=index.model_name, local_files_only=args.offline, device="cpu"
        )
        results = search_knowledge_index(
            args.query, index, model, top_k=args.top_k,
            include_history=args.history,
            scope=args.scope,
            paper_versions=(
                tuple(part.strip() for part in args.paper_versions.split(","))
                if args.paper_versions else None
            ),
            source_title=args.source_title,
        )
        evidence = tuple(
            result for result in results if result.similarity >= args.min_similarity
        )
        if args.retrieval_only:
            if not evidence:
                print("没有达到阈值的证据。")
            for rank, result in enumerate(evidence, start=1):
                print(
                    f"[{rank}] {result.source_title} | {result.locator} | "
                    f"相似度 {result.similarity:.4f} | {result.chunk_id}"
                )
            return 0
        generator = GatewayChatCompletionGenerator.from_config(args.config) if evidence else None
        answer = answer_question(args.query, evidence, generator)
        print(format_answer(answer))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, GatewayRequestError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
