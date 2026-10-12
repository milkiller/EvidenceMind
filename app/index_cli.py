"""Command-line entry point for building one local PDF vector index."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .chunking import ChunkConfig, ChunkStrategy, chunk_document
from .embedding import DEFAULT_MODEL_NAME, SentenceTransformerEmbedding
from .json_store import load_document_json
from .pdf_reader import PDFDocument, PDFReadError, read_pdf
from .vector_index import build_vector_index, save_vector_index


SUPPORTED_STRATEGIES = (
    ChunkStrategy.FIXED_CHAR,
    ChunkStrategy.CHAR_OVERLAP,
    ChunkStrategy.PARAGRAPH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 PDF 或解析 JSON，使用本地模型建立透明 JSON 向量索引。"
    )
    parser.add_argument("document_path", help="要建立索引的 PDF 或解析 JSON 路径")
    parser.add_argument("output_path", help="向量索引 JSON 输出路径")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in SUPPORTED_STRATEGIES],
        default=ChunkStrategy.CHAR_OVERLAP.value,
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="只从本地缓存加载模型，不访问 Hugging Face Hub",
    )
    return parser


def _load_document(document_path: str) -> PDFDocument:
    path = Path(document_path)
    if path.suffix.lower() == ".json":
        return load_document_json(path)
    return read_pdf(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunk_config = ChunkConfig(
        strategy=ChunkStrategy(args.strategy),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    try:
        document = _load_document(args.document_path)
        chunks = chunk_document(document, chunk_config)
        embedding_model = SentenceTransformerEmbedding(
            model_name=args.model_name,
            device=args.device,
            local_files_only=args.offline,
        )
        index = build_vector_index(
            chunks,
            embedding_model,
            source_filename=document.filename,
            model_name=embedding_model.model_name,
            normalize_embeddings=embedding_model.normalize_embeddings,
            chunk_config=chunk_config,
        )
        output_path = save_vector_index(index, args.output_path)
    except (FileNotFoundError, ValueError, RuntimeError, PDFReadError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"索引已保存：{output_path}")
    print(f"模型：{index.config.model_name}")
    print(f"向量维度：{index.config.embedding_dimension}")
    print(f"Chunk 数量：{len(index.chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
