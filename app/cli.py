"""Command-line entry point for the minimal PDF reader."""

import argparse
from collections.abc import Sequence

from .pdf_reader import PDFReadError, read_pdf
from .json_store import save_document_json

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 PDF，并输出文件名、页数和逐页文本摘要。"
    )
    parser.add_argument("pdf_path", help="要读取的 PDF 文件路径")
    parser.add_argument(
        "--output",
        help="保存完整解析结果的JSON文件路径（可选）",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default= 1200,
        help="每页最多显示的字符数，默认 1000",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preview_chars < 0:
        raise SystemExit("--preview-chars 不能小于 0")

    try:
        document = read_pdf(args.pdf_path)
    except (FileNotFoundError, ValueError, PDFReadError) as exc:
        raise SystemExit(str(exc)) from exc
    
    if args.output:
        output_path = save_document_json(document, args.output)
    print(f"JSON 已保存：{output_path}")

    print(f"文件名：{document.filename}")
    print(f"页数：{document.page_count}")
    for page in document.pages:
        preview = page.text[: args.preview_chars]
        print(f"\n--- 第 {page.page_number} 页 · {len(page.text)} 字符 ---")
        print(preview if preview else "[未提取到文本]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

