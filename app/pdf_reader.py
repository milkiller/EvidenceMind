"""Read page-level text and metadata from PDF files."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # pragma: no cover - exercised only without optional install
    PdfReader = None  # type: ignore[assignment,misc]


class PDFReadError(RuntimeError):
    """Raised when a PDF exists but cannot be parsed safely."""


@dataclass(frozen=True, slots=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class PDFDocument:
    path: Path
    filename: str
    page_count: int
    pages: tuple[PageText, ...]


def read_pdf(pdf_path: str | Path) -> PDFDocument:
    """Validate a PDF path and extract text while preserving page numbers."""
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{path}")
    if not path.is_file():
        raise ValueError(f"PDF 路径不是文件：{path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"只支持 .pdf 文件：{path.name}")

    if PdfReader is not None:
        try:
            pages = _read_with_pypdf(path)
        except PDFReadError:
            if shutil.which("pdftotext") is None:
                raise
            pages = _read_with_poppler(path)
        if pages and not any(page.text for page in pages) and shutil.which("pdftotext"):
            pages = _read_with_poppler(path)
    else:
        pages = _read_with_poppler(path)

    return PDFDocument(
        path=path,
        filename=path.name,
        page_count=len(pages),
        pages=pages,
    )


def _read_with_pypdf(path: Path) -> tuple[PageText, ...]:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise PDFReadError("PDF 已加密，无法使用空密码读取")
        return tuple(
            PageText(page_number=index, text=(page.extract_text() or "").strip())
            for index, page in enumerate(reader.pages, start=1)
        )
    except PDFReadError:
        raise
    except Exception as exc:
        raise PDFReadError(f"PDF 解析失败：{path.name}") from exc


def _read_with_poppler(path: Path) -> tuple[PageText, ...]:
    """Use pdftotext as a local fallback when Python dependencies are unavailable."""
    executable = shutil.which("pdftotext")
    if executable is None:
        raise PDFReadError("未安装 pypdf，且找不到 pdftotext；请先安装 requirements.txt")
    try:
        result = subprocess.run(
            [executable, "-layout", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PDFReadError(f"PDF 解析失败：{path.name}") from exc

    raw_pages = result.stdout.split("\f")
    if raw_pages and raw_pages[-1] == "":
        raw_pages.pop()
    return tuple(
        PageText(page_number=index, text=text.strip())
        for index, text in enumerate(raw_pages, start=1)
    )
