"""Persist and restore page-level PDF document metadata."""

import json
from pathlib import Path

from .pdf_reader import PDFDocument, PageText


def save_document_json(document: PDFDocument, output_json_path: str | Path) -> Path:
    """Save a PDFDocument to a JSON file."""
    path = Path(output_json_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "source_path": str(document.path),
        "filename": document.filename,
        "page_count": document.page_count,
        "pages": [
            {"page_number": page.page_number, "text": page.text}
            for page in document.pages
        ],
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_document_json(document_json_path: str | Path) -> PDFDocument:
    """Restore a PDFDocument without requiring the original PDF to exist."""
    path = Path(document_json_path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported document JSON schema version")

    pages = tuple(
        PageText(page_number=item["page_number"], text=item["text"])
        for item in data["pages"]
    )
    if data["page_count"] != len(pages):
        raise ValueError("page_count does not match the number of stored pages")
    return PDFDocument(
        path=Path(data["source_path"]),
        filename=data["filename"],
        page_count=data["page_count"],
        pages=pages,
    )
