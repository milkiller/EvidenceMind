import tempfile
import unittest
from pathlib import Path

from app.json_store import load_document_json, save_document_json
from app.pdf_reader import PDFDocument, PageText


class DocumentJsonStoreTests(unittest.TestCase):
    def test_save_and_load_round_trip_without_original_pdf(self) -> None:
        document = PDFDocument(
            path=Path("missing-source.pdf"),
            filename="paper.pdf",
            page_count=2,
            pages=(
                PageText(page_number=1, text="第一页"),
                PageText(page_number=2, text="第二页"),
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            json_path = save_document_json(
                document,
                Path(directory) / "document.json",
            )
            restored = load_document_json(json_path)

        self.assertEqual(restored, document)


if __name__ == "__main__":
    unittest.main()
