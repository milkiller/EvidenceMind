import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.pdf_reader import PDFReadError, PageText, read_pdf


def build_two_page_pdf() -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F1 6 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F1 6 0 R >> >> /Contents 7 0 R >>",
        b"<< /Length 42 >>\nstream\nBT /F1 12 Tf 20 100 Td (First page) Tj ET\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length 43 >>\nstream\nBT /F1 12 Tf 20 100 Td (Second page) Tj ET\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(output)


class ReadPdfTests(unittest.TestCase):
    def test_reads_filename_page_count_and_page_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pdf"
            path.write_bytes(build_two_page_pdf())

            document = read_pdf(path)

            self.assertEqual(document.filename, "sample.pdf")
            self.assertEqual(document.page_count, 2)
            self.assertEqual([page.page_number for page in document.pages], [1, 2])
            self.assertEqual([page.text for page in document.pages], ["First page", "Second page"])

    def test_rejects_non_pdf_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("not a pdf", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "只支持 .pdf"):
                read_pdf(path)

    def test_reports_missing_file(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "PDF 文件不存在"):
            read_pdf("missing.pdf")

    def test_uses_poppler_when_pypdf_cannot_extract_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fallback.pdf"
            path.write_bytes(build_two_page_pdf())
            with (
                patch("app.pdf_reader._read_with_pypdf", side_effect=PDFReadError("bad font")),
                patch("app.pdf_reader.shutil.which", return_value="pdftotext"),
                patch("app.pdf_reader._read_with_poppler", return_value=(PageText(1, "ok"),)),
            ):
                document = read_pdf(path)
            self.assertEqual(document.pages, (PageText(1, "ok"),))


if __name__ == "__main__":
    unittest.main()
