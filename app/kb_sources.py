"""Read local paper revisions, Obsidian notes, and a Zotero collection."""

from dataclasses import dataclass
from contextlib import closing
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from .pdf_reader import PDFReadError, read_pdf


PAPER_VERSION = re.compile(r"(?<!\d)(?P<version>\d+\.\d+)(?!\d)")
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


@dataclass(frozen=True, slots=True)
class TextSection:
    locator: str
    text: str


@dataclass(frozen=True, slots=True)
class KnowledgeSource:
    source_id: str
    kind: str
    title: str
    origin: str
    version: str | None
    is_current: bool
    sections: tuple[TextSection, ...]


@dataclass(frozen=True, slots=True)
class SourceCollection:
    sources: tuple[KnowledgeSource, ...]
    warnings: tuple[str, ...]


def collect_sources(
    papers_root: str | Path,
    obsidian_root: str | Path,
    zotero_db: str | Path,
    *,
    current_version: str = "5.1",
    paper_glob: str = "*.docx",
    zotero_collection: str = "odor",
    excluded_collection: str = "科研绘图",
    include_zotero_pdfs: bool = True,
) -> SourceCollection:
    """Collect only user-selected source types without modifying source files."""
    papers = Path(papers_root).expanduser().resolve()
    vault = Path(obsidian_root).expanduser().resolve()
    database = Path(zotero_db).expanduser().resolve()
    for directory in (papers, vault):
        if not directory.is_dir():
            raise FileNotFoundError(f"source directory not found: {directory}")
    if not database.is_file():
        raise FileNotFoundError(f"Zotero database not found: {database}")

    sources: list[KnowledgeSource] = []
    warnings: list[str] = []
    paper_paths = sorted(
        path for path in papers.glob(paper_glob)
        if path.is_file() and path.suffix.lower() == ".docx"
        and not path.name.startswith("~$")
        and PAPER_VERSION.search(path.stem)
    )
    for path in paper_paths:
        version = PAPER_VERSION.search(path.stem).group("version")  # type: ignore[union-attr]
        paragraphs = _read_docx_paragraphs(path)
        sections = _group_lines(paragraphs, "段落")
        if sections:
            sources.append(
                KnowledgeSource(
                    source_id=_source_id("paper", str(path)),
                    kind="paper",
                    title=path.stem,
                    origin=str(path),
                    version=version,
                    is_current=version == current_version,
                    sections=sections,
                )
            )
        else:
            warnings.append(f"empty paper: {path.name}")
    if not any(source.is_current for source in sources):
        raise ValueError(f"current paper version {current_version} was not found")

    for path in sorted(vault.rglob("*")):
        if not path.is_file() or ".obsidian" in path.parts:
            continue
        if path.suffix.lower() == ".md":
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            sections = _group_lines(lines, "行")
            if not sections:
                warnings.append(f"empty note: {path.relative_to(vault)}")
                continue
            sources.append(
                KnowledgeSource(
                    _source_id("obsidian", str(path)),
                    "obsidian",
                    str(path.relative_to(vault)),
                    str(path),
                    None,
                    False,
                    sections,
                )
            )
        elif path.suffix.lower() == ".pdf":
            source = _pdf_source(path, "obsidian_pdf", path.name)
            if source is None:
                warnings.append(f"unreadable PDF: {path.relative_to(vault)}")
            else:
                sources.append(source)

    zotero_sources, zotero_warnings = _read_zotero(
        database,
        zotero_collection,
        excluded_collection,
        include_zotero_pdfs,
    )
    sources.extend(zotero_sources)
    warnings.extend(zotero_warnings)
    return SourceCollection(tuple(sources), tuple(warnings))


def _source_id(kind: str, identity: str) -> str:
    return f"{kind}:{sha256(identity.encode('utf-8')).hexdigest()[:12]}"


def _read_docx_paragraphs(path: Path) -> list[str]:
    """Extract visible Word text; DOCX has no reliable page numbers here."""
    try:
        with ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
    except (BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError(f"cannot read DOCX text: {path.name}") from exc
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:body//w:p", WORD_NS):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", WORD_NS))
        paragraphs.append(text.strip())
    return paragraphs


def _group_lines(lines: list[str], unit: str, limit: int = 900) -> tuple[TextSection, ...]:
    sections: list[TextSection] = []
    buffer: list[str] = []
    start = 0
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if buffer and sum(len(part) + 1 for part in buffer) + len(line) > limit:
            sections.append(TextSection(f"{unit} {start}-{number - 1}", "\n".join(buffer)))
            buffer = []
        if not buffer:
            start = number
        buffer.append(line)
    if buffer:
        sections.append(TextSection(f"{unit} {start}-{len(lines)}", "\n".join(buffer)))
    return tuple(sections)


def _pdf_source(path: Path, kind: str, title: str, *, identity: str | None = None) -> KnowledgeSource | None:
    try:
        document = read_pdf(path)
    except (PDFReadError, FileNotFoundError, ValueError):
        return None
    sections = tuple(
        TextSection(f"第 {page.page_number} 页", page.text)
        for page in document.pages
        if page.text.strip()
    )
    if not sections:
        return None
    return KnowledgeSource(
        _source_id(kind, identity or str(path)), kind, title, str(path), None, False, sections
    )


class _NoteText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_note(html: str) -> str:
    parser = _NoteText()
    parser.feed(html)
    return " ".join(part.strip() for part in parser.parts if part.strip())


def _read_zotero(
    database: Path,
    collection_name: str,
    excluded_name: str,
    include_pdfs: bool,
) -> tuple[list[KnowledgeSource], list[str]]:
    """Read a live Zotero database through an immutable, non-writing view."""
    uri = f"file:///{quote(database.as_posix().lstrip('/'), safe='/:')}?immutable=1"
    warnings: list[str] = []
    if database.with_name(database.name + "-journal").exists():
        warnings.append("Zotero is open: the read-only snapshot may omit uncommitted changes")
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Zotero read-only snapshot failed integrity check")
        roots = connection.execute(
            "SELECT collectionID FROM collections WHERE collectionName = ?", (collection_name,)
        ).fetchall()
        if len(roots) != 1:
            raise ValueError(f"expected one Zotero collection named {collection_name}")
        root_id = roots[0][0]
        included = _collection_tree(connection, root_id)
        excluded_roots = connection.execute(
            "SELECT collectionID FROM collections WHERE collectionName = ? AND parentCollectionID IN "
            f"({','.join('?' for _ in included)})",
            (excluded_name, *included),
        ).fetchall()
        excluded = set().union(*(_collection_tree(connection, row[0]) for row in excluded_roots))
        allowed = included - excluded
        if not allowed:
            return [], warnings
        ids = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT itemID FROM collectionItems WHERE collectionID IN "
                f"({','.join('?' for _ in allowed)})",
                tuple(sorted(allowed)),
            )
        }
        if excluded:
            ids -= {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT itemID FROM collectionItems WHERE collectionID IN "
                    f"({','.join('?' for _ in excluded)})",
                    tuple(sorted(excluded)),
                )
            }
        sources: list[KnowledgeSource] = []
        seen_attachments: set[int] = set()
        for item_id in sorted(ids):
            item = connection.execute(
                "SELECT i.key, t.typeName FROM items i JOIN itemTypes t ON t.itemTypeID=i.itemTypeID "
                "WHERE i.itemID=?", (item_id,),
            ).fetchone()
            if item is None:
                continue
            key, item_type = item
            fields = dict(connection.execute(
                "SELECT f.fieldName, v.value FROM itemData d JOIN fields f ON f.fieldID=d.fieldID "
                "JOIN itemDataValues v ON v.valueID=d.valueID WHERE d.itemID=?", (item_id,),
            ))
            title = fields.get("title") or key
            if item_type != "attachment":
                parts = [f"标题：{title}"]
                for field, label in (("date", "日期"), ("DOI", "DOI"), ("abstractNote", "摘要")):
                    if fields.get(field):
                        parts.append(f"{label}：{fields[field]}")
                notes = connection.execute(
                    "SELECT note FROM itemNotes WHERE parentItemID=?", (item_id,)
                ).fetchall()
                parts.extend(f"笔记：{_plain_note(note)}" for (note,) in notes if note)
                sources.append(KnowledgeSource(
                    _source_id("zotero", key), "zotero", title, f"zotero:{key}",
                    None, False, (TextSection("题录/摘要/笔记", "\n".join(parts)),),
                ))
            if not include_pdfs:
                continue
            attachments = connection.execute(
                "SELECT a.itemID, a.path, i.key FROM itemAttachments a "
                "JOIN items i ON i.itemID=a.itemID "
                "WHERE (a.parentItemID=? OR a.itemID=?) AND a.contentType='application/pdf'",
                (item_id, item_id),
            ).fetchall()
            for attachment_id, stored_path, attachment_key in attachments:
                if attachment_id in seen_attachments:
                    continue
                seen_attachments.add(attachment_id)
                if not stored_path or not stored_path.startswith("storage:"):
                    warnings.append(f"Zotero PDF has no supported local path: {attachment_key}")
                    continue
                path = (database.parent / "storage" / attachment_key / stored_path[8:]).resolve()
                if not path.is_relative_to((database.parent / "storage").resolve()):
                    warnings.append(f"Zotero PDF path escaped storage: {attachment_key}")
                    continue
                source = _pdf_source(path, "zotero_pdf", title, identity=attachment_key)
                if source is None:
                    warnings.append(f"Zotero PDF missing or has no text layer: {attachment_key}")
                else:
                    sources.append(source)
    return sources, warnings


def _collection_tree(connection: sqlite3.Connection, root_id: int) -> set[int]:
    return {
        row[0]
        for row in connection.execute(
            "WITH RECURSIVE descendants(id) AS (SELECT ? UNION ALL "
            "SELECT c.collectionID FROM collections c JOIN descendants d "
            "ON c.parentCollectionID=d.id) SELECT id FROM descendants",
            (root_id,),
        )
    }
