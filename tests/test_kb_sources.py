import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

from app.kb_sources import collect_sources


DOCX_XML = (
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:body><w:p><w:r><w:t>method</w:t></w:r></w:p></w:body></w:document>'
)


class KnowledgeSourcesTests(unittest.TestCase):
    def test_collects_current_and_old_papers_notes_and_filtered_zotero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            papers = root / "papers"
            vault = root / "vault"
            papers.mkdir()
            vault.mkdir()
            for version in ("4.0", "5.1"):
                path = papers / f"sample-paper-{version}-draft.docx"
                with ZipFile(path, "w") as archive:
                    archive.writestr("word/document.xml", DOCX_XML)
            (papers / "~$draft_5.1.docx").write_bytes(b"Word lock file")
            (vault / "note.md").write_text("# Note\nEvidence", encoding="utf-8")
            hidden = vault / ".obsidian"
            hidden.mkdir()
            (hidden / "settings.md").write_text("private config", encoding="utf-8")

            database = root / "zotero.sqlite"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript("""
                    CREATE TABLE collections (collectionID INTEGER, collectionName TEXT,
                        parentCollectionID INTEGER);
                    CREATE TABLE collectionItems (collectionID INTEGER, itemID INTEGER);
                    CREATE TABLE items (itemID INTEGER, key TEXT, itemTypeID INTEGER);
                    CREATE TABLE itemTypes (itemTypeID INTEGER, typeName TEXT);
                    CREATE TABLE fields (fieldID INTEGER, fieldName TEXT);
                    CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
                    CREATE TABLE itemDataValues (valueID INTEGER, value TEXT);
                    CREATE TABLE itemNotes (parentItemID INTEGER, note TEXT);
                    CREATE TABLE itemAttachments (itemID INTEGER, parentItemID INTEGER,
                        path TEXT, contentType TEXT);
                    INSERT INTO collections VALUES (1,'odor',NULL),(2,'科研绘图',1);
                    INSERT INTO collectionItems VALUES (1,10),(2,11);
                    INSERT INTO itemTypes VALUES (1,'journalArticle');
                    INSERT INTO items VALUES (10,'GOOD',1),(11,'SKIP',1);
                    INSERT INTO fields VALUES (1,'title'),(2,'abstractNote');
                    INSERT INTO itemData VALUES (10,1,1),(10,2,2),(11,1,3);
                    INSERT INTO itemDataValues VALUES (1,'Odor paper'),(2,'Abstract'),
                        (3,'Drawing paper');
                    INSERT INTO itemNotes VALUES (10,'<p>My <b>note</b></p>');
                """)
                connection.commit()
            collection = collect_sources(papers, vault, database, include_zotero_pdfs=False)
            kinds = [source.kind for source in collection.sources]
            self.assertEqual(kinds.count("paper"), 2)
            self.assertEqual(kinds.count("obsidian"), 1)
            self.assertEqual(kinds.count("zotero"), 1)
            self.assertEqual([s.version for s in collection.sources if s.is_current], ["5.1"])
            self.assertIn("My note", collection.sources[-1].sections[0].text)
            self.assertNotIn("Drawing paper", [s.title for s in collection.sources])


if __name__ == "__main__":
    unittest.main()
