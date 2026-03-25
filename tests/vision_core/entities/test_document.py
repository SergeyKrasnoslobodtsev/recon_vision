from hashlib import sha1

from vision_core.entities.document import Document
from vision_core.entities.page import Page


class TestDocument:
    """Проверяет каноническую сущность документа."""

    def test_from_pdf_bytes_sets_hash_pages_and_metadata(self):
        pdf_bytes = b"%PDF-1.4 test document"
        pages = [Page(page_number=0), Page(page_number=1)]

        document = Document.from_pdf_bytes(
            pdf_bytes=pdf_bytes,
            pages=pages,
            metadata={"source": "unit-test"},
        )

        assert document.pages == pages
        assert document.source_hash == sha1(pdf_bytes).hexdigest()
        assert document.metadata == {"source": "unit-test"}

    def test_num_pages_returns_page_count(self):
        document = Document(pages=[Page(page_number=0), Page(page_number=1), Page(page_number=2)])

        assert document.num_pages == 3
