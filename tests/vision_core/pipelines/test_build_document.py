from hashlib import sha1
from unittest.mock import MagicMock

import numpy as np

from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.pipelines import build_document as build_document_module
from vision_core.pipelines.build_document import DocumentBuildPipeline


class FakeLoader:
    """Подменяет PDFLoader в тестах pipeline."""

    closed = False

    def __init__(self, pdf_bytes: bytes):
        self.pdf_bytes = pdf_bytes
        self.num_pages = 2

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_page_image(self, page_num: int, dpi: int = 300):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def get_page_size(self, page_num: int) -> tuple[float, float]:
        return (100.0 + page_num, 200.0 + page_num)

    def has_text_layer(self, page_num: int) -> bool:
        return page_num % 2 == 0

    def close(self):
        self.__class__.closed = True


class TestDocumentBuildPipeline:
    """Проверяет сборку канонического документа из PDF."""

    def test_build_creates_document_with_page_metadata(self, monkeypatch):
        pdf_bytes = b"%PDF-1.4 pipeline test"
        FakeLoader.closed = False
        monkeypatch.setattr(build_document_module, "PDFLoader", FakeLoader)

        pipeline = DocumentBuildPipeline.__new__(DocumentBuildPipeline)
        pipeline.dpi = 200
        pipeline._process_page = MagicMock(
            side_effect=lambda image, page_number: Page(metadata={"analyzed": True, "image": image})
        )

        document = pipeline.build(pdf_bytes)

        assert isinstance(document, Document)
        assert document.source_hash == sha1(pdf_bytes).hexdigest()
        assert document.metadata == {"dpi": 200, "num_pages": 2}
        assert document.num_pages == 2

        first_page = document.pages[0]
        second_page = document.pages[1]

        assert first_page.page_number == 0
        assert first_page.metadata["source_page_number"] == 0
        assert first_page.metadata["page_size"] == [100.0, 200.0]
        assert first_page.metadata["has_text_layer"] is True

        assert second_page.page_number == 1
        assert second_page.metadata["source_page_number"] == 1
        assert second_page.metadata["page_size"] == [101.0, 201.0]
        assert second_page.metadata["has_text_layer"] is False

        assert FakeLoader.closed is True
