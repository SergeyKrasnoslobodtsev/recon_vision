from hashlib import sha1

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
        return {"page_num": page_num, "dpi": dpi}

    def get_page_size(self, page_num: int) -> tuple[float, float]:
        return (100.0 + page_num, 200.0 + page_num)

    def has_text_layer(self, page_num: int) -> bool:
        return page_num % 2 == 0

    def close(self):
        self.__class__.closed = True


class FakePageAnalyzer:
    """Подменяет анализатор страниц и фиксирует вызовы."""

    def __init__(self):
        self.images = []

    def analyze_page(self, image) -> Page:
        self.images.append(image)
        return Page(metadata={"analyzed": True, "image": image})


class TestDocumentBuildPipeline:
    """Проверяет сборку канонического документа из PDF."""

    def test_build_creates_document_with_page_metadata(self, monkeypatch):
        pdf_bytes = b"%PDF-1.4 pipeline test"
        analyzer = FakePageAnalyzer()
        FakeLoader.closed = False
        monkeypatch.setattr(build_document_module, "PDFLoader", FakeLoader)

        pipeline = DocumentBuildPipeline(page_analyzer=analyzer, dpi=200)

        document = pipeline.build(pdf_bytes)

        assert isinstance(document, Document)
        assert document.source_hash == sha1(pdf_bytes).hexdigest()
        assert document.metadata == {"dpi": 200, "num_pages": 2}
        assert document.num_pages == 2
        assert analyzer.images == [
            {"page_num": 0, "dpi": 200},
            {"page_num": 1, "dpi": 200},
        ]

        first_page = document.pages[0]
        second_page = document.pages[1]

        assert first_page.page_number == 0
        assert first_page.metadata == {
            "analyzed": True,
            "image": {"page_num": 0, "dpi": 200},
            "source_page_number": 0,
            "page_size": [100.0, 200.0],
            "has_text_layer": True,
        }

        assert second_page.page_number == 1
        assert second_page.metadata == {
            "analyzed": True,
            "image": {"page_num": 1, "dpi": 200},
            "source_page_number": 1,
            "page_size": [101.0, 201.0],
            "has_text_layer": False,
        }
        assert FakeLoader.closed is True
