"""Собирает канонический Document из PDF-файла."""

from __future__ import annotations

from vision_core.analizer.page_analyzer import PageAnalyzer
from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.loader.pdf_loader import PDFLoader


class DocumentBuildPipeline:
    """Строит каноническое представление документа из PDF.

    Attributes:
        page_analyzer: Анализатор отдельной страницы документа.
        dpi: Разрешение рендеринга PDF-страниц.
    """

    def __init__(
        self,
        page_analyzer: PageAnalyzer | None = None,
        dpi: int = 300,
    ):
        """Инициализирует pipeline построения документа.

        Args:
            page_analyzer: Анализатор страниц. Если не задан, создаётся по умолчанию.
            dpi: DPI для рендеринга страниц документа.
        """
        self.page_analyzer = page_analyzer or PageAnalyzer()
        self.dpi = dpi

    def build(self, pdf_bytes: bytes) -> Document:
        """Строит канонический документ из PDF.

        Args:
            pdf_bytes: Исходный PDF-файл в байтах.

        Returns:
            Document: Каноническое представление документа.
        """
        pages: list[Page] = []

        with PDFLoader(pdf_bytes) as loader:
            for page_number in range(loader.num_pages):
                image = loader.get_page_image(page_number, dpi=self.dpi)
                analyzed_page = self.page_analyzer.analyze_page(
                    image,
                    page_number=page_number,
                )
                analyzed_page.page_number = page_number
                analyzed_page.metadata.update(
                    {
                        "source_page_number": page_number,
                        "page_size": list(loader.get_page_size(page_number)),
                        "has_text_layer": loader.has_text_layer(page_number),
                    }
                )
                pages.append(analyzed_page)

        return Document.from_pdf_bytes(
            pdf_bytes=pdf_bytes,
            pages=pages,
            metadata={
                "dpi": self.dpi,
                "num_pages": len(pages),
            },
        )
