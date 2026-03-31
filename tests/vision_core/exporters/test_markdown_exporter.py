"""Тесты для DocumentMarkdownExporter."""

from pathlib import Path

import pytest
from loguru import logger

from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.entities.paragraph import Paragraph, ParagraphType
from vision_core.entities.table import Table
from vision_core.exporters.markdown_exporter import DocumentMarkdownExporter


def _make_cell(row: int, col: int, value: str, colspan: int = 1, rowspan: int = 1) -> Cell:
    return Cell(
        row=row,
        col=col,
        colspan=colspan,
        rowspan=rowspan,
        value=value,
        bbox=BBox(x_min=col * 100, y_min=row * 30, x_max=(col + colspan) * 100, y_max=(row + rowspan) * 30),
    )


def _make_paragraph(text: str, y: float, kind: ParagraphType = ParagraphType.TEXT) -> Paragraph:
    return Paragraph(
        id="p1",
        text=text,
        type=kind,
        bbox=BBox(x_min=0, y_min=y, x_max=500, y_max=y + 20),
    )


class TestDocumentMarkdownExporter:
    def test_simple_table(self):
        cells = [
            _make_cell(0, 0, "Описание"),
            _make_cell(0, 1, "Дебет"),
            _make_cell(0, 2, "Кредит"),
            _make_cell(1, 0, "Сальдо"),
            _make_cell(1, 1, "51 043"),
            _make_cell(1, 2, ""),
        ]
        table = Table(id="t1", bbox=BBox(x_min=0, y_min=50, x_max=300, y_max=150), num_rows=2, num_cols=3, cells=cells)
        page = Page(page_number=0, tables=[table])
        doc = Document(pages=[page])

        md = DocumentMarkdownExporter().export(doc)

        assert "| Описание | Дебет | Кредит |" in md
        assert "| Сальдо | 51 043 |" in md
        assert "---" in md

    def test_colspan_expands(self):
        cells = [
            _make_cell(0, 0, "Заголовок", colspan=2),
            _make_cell(1, 0, "A"),
            _make_cell(1, 1, "B"),
        ]
        table = Table(id="t1", bbox=BBox(x_min=0, y_min=0, x_max=200, y_max=60), num_rows=2, num_cols=2, cells=cells)
        page = Page(page_number=0, tables=[table])
        doc = Document(pages=[page])

        md = DocumentMarkdownExporter().export(doc)

        # colspan=2: первая ячейка — значение, вторая — маркер "<"
        assert "| Заголовок | < |" in md

    def test_reading_order_by_y(self):
        paragraph = _make_paragraph("Введение", y=10)
        cells = [_make_cell(0, 0, "Данные"), _make_cell(0, 1, "Значение")]
        table = Table(id="t1", bbox=BBox(x_min=0, y_min=100, x_max=200, y_max=150), num_rows=1, num_cols=2, cells=cells)
        page = Page(page_number=0, tables=[table], paragraphs=[paragraph])
        doc = Document(pages=[page])

        md = DocumentMarkdownExporter().export(doc)

        # Параграф (y=10) должен идти до таблицы (y=100)
        assert md.index("Введение") < md.index("Данные")

    def test_paragraph_types(self):
        header = _make_paragraph("Акт сверки", y=0, kind=ParagraphType.HEADER)
        footer = _make_paragraph("Подпись", y=900, kind=ParagraphType.FOOTER)
        text = _make_paragraph("Описание", y=100, kind=ParagraphType.TEXT)
        page = Page(page_number=0, paragraphs=[header, footer, text])
        doc = Document(pages=[page])

        md = DocumentMarkdownExporter().export(doc)

        assert "### Акт сверки" in md
        assert "*Подпись*" in md
        assert "Описание" in md

    def test_multipage_separator(self):
        page1 = Page(page_number=0, paragraphs=[_make_paragraph("Страница 1", y=0)])
        page2 = Page(page_number=1, paragraphs=[_make_paragraph("Страница 2", y=0)])
        doc = Document(pages=[page1, page2])

        md = DocumentMarkdownExporter().export(doc)

        assert "---" in md
        assert "Страница 1" in md
        assert "Страница 2" in md

    def test_integration_on_real_pdf(self, pdf_path: Path, output_dir: Path):
        """Экспортирует реальный PDF в Markdown и сохраняет результат."""
        from vision_core.pipelines.build_document import DocumentBuildPipeline

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")

        pipeline = DocumentBuildPipeline()
        exporter = DocumentMarkdownExporter()

        for test_file in pdf_files:
            logger.info(f"Экспорт: {test_file.name}")
            pdf_bytes = test_file.read_bytes()
            document = pipeline.build(pdf_bytes)
            md = exporter.export(document)

            out_path = output_dir / f"{test_file.stem}.md"
            out_path.write_text(md, encoding="utf-8")
            logger.success(f"Markdown сохранён: {out_path}")

            assert len(md) > 0
            assert "Страница" in md
