from hashlib import sha1
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from vision_core.debug_image_observer import DebugImageObserver
from vision_core.entities.bbox import BBox
from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.entities.table import Table
from vision_core.pipelines import build_document as build_document_module
from vision_core.pipelines.build_document import DocumentBuildPipeline
from vision_core.postprocessor.table_id_assigner import TableIdAssigner


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
        pipeline.continuation_linker = MagicMock()
        pipeline.table_id_assigner = MagicMock()
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

        pipeline.continuation_linker.link.assert_called_once()
        pipeline.table_id_assigner.assign.assert_called_once()
        assert FakeLoader.closed is True

    def test_build_reassigns_table_ids_in_document_order(self, monkeypatch):
        pdf_bytes = b"%PDF-1.4 pipeline ids"
        FakeLoader.closed = False
        monkeypatch.setattr(build_document_module, "PDFLoader", FakeLoader)

        pipeline = DocumentBuildPipeline.__new__(DocumentBuildPipeline)
        pipeline.dpi = 300
        pipeline.table_id_assigner = TableIdAssigner()
        pipeline.continuation_linker = MagicMock()
        pipeline.continuation_linker.link.side_effect = lambda pages: setattr(
            pages[1].tables[0], "continuation_of", "0"
        )

        def _page_with_table(_: np.ndarray, page_number: int) -> Page:
            return Page(
                tables=[
                    Table(
                        id="table_0",
                        bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10),
                        num_rows=2,
                        num_cols=2,
                    )
                ],
                metadata={"image_shape": [100, 100]},
            )

        pipeline._process_page = MagicMock(side_effect=_page_with_table)

        document = pipeline.build(pdf_bytes)

        first_table = document.pages[0].tables[0]
        second_table = document.pages[1].tables[0]

        assert first_table.id == "0"
        assert second_table.id == "1"
        assert second_table.continuation_of == "0"

    def test_integration(self, pdf_file: Path, output_dir: Path):
        observer = DebugImageObserver(output_dir=output_dir)

        pipeline = DocumentBuildPipeline(debug_image=observer)

        document = pipeline.build(pdf_file.read_bytes())

        text = document.to_markdown()

        # save results for manual inspection
        output_dir.joinpath("document.md").write_text(text)
