import pymupdf
import pytest

from app.application.dto.fill_reconciliation_act import FillReconciliationActCommand
from app.domain.entities.ledger_entry import LedgerEntry, RowReference
from app.domain.entities.process import ProcessState
from app.infrastructure.services.pdf_filler import DocumentPdfFiller
from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.entities.table import Table


def _build_pdf_bytes() -> bytes:
    document = pymupdf.open()
    document.new_page(width=240, height=180)
    try:
        return document.tobytes()
    finally:
        document.close()


def _build_document_payload() -> Document:
    cells = [
        Cell(row=0, col=0, bbox=BBox(x_min=10, y_min=20, x_max=70, y_max=40)),
        Cell(row=0, col=1, bbox=BBox(x_min=70, y_min=20, x_max=170, y_max=40)),
        Cell(row=0, col=2, bbox=BBox(x_min=170, y_min=20, x_max=230, y_max=40)),
    ]
    table = Table(
        id="table-1",
        bbox=BBox(x_min=10, y_min=20, x_max=230, y_max=40),
        num_rows=1,
        num_cols=3,
        cells=cells,
    )
    page = Page(
        page_number=0,
        tables=[table],
        metadata={
            "image_shape": [180, 240],
            "page_size": [240.0, 180.0],
        },
    )
    return Document(pages=[page], source_hash="test-hash")


class TestDocumentPdfFiller:
    """Проверяет заполнение PDF на основе канонического документа."""

    @pytest.mark.asyncio
    async def test_fill_writes_entry_data_and_comments_to_pdf(self):
        filler = DocumentPdfFiller()
        process_state = ProcessState(
            process_id="process-123",
            source_pdf=_build_pdf_bytes(),
            document_payload=_build_document_payload(),
        )
        command = FillReconciliationActCommand(
            process_id="process-123",
            comments="Проверено вручную",
            debit=[
                LedgerEntry(
                    record="Реализация",
                    value=1200.5,
                    date="2025-01-15",
                    row_reference=RowReference(id_table="table-1", id_row="0"),
                )
            ],
            credit=[],
        )

        filled_pdf_bytes = await filler.fill(process_state, command)

        filled_document = pymupdf.open(stream=filled_pdf_bytes, filetype="pdf")
        try:
            text = filled_document[0].get_text()
        finally:
            filled_document.close()

        assert "2025-01-15" in text
        assert "Реализация" in text
        assert "1200.5" in text
        assert "Комментарии:" in text
        assert "Проверено вручную" in text

    @pytest.mark.asyncio
    async def test_fill_raises_when_document_payload_is_missing(self):
        filler = DocumentPdfFiller()
        process_state = ProcessState(
            process_id="process-123",
            source_pdf=_build_pdf_bytes(),
            document_payload=None,
        )
        command = FillReconciliationActCommand(process_id="process-123")

        with pytest.raises(ValueError, match="канонический Document"):
            await filler.fill(process_state, command)
