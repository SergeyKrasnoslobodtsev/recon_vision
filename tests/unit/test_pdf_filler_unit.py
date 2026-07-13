import numpy as np
import pymupdf
import pytest

from app.application.dto.fill_reconciliation_act import FillReconciliationActCommand
from app.domain.entities.ledger_entry import LedgerEntry, RowReference
from app.domain.entities.process import ProcessState
from app.infrastructure.services.pdf_fill.render import format_amount
from app.infrastructure.services.pdf_filler import DocumentPdfFiller
from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.entities.table import Table


def _build_pdf_bytes() -> bytes:
    document = pymupdf.open()
    document.new_page(width=320, height=180)
    try:
        return document.tobytes()
    finally:
        document.close()


def _build_document_payload() -> Document:
    cells = [
        Cell(
            row=0,
            col=0,
            bbox=BBox(x_min=10, y_min=20, x_max=70, y_max=40),
            value="Операция",
        ),
        Cell(
            row=0,
            col=1,
            colspan=2,
            bbox=BBox(x_min=70, y_min=20, x_max=170, y_max=40),
            value="От продавца",
        ),
        Cell(
            row=0,
            col=3,
            colspan=2,
            bbox=BBox(x_min=170, y_min=20, x_max=270, y_max=40),
            value="От покупателя",
        ),
        Cell(row=1, col=0, bbox=BBox(x_min=10, y_min=40, x_max=70, y_max=60), value="Строка"),
        Cell(row=1, col=1, bbox=BBox(x_min=70, y_min=40, x_max=120, y_max=60), value="Дебет"),
        Cell(row=1, col=2, bbox=BBox(x_min=120, y_min=40, x_max=170, y_max=60), value="Кредит"),
        Cell(row=1, col=3, bbox=BBox(x_min=170, y_min=40, x_max=220, y_max=60), value="Дебет"),
        Cell(row=1, col=4, bbox=BBox(x_min=220, y_min=40, x_max=270, y_max=60), value="Кредит"),
        Cell(
            row=2,
            col=0,
            bbox=BBox(x_min=10, y_min=60, x_max=70, y_max=80),
            blobs=[BBox(x_min=18, y_min=65, x_max=62, y_max=74)],
        ),
        Cell(
            row=2,
            col=1,
            bbox=BBox(x_min=70, y_min=60, x_max=120, y_max=80),
            blobs=[BBox(x_min=78, y_min=65, x_max=112, y_max=74)],
        ),
        Cell(
            row=2,
            col=2,
            bbox=BBox(x_min=120, y_min=60, x_max=170, y_max=80),
            blobs=[BBox(x_min=128, y_min=65, x_max=162, y_max=74)],
        ),
        Cell(
            row=2,
            col=3,
            bbox=BBox(x_min=170, y_min=60, x_max=220, y_max=80),
            blobs=[BBox(x_min=178, y_min=65, x_max=212, y_max=74)],
        ),
        Cell(
            row=2,
            col=4,
            bbox=BBox(x_min=220, y_min=60, x_max=270, y_max=80),
            blobs=[BBox(x_min=228, y_min=65, x_max=262, y_max=74)],
        ),
    ]
    table = Table(
        id="table-1",
        bbox=BBox(x_min=10, y_min=20, x_max=270, y_max=80),
        num_rows=3,
        num_cols=5,
        cells=cells,
        dc_cols={1, 2, 3, 4},
    )
    page = Page(
        page_number=0,
        tables=[table],
        metadata={
            "source_image_shape": [180, 240],
            "image_shape": [180, 320],
            "page_size": [320.0, 180.0],
            "orientation_deg": 0.0,
            "deskew_angle_deg": 0.0,
        },
    )
    return Document(pages=[page], source_hash="test-hash")


def _build_two_page_pdf_bytes() -> bytes:
    document = pymupdf.open()
    document.new_page(width=320, height=180)
    document.new_page(width=320, height=180)
    try:
        return document.tobytes()
    finally:
        document.close()


def _build_two_page_document_payload() -> Document:
    first_page = _build_document_payload().pages[0]
    second_page = Page(
        page_number=1,
        tables=[],
        metadata={
            "source_image_shape": [180, 320],
            "image_shape": [180, 320],
            "page_size": [320.0, 180.0],
            "orientation_deg": 0.0,
            "deskew_angle_deg": 0.0,
        },
    )
    return Document(pages=[first_page, second_page], source_hash="test-hash")


def _render_page(pdf_bytes: bytes, page_number: int) -> np.ndarray:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = document[page_number].get_pixmap(dpi=72, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    finally:
        document.close()


def _render_first_page(pdf_bytes: bytes) -> np.ndarray:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = document[0].get_pixmap(dpi=72, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    finally:
        document.close()


def _region_has_drawing(image: np.ndarray, bbox: BBox) -> bool:
    x_min, y_min, x_max, y_max = bbox.to_tuple()
    region = image[y_min:y_max, x_min:x_max]
    return region.size > 0 and np.any(region < 250)


class TestDocumentPdfFiller:
    """Проверяет заполнение PDF на основе канонического документа."""

    def test_format_amount_uses_spaces_and_comma(self):
        assert format_amount(0) == "0,00"
        assert format_amount(1200.5) == "1 200,50"
        assert format_amount(1234567.89) == "1 234 567,89"

    @pytest.mark.asyncio
    async def test_fill_draws_only_debit_credit_values_and_comments(self):
        filler = DocumentPdfFiller()
        document_payload = _build_document_payload()
        process_state = ProcessState(
            process_id="process-123",
            source_pdf=_build_pdf_bytes(),
            document_payload=document_payload,
        )
        command = FillReconciliationActCommand(
            process_id="process-123",
            comments="Проверено вручную",
            debit=[
                LedgerEntry(
                    record="",
                    value=1200.5,
                    row_reference=RowReference(id_table="table-1", id_row="2", id_col=1, buyer_col=3),
                )
            ],
            credit=[
                LedgerEntry(
                    record="",
                    value=900.0,
                    row_reference=RowReference(id_table="table-1", id_row="2", id_col=2, buyer_col=4),
                )
            ],
        )

        filled_pdf_bytes = await filler.fill(process_state, command)
        rendered_page = _render_first_page(filled_pdf_bytes)
        table = document_payload.pages[0].tables[0]

        assert not _region_has_drawing(rendered_page, table.get_cell(0, 0).bbox)
        assert not _region_has_drawing(rendered_page, table.get_cell(2, 1).bbox)
        assert not _region_has_drawing(rendered_page, table.get_cell(2, 2).bbox)
        assert _region_has_drawing(rendered_page, table.get_cell(2, 3).bbox)
        assert _region_has_drawing(rendered_page, table.get_cell(2, 4).bbox)
        assert _region_has_drawing(rendered_page, BBox(x_min=180, y_min=140, x_max=319, y_max=179))

    @pytest.mark.asyncio
    async def test_fill_draws_comments_on_page_with_last_table(self):
        filler = DocumentPdfFiller()
        document_payload = _build_two_page_document_payload()
        process_state = ProcessState(
            process_id="process-123",
            source_pdf=_build_two_page_pdf_bytes(),
            document_payload=document_payload,
        )

        filled_pdf_bytes = await filler.fill(
            process_state,
            FillReconciliationActCommand(process_id="process-123", comments="Комментарий"),
        )
        first_page = _render_page(filled_pdf_bytes, 0)
        second_page = _render_page(filled_pdf_bytes, 1)
        comments_bbox = BBox(x_min=180, y_min=140, x_max=319, y_max=179)

        assert _region_has_drawing(first_page, comments_bbox)
        assert not _region_has_drawing(second_page, comments_bbox)

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

    @pytest.mark.asyncio
    async def test_fill_returns_rotated_pdf_when_page_was_rotated(self):
        filler = DocumentPdfFiller()
        document_payload = _build_document_payload()
        document_payload.pages[0].metadata.update(
            {
                "source_image_shape": [180, 240],
                "image_shape": [320, 180],
                "page_size": [180.0, 320.0],
                "orientation_deg": 90.0,
                "deskew_angle_deg": 0.0,
            }
        )
        process_state = ProcessState(
            process_id="process-123",
            source_pdf=_build_pdf_bytes(),
            document_payload=document_payload,
        )

        filled_pdf_bytes = await filler.fill(process_state, FillReconciliationActCommand(process_id="process-123"))
        filled_document = pymupdf.open(stream=filled_pdf_bytes, filetype="pdf")
        try:
            page_rect = filled_document[0].rect
        finally:
            filled_document.close()

        assert page_rect.width == pytest.approx(180.0)
        assert page_rect.height == pytest.approx(320.0)
