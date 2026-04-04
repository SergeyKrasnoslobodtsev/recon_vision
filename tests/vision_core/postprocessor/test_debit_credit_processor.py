import pytest

from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.page import Page
from vision_core.entities.table import Table
from vision_core.postprocessor.debit_credit_processor import DebitCreditProcessor


def _bbox() -> BBox:
    return BBox(x_min=0, y_min=0, x_max=10, y_max=10)


def _make_table(
    table_id: str,
    cells: list[Cell],
    num_rows: int,
    num_cols: int,
    continuation_of: str | None = None,
) -> Table:
    return Table(
        id=table_id,
        bbox=_bbox(),
        num_rows=num_rows,
        num_cols=num_cols,
        cells=cells,
        continuation_of=continuation_of,
    )


def _make_page(*tables: Table) -> Page:
    return Page(tables=list(tables))


class TestDebitCreditProcessor:
    def test_normalizes_debit_credit_cells(self):
        # Таблица: заголовок row=0, данные row=1
        # col0=Документ, col1=Дебет, col2=Кредит
        table = _make_table(
            "t0",
            num_rows=2,
            num_cols=3,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(), value="Документ"),
                Cell(row=0, col=1, bbox=_bbox(), value="Дебет"),
                Cell(row=0, col=2, bbox=_bbox(), value="Кредит"),
                Cell(row=1, col=0, bbox=_bbox(), value="Реализация товаров"),
                Cell(row=1, col=1, bbox=_bbox(), value="47 761,70"),
                Cell(row=1, col=2, bbox=_bbox(), value="-"),
            ],
        )

        DebitCreditProcessor().process([_make_page(table)])

        assert table.cells[4].value == "47761.70"
        assert table.cells[5].value == "0.00"

    def test_header_cells_are_not_normalized(self):
        table = _make_table(
            "t0",
            num_rows=2,
            num_cols=2,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(), value="Дебет"),
                Cell(row=0, col=1, bbox=_bbox(), value="Кредит"),
                Cell(row=1, col=0, bbox=_bbox(), value="1 000,00"),
                Cell(row=1, col=1, bbox=_bbox(), value="-"),
            ],
        )

        DebitCreditProcessor().process([_make_page(table)])

        assert table.cells[0].value == "Дебет"
        assert table.cells[1].value == "Кредит"

    def test_non_dc_columns_are_not_touched(self):
        table = _make_table(
            "t0",
            num_rows=2,
            num_cols=2,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(), value="Документ"),
                Cell(row=0, col=1, bbox=_bbox(), value="Дебет"),
                Cell(row=1, col=0, bbox=_bbox(), value="Реализация"),
                Cell(row=1, col=1, bbox=_bbox(), value="1 000,00"),
            ],
        )

        DebitCreditProcessor().process([_make_page(table)])

        assert table.cells[2].value == "Реализация"

    def test_continuation_table_inherits_dc_cols(self):
        root = _make_table(
            "t0",
            num_rows=2,
            num_cols=2,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(), value="Дебет"),
                Cell(row=0, col=1, bbox=_bbox(), value="Кредит"),
                Cell(row=1, col=0, bbox=_bbox(), value="1 000,00"),
                Cell(row=1, col=1, bbox=_bbox(), value="-"),
            ],
        )
        continuation = _make_table(
            "t1",
            num_rows=1,
            num_cols=2,
            continuation_of="t0",
            cells=[
                Cell(row=0, col=0, bbox=_bbox(), value="2 500,00"),
                Cell(row=0, col=1, bbox=_bbox(), value="800,50"),
            ],
        )

        DebitCreditProcessor().process([_make_page(root), _make_page(continuation)])

        assert continuation.cells[0].value == "2500.00"
        assert continuation.cells[1].value == "800.50"

    def test_table_without_dc_header_is_skipped(self):
        table = _make_table(
            "t0",
            num_rows=2,
            num_cols=2,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(), value="Дата"),
                Cell(row=0, col=1, bbox=_bbox(), value="Документ"),
                Cell(row=1, col=0, bbox=_bbox(), value="01.01.2024"),
                Cell(row=1, col=1, bbox=_bbox(), value="Акт №1"),
            ],
        )

        DebitCreditProcessor().process([_make_page(table)])

        assert table.cells[2].value == "01.01.2024"
        assert table.cells[3].value == "Акт №1"

    def test_case_insensitive_header_detection(self):
        table = _make_table(
            "t0",
            num_rows=2,
            num_cols=1,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(), value="ДЕБЕТ"),
                Cell(row=1, col=0, bbox=_bbox(), value="5 000,00"),
            ],
        )

        DebitCreditProcessor().process([_make_page(table)])

        assert table.cells[1].value == "5000.00"

    def test_rows_above_dc_header_are_not_normalized(self):
        # row=0: "по данным Продавца" (над заголовком DC)
        # row=1: "Дебет", "Кредит"
        # row=2: данные
        table = _make_table(
            "t0",
            num_rows=3,
            num_cols=2,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(), value="по данным Продавца"),
                Cell(row=0, col=1, bbox=_bbox(), value="по данным Покупателя"),
                Cell(row=1, col=0, bbox=_bbox(), value="Дебет"),
                Cell(row=1, col=1, bbox=_bbox(), value="Кредит"),
                Cell(row=2, col=0, bbox=_bbox(), value="51 043 498,57"),
                Cell(row=2, col=1, bbox=_bbox(), value="-"),
            ],
        )

        DebitCreditProcessor().process([_make_page(table)])

        assert table.cells[0].value == "по данным Продавца"
        assert table.cells[1].value == "по данным Покупателя"
        assert table.cells[2].value == "Дебет"
        assert table.cells[3].value == "Кредит"
        assert table.cells[4].value == "51043498.57"
        assert table.cells[5].value == "0.00"

    def test_empty_pages(self):
        DebitCreditProcessor().process([])  # не бросает исключений
