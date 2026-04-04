import pytest

from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.page import Page
from vision_core.entities.table import Table
from vision_core.postprocessor.row_splitter import RowSplitter


def _bbox(x0=0, y0=0, x1=100, y1=100) -> BBox:
    return BBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def _blob(y0, y1) -> BBox:
    return BBox(x_min=10, y_min=y0, x_max=90, y_max=y1)


def _make_table(table_id, cells, num_rows, num_cols, continuation_of=None):
    return Table(
        id=table_id,
        bbox=_bbox(0, 0, 400, 400),
        num_rows=num_rows,
        num_cols=num_cols,
        cells=cells,
        continuation_of=continuation_of,
    )


def _make_page(*tables):
    return Page(tables=list(tables))


class TestRowSplitter:
    def test_no_split_when_single_blob_per_dc_cell(self):
        table = _make_table(
            "t0", num_rows=2, num_cols=2,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(0, 0, 100, 20), value="Дебет"),
                Cell(row=0, col=1, bbox=_bbox(100, 0, 200, 20), value="Кредит"),
                Cell(row=1, col=0, bbox=_bbox(0, 20, 100, 50), value="1 000,00",
                     blobs=[_blob(25, 45)]),
                Cell(row=1, col=1, bbox=_bbox(100, 20, 200, 50), value="-",
                     blobs=[_blob(25, 45)]),
            ],
        )
        RowSplitter().split([_make_page(table)])

        assert table.num_rows == 2

    def test_splits_row_with_three_blobs(self):
        # строка row=1 содержит 3 значения в col=0 (Дебет)
        table = _make_table(
            "t0", num_rows=2, num_cols=2,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(0, 0, 100, 20), value="Дебет"),
                Cell(row=0, col=1, bbox=_bbox(100, 0, 200, 20), value="Кредит"),
                Cell(row=1, col=0, bbox=_bbox(0, 20, 100, 110),
                     value="100,00\n200,00\n300,00",
                     blobs=[_blob(22, 38), _blob(52, 68), _blob(82, 98)]),
                Cell(row=1, col=1, bbox=_bbox(100, 20, 200, 110),
                     value="-\n-\n-",
                     blobs=[_blob(22, 38), _blob(52, 68), _blob(82, 98)]),
            ],
        )
        page = _make_page(table)
        RowSplitter().split([page])
        result = page.tables[0]

        assert result.num_rows == 4  # 1 header + 3 data rows
        data_cells_col0 = [c for c in result.cells if c.row >= 1 and c.col == 0]
        assert len(data_cells_col0) == 3
        assert data_cells_col0[0].value == "100,00"
        assert data_cells_col0[1].value == "200,00"
        assert data_cells_col0[2].value == "300,00"

    def test_non_dc_column_split_synced_with_dc(self):
        # col=0 - Документ (текст), col=1 - Дебет
        table = _make_table(
            "t0", num_rows=2, num_cols=2,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(0, 0, 100, 20), value="Документ"),
                Cell(row=0, col=1, bbox=_bbox(100, 0, 200, 20), value="Дебет"),
                Cell(row=1, col=0, bbox=_bbox(0, 20, 100, 110),
                     value="Акт №1\nАкт №2",
                     blobs=[_blob(22, 38), _blob(52, 68)]),
                Cell(row=1, col=1, bbox=_bbox(100, 20, 200, 110),
                     value="500,00\n700,00",
                     blobs=[_blob(22, 38), _blob(52, 68)]),
            ],
        )
        page = _make_page(table)
        RowSplitter().split([page])
        result = page.tables[0]

        assert result.num_rows == 3
        doc_cells = sorted(
            [c for c in result.cells if c.row >= 1 and c.col == 0],
            key=lambda c: c.row,
        )
        assert doc_cells[0].value == "Акт №1"
        assert doc_cells[1].value == "Акт №2"

    def test_row_indices_recalculated_correctly(self):
        # две строки данных, первая разбивается на 2
        table = _make_table(
            "t0", num_rows=3, num_cols=1,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(0, 0, 100, 20), value="Дебет"),
                Cell(row=1, col=0, bbox=_bbox(0, 20, 100, 80),
                     value="100,00\n200,00",
                     blobs=[_blob(25, 40), _blob(55, 70)]),
                Cell(row=2, col=0, bbox=_bbox(0, 80, 100, 110),
                     value="300,00",
                     blobs=[_blob(85, 100)]),
            ],
        )
        page = _make_page(table)
        RowSplitter().split([page])
        result = page.tables[0]

        assert result.num_rows == 4
        rows = sorted({c.row for c in result.cells})
        assert rows == [0, 1, 2, 3]

    def test_empty_cell_duplicated_for_each_subrow(self):
        # пустая ячейка (без blobs) дублируется по числу строк
        table = _make_table(
            "t0", num_rows=2, num_cols=2,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(0, 0, 100, 20), value="Документ"),
                Cell(row=0, col=1, bbox=_bbox(100, 0, 200, 20), value="Дебет"),
                Cell(row=1, col=0, bbox=_bbox(0, 20, 100, 80), value=None, blobs=[]),
                Cell(row=1, col=1, bbox=_bbox(100, 20, 200, 80),
                     value="100,00\n200,00",
                     blobs=[_blob(25, 40), _blob(55, 70)]),
            ],
        )
        page = _make_page(table)
        RowSplitter().split([page])
        result = page.tables[0]

        empty_cells = [c for c in result.cells if c.row >= 1 and c.col == 0]
        assert len(empty_cells) == 2

    def test_continuation_table_inherits_dc_cols(self):
        root = _make_table(
            "t0", num_rows=2, num_cols=1,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(0, 0, 100, 20), value="Дебет"),
                Cell(row=1, col=0, bbox=_bbox(0, 20, 100, 80),
                     value="100,00\n200,00",
                     blobs=[_blob(25, 40), _blob(55, 70)]),
            ],
        )
        cont = _make_table(
            "t1", num_rows=1, num_cols=1,
            continuation_of="t0",
            cells=[
                Cell(row=0, col=0, bbox=_bbox(0, 0, 100, 80),
                     value="300,00\n400,00",
                     blobs=[_blob(10, 30), _blob(50, 70)]),
            ],
        )
        page1 = _make_page(root)
        page2 = _make_page(cont)
        RowSplitter().split([page1, page2])

        assert page2.tables[0].num_rows == 2

    def test_table_without_dc_header_not_split(self):
        table = _make_table(
            "t0", num_rows=2, num_cols=1,
            cells=[
                Cell(row=0, col=0, bbox=_bbox(0, 0, 100, 20), value="Документ"),
                Cell(row=1, col=0, bbox=_bbox(0, 20, 100, 80),
                     value="Акт №1\nАкт №2",
                     blobs=[_blob(25, 40), _blob(55, 70)]),
            ],
        )
        page = _make_page(table)
        RowSplitter().split([page])

        assert page.tables[0].num_rows == 2

    def test_empty_pages(self):
        RowSplitter().split([])  # не бросает исключений
