from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.table import Table
from vision_core.ocr.base import OcrResult
from vision_core.postprocessor.cell_text_filler import CellTextFiller


def _make_table() -> Table:
    return Table(
        id="0",
        bbox=BBox(x_min=0, y_min=0, x_max=20, y_max=10),
        num_rows=1,
        num_cols=2,
        cells=[
            Cell(row=0, col=0, bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10)),
            Cell(row=0, col=1, bbox=BBox(x_min=10, y_min=0, x_max=20, y_max=10)),
        ],
    )


class TestCellTextFiller:
    def test_fill_cells_assigns_text_by_overlap(self):
        table = Table(
            id="0",
            bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10),
            num_rows=1,
            num_cols=1,
            cells=[Cell(row=0, col=0, bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10))],
        )
        filler = CellTextFiller(cell_overlap_threshold=0.4)
        ocr_results = [
            OcrResult(text="value", confidence=0.95, bbox=(2, 2, 20, 8)),
        ]

        filler.fill_cells([table], ocr_results)

        assert table.cells[0].value == "value"
        assert table.cells[0].blobs == [BBox(x_min=2, y_min=2, x_max=20, y_max=8)]

    def test_fill_cells_skips_ambiguous_bbox_between_cells(self):
        table = _make_table()
        filler = CellTextFiller(cell_overlap_threshold=0.5)
        ocr_results = [
            OcrResult(text="shared", confidence=0.95, bbox=(5, 1, 15, 9)),
        ]

        filler.fill_cells([table], ocr_results)

        assert table.cells[0].value == ""
        assert table.cells[1].value == ""
        assert table.cells[0].blobs == []
        assert table.cells[1].blobs == []

    def test_fill_cells_resets_previous_assignments(self):
        table = Table(
            id="0",
            bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10),
            num_rows=1,
            num_cols=1,
            cells=[Cell(row=0, col=0, bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10))],
        )
        filler = CellTextFiller()

        filler.fill_cells(
            [table],
            [OcrResult(text="first", confidence=0.95, bbox=(1, 1, 5, 5))],
        )
        filler.fill_cells(
            [table],
            [OcrResult(text="second", confidence=0.95, bbox=(2, 2, 6, 6))],
        )

        assert table.cells[0].value == "second"
        assert table.cells[0].blobs == [BBox(x_min=2, y_min=2, x_max=6, y_max=6)]

    def test_exclude_table_text_filters_overlap_without_center_hit(self):
        table = _make_table()
        filler = CellTextFiller(table_overlap_threshold=0.4)
        ocr_results = [
            OcrResult(text="table-text", confidence=0.95, bbox=(12, 1, 30, 9)),
            OcrResult(text="paragraph", confidence=0.95, bbox=(30, 1, 40, 9)),
        ]

        filtered = filler.exclude_table_text(ocr_results, [table])

        assert [item.text for item in filtered] == ["paragraph"]
