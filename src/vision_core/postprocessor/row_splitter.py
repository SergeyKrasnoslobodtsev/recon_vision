"""Постпроцессор для разбивки строк таблицы без горизонтальных линий."""

from __future__ import annotations

from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.page import Page
from vision_core.entities.table import Table


class RowSplitter:
    """Разбивает строки таблицы, содержащие несколько значений в ячейках дебет/кредит.

    Индикатор необходимости разбивки: максимальное число blobs в дебет/кредит ячейках
    одной строки n > 1. Разбивка применяется синхронно ко всем ячейкам строки.

    Порядок в пайплайне: после continuation_linker, до debit_credit_processor.
    """

    def split(self, pages: list[Page]) -> None:
        """Разбивает строки таблиц in-place (заменяет table объекты на новые).

        Args:
            pages: Страницы документа после dc_cols_resolver.resolve().
        """
        for page in pages:
            page.tables = [self._split_table(table, table.dc_cols) for table in page.tables]

    def _split_table(self, table: Table, dc_cols: set[int]) -> Table:
        """Разбивает строки одной таблицы, возвращает новый объект Table."""
        if not dc_cols:
            return table

        header_row = table.get_dc_header_row()
        new_cells: list[Cell] = []
        row_offset = 0

        for row_idx in range(table.num_rows):
            row_cells = [c for c in table.cells if c.row == row_idx]

            if row_idx <= header_row:
                new_cells.extend(row_cells)
                continue

            n = self._count_splits(row_cells, dc_cols)

            if n <= 1:
                for cell in row_cells:
                    new_cells.append(_copy_cell(cell, row=cell.row + row_offset))
            else:
                boundaries = self._calc_boundaries(row_cells, dc_cols, n)
                for cell in row_cells:
                    new_cells.extend(self._split_cell(cell, n, boundaries, row_offset))
                row_offset += n - 1

        return Table(
            id=table.id,
            bbox=table.bbox,
            num_rows=table.num_rows + row_offset,
            num_cols=table.num_cols,
            cells=new_cells,
            start_page=table.start_page,
            end_page=table.end_page,
            continuation_of=table.continuation_of,
            dc_cols=table.dc_cols.copy(),
        )

    def _count_splits(self, row_cells: list[Cell], dc_cols: set[int]) -> int:
        """Возвращает число строк для разбивки по количеству blobs в DC-ячейках с фактическими значениями."""
        counts = [
            len(cell.blobs)
            for cell in row_cells
            if cell.col in dc_cols and cell.blobs and cell.value and cell.value.strip()
        ]
        return max(counts, default=1)

    def _calc_boundaries(
        self,
        row_cells: list[Cell],
        dc_cols: set[int],
        n: int,
    ) -> list[float]:
        """Вычисляет границы для разбивки строки на подстроки."""
        for cell in row_cells:
            if cell.col not in dc_cols or len(cell.blobs) != n:
                continue
            if not cell.value or not cell.value.strip():
                continue
            blobs = sorted(cell.blobs, key=lambda b: b.y_min)
            return [(blobs[i].y_max + blobs[i + 1].y_min) / 2 for i in range(n - 1)]

        # fallback: равномерное деление bbox первой DC-ячейки с контентом
        for cell in row_cells:
            if cell.col in dc_cols and cell.value and cell.value.strip():
                step = cell.bbox.height / n
                return [cell.bbox.y_min + step * (i + 1) for i in range(n - 1)]

        return []

    def _split_cell(
        self,
        cell: Cell,
        n: int,
        boundaries: list[float],
        row_offset: int,
    ) -> list[Cell]:
        """Разбивает одну ячейку на n подъячеек по границам, возвращает список новых ячеек."""
        lines = (cell.value or "").split("\n")
        blobs = cell.blobs

        pairs: list[tuple[BBox | None, str]] = []
        if len(blobs) == len(lines):
            pairs = list(zip(blobs, lines, strict=False))
        else:
            pairs = [(b, "") for b in blobs] + [(None, l) for l in lines[len(blobs) :]]

        sub_rows: list[list[tuple[BBox | None, str]]] = [[] for _ in range(n)]
        for blob, text in pairs:
            idx = _assign_subrow(blob, boundaries) if blob is not None else 0
            sub_rows[idx].append((blob, text))

        result: list[Cell] = []
        for i, sub_pairs in enumerate(sub_rows):
            y_min = cell.bbox.y_min if i == 0 else boundaries[i - 1]
            y_max = cell.bbox.y_max if i == n - 1 else boundaries[i]
            sub_blobs = [p[0] for p in sub_pairs if p[0] is not None]
            sub_texts = [p[1] for p in sub_pairs]
            result.append(
                Cell(
                    row=cell.row + row_offset + i,
                    col=cell.col,
                    colspan=cell.colspan,
                    rowspan=1,
                    value="\n".join(sub_texts) if sub_texts else None,
                    bbox=BBox(
                        x_min=cell.bbox.x_min,
                        y_min=y_min,
                        x_max=cell.bbox.x_max,
                        y_max=y_max,
                    ),
                    blobs=sub_blobs,
                )
            )
        return result


def _copy_cell(cell: Cell, row: int) -> Cell:
    """Создает копию ячейки с новым row-индексом, остальные атрибуты сохраняются."""
    return Cell(
        row=row,
        col=cell.col,
        colspan=cell.colspan,
        rowspan=cell.rowspan,
        value=cell.value,
        bbox=cell.bbox,
        blobs=cell.blobs,
    )


def _assign_subrow(blob: BBox, boundaries: list[float]) -> int:
    """Определяет индекс подстроки для данного blob по его вертикальному положению."""
    cy = (blob.y_min + blob.y_max) / 2
    for i, boundary in enumerate(boundaries):
        if cy < boundary:
            return i
    return len(boundaries)
