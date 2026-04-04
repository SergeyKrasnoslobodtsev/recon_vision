"""Постпроцессор для нормализации денежных значений в дебет/кредит колонках."""

from __future__ import annotations

from loguru import logger

from vision_core.entities.cell import Cell
from vision_core.entities.page import Page
from vision_core.postprocessor.dc_cols import build_dc_cols_map, is_dc_header
from vision_core.utils.currency import parse_currency


class DebitCreditProcessor:
    """Нормализует денежные значения в дебет/кредит ячейках.

    Перезаписывает cell.value в формат «ЧИСЛО.КК» (например «47761.70»).
    Заголовочные ячейки не трогает. Пустые и прочерк -> «0.00».
    """

    def process(self, pages: list[Page]) -> None:
        """Нормализует денежные ячейки in-place.

        Args:
            pages: Страницы документа после row_splitter.split().
        """
        dc_cols_map = build_dc_cols_map(pages)
        logger.debug(f"Колонки дебет/кредит для нормализации: {dc_cols_map}")

        for page in pages:
            for table in page.tables:
                dc_cols = dc_cols_map.get(table.id, set())
                if not dc_cols:
                    continue
                self._normalize(table.id, table.cells, dc_cols)

    def _find_dc_header_row(self, cells: list[Cell], dc_cols: set[int]) -> int:
        """Возвращает row-индекс строки с заголовками дебет/кредит, или -1."""
        for cell in cells:
            if cell.col in dc_cols and cell.value and is_dc_header(cell.value):
                return cell.row
        return -1

    def _normalize(self, table_id: str, cells: list[Cell], dc_cols: set[int]) -> None:
        header_row = self._find_dc_header_row(cells, dc_cols)
        for cell in cells:
            if cell.col not in dc_cols:
                continue
            if cell.row <= header_row:
                continue
            try:
                cell.value = f"{parse_currency(cell.value):.2f}"
            except ValueError:
                logger.warning(f"Не удалось распарсить {cell.value} в таблице {table_id} row={cell.row} col={cell.col}")
