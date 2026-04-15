"""Постпроцессор для нормализации денежных значений в дебет/кредит колонках."""

from __future__ import annotations

from loguru import logger

from vision_core.entities.page import Page
from vision_core.entities.table import Table
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
        for page in pages:
            for table in page.tables:
                if not table.dc_cols:
                    continue
                logger.debug(f"Колонки дебет/кредит таблицы {table.id}: {table.dc_cols}")
                self._normalize(table)

    def _normalize(self, table: Table) -> None:
        header_row = table.get_dc_header_row()
        for cell in table.cells:
            if cell.col not in table.dc_cols:
                continue
            if cell.row <= header_row:
                continue
            try:
                cell.value = f"{parse_currency(cell.value):.2f}"
            except ValueError:
                logger.warning(f"Не удалось распарсить {cell.value} в таблице {table.id} row={cell.row} col={cell.col}")
