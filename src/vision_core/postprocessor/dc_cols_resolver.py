"""Определяет дебет/кредит колонки и записывает их в table.dc_cols."""

from __future__ import annotations

from vision_core.entities.page import Page
from vision_core.entities.table import Table
from vision_core.exceptions import DcColsInvalidPositionError, DcColsNotFoundError

_DEBIT_KEYWORDS = {"дебет"}
_CREDIT_KEYWORDS = {"кредит"}
_HEADER_SCAN_ROWS = 4


def is_dc_header(value: str) -> bool:
    return _is_dc_header(value)


def _is_dc_header(value: str) -> bool:
    normalized = value.strip().lower()
    return any(kw in normalized for kw in _DEBIT_KEYWORDS | _CREDIT_KEYWORDS)


class DcColsResolver:
    """Заполняет table.dc_cols для всех таблиц документа.

    Для корневых таблиц — детектирует по заголовку, при частичном совпадении
    восстанавливает недостающую колонку по позиции (дебет слева, кредит справа).
    Для таблиц-продолжений — наследует dc_cols из корня цепочки.

    Порядок в пайплайне: после continuation_linker, до row_splitter.
    """

    def resolve(self, pages: list[Page]) -> None:
        """Заполняет dc_cols in-place для всех таблиц.

        Args:
            pages: Страницы документа после continuation_linker.link().

        Raises:
            DcColsNotFoundError: Если в корневой таблице не найдены ни дебет, ни кредит.
            DcColsInvalidPositionError: Если восстановленная по позиции колонка выходит за пределы таблицы.
        """
        all_tables: dict[str, Table] = {t.id: t for p in pages for t in p.tables}

        for table in all_tables.values():
            if table.continuation_of is None:
                table.dc_cols = _detect_dc_cols(table)

        for table in all_tables.values():
            if table.continuation_of is not None and not table.dc_cols:
                root = _find_root(table, all_tables)
                table.dc_cols = root.dc_cols.copy()


def _detect_dc_cols(table: Table) -> set[int]:
    debit_cols: set[int] = set()
    credit_cols: set[int] = set()

    for row in table.get_rows()[:_HEADER_SCAN_ROWS]:
        for cell in row:
            if not cell.value:
                continue
            normalized = cell.value.strip().lower()
            if any(kw in normalized for kw in _DEBIT_KEYWORDS):
                debit_cols.add(cell.col)
            elif any(kw in normalized for kw in _CREDIT_KEYWORDS):
                credit_cols.add(cell.col)

    if debit_cols and credit_cols:
        return debit_cols | credit_cols

    if debit_cols and not credit_cols:
        debit_col = min(debit_cols)
        credit_col = debit_col + 1
        if credit_col >= table.num_cols:
            raise DcColsInvalidPositionError(table_id=table.id, col=credit_col, num_cols=table.num_cols)
        return debit_cols | {credit_col}

    if credit_cols and not debit_cols:
        credit_col = min(credit_cols)
        debit_col = credit_col - 1
        if debit_col < 0:
            raise DcColsInvalidPositionError(table_id=table.id, col=debit_col, num_cols=table.num_cols)
        return {debit_col} | credit_cols

    raise DcColsNotFoundError(table_id=table.id)


def _find_root(table: Table, all_tables: dict[str, Table]) -> Table:
    visited: set[str] = set()
    current = table
    while current.continuation_of is not None:
        if current.continuation_of in visited:
            break
        visited.add(current.id)
        parent = all_tables.get(current.continuation_of)
        if parent is None:
            break
        current = parent
    return current
