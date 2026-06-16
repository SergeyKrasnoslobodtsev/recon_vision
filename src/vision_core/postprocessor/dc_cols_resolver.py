"""Определяет дебет/кредит колонки"""

from __future__ import annotations

from loguru import logger

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
        all_tables = {t.id: t for p in pages for t in p.tables}
        children = _build_continuation_children(all_tables)

        for table in all_tables.values():
            if table.continuation_of is None:
                table.dc_cols = _detect_root_dc_cols(table, children)

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

    if not debit_cols and not credit_cols:
        raise DcColsNotFoundError(table_id=table.id)

    # Кредит найден — дебет должен быть слева от каждого кредита
    for col in sorted(credit_cols):
        partner = col - 1
        if partner not in credit_cols and partner not in debit_cols:
            if partner < 0:
                raise DcColsInvalidPositionError(table_id=table.id, col=partner, num_cols=table.num_cols)
            debit_cols.add(partner)

    # Дебет найден — кредит должен быть справа от каждого дебета
    for col in sorted(debit_cols):
        partner = col + 1
        if partner not in debit_cols and partner not in credit_cols:
            if partner >= table.num_cols:
                raise DcColsInvalidPositionError(table_id=table.id, col=partner, num_cols=table.num_cols)
            credit_cols.add(partner)

    return debit_cols | credit_cols


def _build_continuation_children(all_tables: dict[str, Table]) -> dict[str, list[Table]]:
    children: dict[str, list[Table]] = {}
    for table in all_tables.values():
        if table.continuation_of and table.continuation_of in all_tables:
            children.setdefault(table.continuation_of, []).append(table)
    return children


def _detect_root_dc_cols(root: Table, children: dict[str, list[Table]]) -> set[int]:
    try:
        return _detect_dc_cols(root)
    except DcColsNotFoundError:
        logger.warning(f"Колонки Д/К не найдены в корневой таблице {root.id}, пытаемся найти в цепочке продолжений")
        header_table = _find_dc_header_in_chain(root, children)
        if header_table is None:
            raise
        return _detect_dc_cols(header_table)


def _find_dc_header_in_chain(root: Table, children: dict[str, list[Table]]) -> Table | None:
    visited: set[str] = {root.id}
    stack = [root]

    while stack:
        table = stack.pop()
        for child in children.get(table.id, []):
            if child.id in visited:
                continue
            visited.add(child.id)

            try:
                _detect_dc_cols(child)
                return child
            except DcColsNotFoundError:
                stack.append(child)

    return None


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
