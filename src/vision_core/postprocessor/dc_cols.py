"""Утилита определения дебет/кредит колонок в таблицах акта сверки."""

from __future__ import annotations

from vision_core.entities.page import Page
from vision_core.entities.table import Table

_HEADER_SCAN_ROWS = 4
_DC_KEYWORDS = {"дебет", "кредит"}


def build_dc_cols_map(pages: list[Page]) -> dict[str, set[int]]:
    """Возвращает словарь {table_id -> set of DC col indices} для всех страниц."""
    all_tables: dict[str, Table] = {
        table.id: table
        for page in pages
        for table in page.tables
    }

    dc_cols: dict[str, set[int]] = {}

    for table in all_tables.values():
        if table.continuation_of is None:
            cols = _detect_dc_cols(table)
            if cols:
                dc_cols[table.id] = cols

    for table in all_tables.values():
        if table.continuation_of is not None and table.id not in dc_cols:
            root_id = _find_root(table, all_tables)
            if root_id in dc_cols:
                dc_cols[table.id] = dc_cols[root_id]

    return dc_cols


def _detect_dc_cols(table: Table) -> set[int]:
    dc_cols: set[int] = set()
    for row in table.get_rows()[:_HEADER_SCAN_ROWS]:
        for cell in row:
            if cell.value and _is_dc_header(cell.value):
                dc_cols.add(cell.col)
    return dc_cols


def is_dc_header(value: str) -> bool:
    return any(kw in value.strip().lower() for kw in _DC_KEYWORDS)


def _is_dc_header(value: str) -> bool:
    return is_dc_header(value)


def _find_root(table: Table, all_tables: dict[str, Table]) -> str:
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
    return current.id
