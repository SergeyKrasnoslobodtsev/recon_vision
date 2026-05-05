"""Определяет контекст заполнения PDF по структуре документа."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from app.domain.entities.ledger_entry import LedgerEntry
from vision_core.entities.cell import Cell
from vision_core.entities.document import Document
from vision_core.entities.table import Table


@dataclass(slots=True, frozen=True)
class TableFillContext:
    page_number: int
    table: Table
    font_size: int


def build_fill_contexts(document: Document) -> dict[str, TableFillContext]:
    return {
        table.id: TableFillContext(
            page_number=page.page_number,
            table=table,
            font_size=estimate_table_font_size(table),
        )
        for page in document.pages
        for table in page.tables
    }


def resolve_entry_cell(
    contexts: dict[str, TableFillContext],
    entry: LedgerEntry,
) -> tuple[Cell, int, int] | None:
    """Возвращает (ячейка, номер страницы, размер шрифта) для записи или None."""
    ref = entry.row_reference
    if ref is None or ref.buyer_col is None:
        logger.warning(f"запись пропущена, ref={ref}")
        return None

    context = contexts.get(ref.id_table)
    if context is None:
        logger.warning(f"таблица {ref.id_table} не найдена")
        return None

    cell = context.table.get_cell(int(ref.id_row), ref.buyer_col)
    if cell is None:
        logger.warning(f"ячейка не найдена: таблица={ref.id_table} R{ref.id_row}:C{ref.buyer_col}")
        return None

    return cell, context.page_number, context.font_size


def estimate_table_font_size(table: Table) -> int:
    blob_heights = [blob.height for cell in table.cells for blob in cell.blobs if blob.height > 0]
    return int(np.mean(blob_heights) * 0.8)
