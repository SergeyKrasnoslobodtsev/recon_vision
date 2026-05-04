"""Определяет контекст заполнения PDF по структуре документа."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.infrastructure.services.extractor.company_ext import Company, extract_companies
from app.infrastructure.services.extractor.dc_ext import _assign_pair_roles, _find_dc_pairs
from vision_core.entities.document import Document
from vision_core.entities.table import Table


@dataclass(slots=True, frozen=True)
class TableFillContext:
    """Хранит подготовленные данные для заполнения одной таблицы.

    Attributes:
        page_number: Номер страницы таблицы.
        table: Таблица документа.
        font_size: Размер шрифта для рисования значений.
        seller_to_buyer_cols: Соответствие колонок продавца колонкам покупателя.
    """

    page_number: int
    table: Table
    font_size: int
    seller_to_buyer_cols: dict[int, int]


def build_fill_contexts(document: Document) -> dict[str, TableFillContext]:
    """Строит контексты заполнения для таблиц документа.

    Args:
        document: Канонический документ.

    Returns:
        dict[str, TableFillContext]: Контексты по идентификатору таблицы.
    """
    companies = extract_companies(document)
    contexts: dict[str, TableFillContext] = {}

    for page in document.pages:
        for table in page.tables:
            contexts[table.id] = TableFillContext(
                page_number=page.page_number,
                table=table,
                font_size=estimate_table_font_size(table),
                seller_to_buyer_cols=resolve_buyer_columns(table, companies),
            )

    return contexts


def resolve_buyer_columns(table: Table, companies: list[Company]) -> dict[int, int]:
    """Определяет соответствие колонок продавца и покупателя.

    Args:
        table: Таблица документа.
        companies: Организации документа с ролями.

    Returns:
        dict[int, int]: Отображение seller_col -> buyer_col.
    """
    pairs = _find_dc_pairs(table)
    if not pairs:
        return {}

    _assign_pair_roles(table, pairs, companies)
    seller_pair = next((pair for pair in pairs if pair.role == "seller"), None)
    buyer_pair = next((pair for pair in pairs if pair.role == "buyer"), None)
    if seller_pair is None or buyer_pair is None:
        return {}

    return {
        seller_pair.debit_col: buyer_pair.debit_col,
        seller_pair.credit_col: buyer_pair.credit_col,
    }


def estimate_table_font_size(table: Table) -> int:
    """Вычисляет размер шрифта по средней высоте blob'ов таблицы.

    Args:
        table: Таблица документа.

    Returns:
        int: Размер шрифта в пикселях.
    """
    blob_heights = [blob.height for cell in table.cells for blob in cell.blobs if blob.height > 0]

    return int(np.mean(blob_heights) * 0.8)
