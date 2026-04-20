"""Извлекает период акта сверки из документа."""

from __future__ import annotations

from loguru import logger

from app.domain.value_objects.period import Period
from extractor.process import extract
from extractor.tokenize import CurrencyReference, DateReference
from vision_core.entities.document import Document


def _has_dc_value(cells_by_col: dict, dc_cols: set) -> bool:
    for col in dc_cols:
        cell = cells_by_col.get(col)
        if cell and cell.value:
            tokens = extract(cell.value).tokens
            if any(isinstance(t, CurrencyReference) and t.value > 0 for t in tokens):
                return True
    return False


def _collect_table_text(document: Document) -> str:
    parts: list[str] = []
    for page in document.pages:
        for table in page.tables:
            if table.continuation_of is not None or not table.dc_cols:
                continue
            left_bound = min(table.dc_cols)
            header_row = table.get_dc_header_row()
            for row_idx, row in enumerate(table.get_rows()):
                if row_idx <= header_row:
                    continue
                cells_by_col = {cell.col: cell for cell in row}
                if not _has_dc_value(cells_by_col, table.dc_cols):
                    continue
                for col, cell in cells_by_col.items():
                    if col < left_bound and cell.value and cell.value.strip():
                        parts.append(cell.value)
    return " ".join(parts)


def _collect_paragraph_text(document: Document) -> str:
    return " ".join(p.text for page in document.pages for p in page.paragraphs if p.text.strip())


def _try_extract(text: str) -> Period | None:
    dates = [t for t in extract(text).tokens if isinstance(t, DateReference)]
    if not dates:
        return None
    for d in dates:
        if d.date_end:
            return Period(start=d.date, end=d.date_end)
    if len(dates) >= 2:
        sorted_dates = sorted(dates, key=lambda d: d.date.split(".")[::-1])
        return Period(start=sorted_dates[0].date, end=sorted_dates[-1].date)
    return None


def extract_period(document: Document) -> Period:
    """Сначала ищет период в ячейках таблицы, затем в абзацах."""
    table_text = _collect_table_text(document)
    period = _try_extract(table_text)
    if period and period.start:
        logger.debug(f"период (таблица): {period.start} - {period.end}")
        return period

    para_text = _collect_paragraph_text(document)
    period = _try_extract(para_text)
    if period and period.start:
        logger.debug(f"период (абзацы): {period.start} - {period.end}")
        return period

    logger.debug("период не определён")
    return Period()
