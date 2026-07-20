"""Извлекает период акта сверки из документа."""

from __future__ import annotations

from loguru import logger

from app.domain.value_objects.period import Period
from extractor.process import extract
from extractor.tokenize import DateReference
from vision_core.entities.document import Document

_KEYWORDS = {"САЛЬДО", "ПЕРИОД", " НА "}
_KEYWORD_WINDOW = 30


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
                for col, cell in cells_by_col.items():
                    if col < left_bound and cell.value and cell.value.strip():
                        parts.append(cell.value)
    return " ".join(parts)


def _collect_paragraph_text(document: Document) -> str:
    return " ".join(p.text for page in document.pages for p in page.paragraphs if p.text.strip())


def _try_extract(text: str) -> Period | None:
    upper = text.upper()
    dates = [
        t
        for t in extract(text).tokens
        if isinstance(t, DateReference)
        and any(kw in upper[max(0, t.token.start - _KEYWORD_WINDOW) : t.token.start] for kw in _KEYWORDS)
    ]
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

    para_text = _collect_paragraph_text(document)
    period = _try_extract(para_text)
    if period and period.start:
        logger.info(f"период (абзацы): {period.start} - {period.end}")
        return period

    table_text = _collect_table_text(document)
    period = _try_extract(table_text)
    if period and period.start:
        logger.info(f"период (таблица): {period.start} - {period.end}")
        return period

    logger.warning("период не определён")
    return Period()
