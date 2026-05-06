"""Извлекает бухгалтерские записи дебет/кредит из таблиц документа."""

from __future__ import annotations

from dataclasses import dataclass, replace

from loguru import logger

from app.domain.entities.ledger_entry import LedgerEntry, RowReference
from app.infrastructure.services.extractor.company_ext import Company
from extractor.process import extract
from extractor.tokenize import CurrencyReference, DateReference
from vision_core.entities.document import Document
from vision_core.entities.table import Table

from .exceptions import DcExtractionError

_SELLER_KEYWORDS = {"ОТ ПРОДАВЦА", "ПРОДАВЕЦ", "ПО ДАННЫМ ПРОДАВЦА"}
_BUYER_KEYWORDS = {"ОТ ПОКУПАТЕЛЯ", "ПОКУПАТЕЛЬ", "ПО ДАННЫМ ПОКУПАТЕЛЯ"}


@dataclass
class _DcPair:
    debit_col: int
    credit_col: int
    role: str = "unknown"  # "seller" | "buyer" | "unknown"


def _find_dc_pairs(table: Table) -> list[_DcPair]:
    sorted_cols = sorted(table.dc_cols)
    return [_DcPair(debit_col=sorted_cols[i], credit_col=sorted_cols[i + 1]) for i in range(0, len(sorted_cols) - 1, 2)]


def _detect_role(text: str, companies: list[Company]) -> str | None:
    normalized = extract(text).text
    if any(kw in normalized for kw in _SELLER_KEYWORDS):
        return "seller"
    if any(kw in normalized for kw in _BUYER_KEYWORDS):
        return "buyer"
    for c in companies:
        if c.name in normalized:
            return c.role
    return None


def _apply_symmetry(pairs: list[_DcPair]) -> None:
    sellers = [p for p in pairs if p.role == "seller"]
    buyers = [p for p in pairs if p.role == "buyer"]
    unknowns = [p for p in pairs if p.role == "unknown"]
    if sellers and unknowns and not buyers:
        for p in unknowns:
            p.role = "buyer"
    elif buyers and unknowns and not sellers:
        for p in unknowns:
            p.role = "seller"


def _assign_pair_roles(table: Table, pairs: list[_DcPair], companies: list[Company]) -> None:
    header_row = table.get_dc_header_row()
    if header_row <= 0:
        _apply_symmetry(pairs)
        return
    rows = table.get_rows()
    for row_idx in range(header_row):
        for cell in rows[row_idx]:
            if not cell.value:
                continue
            role = _detect_role(cell.value, companies)
            if role is None:
                continue
            covered = set(range(cell.col, cell.col + cell.colspan))
            for pair in pairs:
                if pair.role == "unknown" and (pair.debit_col in covered or pair.credit_col in covered):
                    pair.role = role
    _apply_symmetry(pairs)


def _parse_value(cell) -> float:
    if cell is None or not cell.value or not cell.value.strip():
        return 0.0
    currencies = [t for t in extract(cell.value).tokens if isinstance(t, CurrencyReference)]
    return currencies[0].value if currencies else 0.0


def _only_tokens(text: str, tokens: list) -> bool:
    """True если в тексте нет ничего кроме указанных токенов и пробелов."""
    leftover = text
    for tok in sorted(tokens, key=lambda t: t.token.start, reverse=True):
        leftover = leftover[: tok.token.start] + leftover[tok.token.end :]
    return not leftover.strip()


def _build_row_data(cells: dict, left_bound: int) -> tuple[str, str | None]:
    """Собирает текст записи и дату из ячеек левее dc_cols."""
    record_parts: list[str] = []
    date: str | None = None

    for col in sorted(k for k in cells if k < left_bound):
        cell = cells[col]
        if not cell.value or not cell.value.strip():
            continue
        result = extract(cell.value)
        dates = [t for t in result.tokens if isinstance(t, DateReference)]
        currencies = [t for t in result.tokens if isinstance(t, CurrencyReference)]

        if dates:
            date = date or dates[0].date
        if _only_tokens(result.text, dates + currencies):
            continue  # ячейка целиком из дат/сумм — в запись не включаем
        record_parts.append(result.text.strip())

    return " ".join(record_parts), date


def _extract_entries(table: Table, pairs: list[_DcPair], role: str) -> tuple[list[LedgerEntry], list[LedgerEntry]]:
    role_pairs = [p for p in pairs if p.role == role]
    if not role_pairs:
        raise DcExtractionError(table_id=table.id, details=f"колонки '{role}' не найдены")
    header_row = table.get_dc_header_row()
    left_bound = min(table.dc_cols)
    rows = table.get_rows()

    debit_entries: list[LedgerEntry] = []
    credit_entries: list[LedgerEntry] = []

    for row_idx, row in enumerate(rows):
        if row_idx <= header_row:
            continue
        cells = {cell.col: cell for cell in row}
        record, date = _build_row_data(cells, left_bound)

        for pair in role_pairs:
            d_cell = cells.get(pair.debit_col)
            c_cell = cells.get(pair.credit_col)
            debit_entries.append(
                LedgerEntry(
                    record=record,
                    value=_parse_value(d_cell),
                    date=date,
                    row_reference=RowReference(
                        id_table=table.id,
                        id_row=str(row_idx),
                        id_col=pair.debit_col,
                    ),
                )
            )
            credit_entries.append(
                LedgerEntry(
                    record=record,
                    value=_parse_value(c_cell),
                    date=date,
                    row_reference=RowReference(
                        id_table=table.id,
                        id_row=str(row_idx),
                        id_col=pair.credit_col,
                    ),
                )
            )

    return debit_entries, credit_entries


def extract_dc(
    document: Document,
    companies: list[Company],
) -> tuple[list[LedgerEntry], list[LedgerEntry]]:
    """Извлекает дебет/кредит из всех таблиц документа.

    Каждая запись содержит значение продавца и buyer_col — колонку покупателя для заполнения PDF.
    Поднимает ValueError если в таблице не найдены колонки продавца или покупателя.
    """
    all_tables: dict[str, Table] = {t.id: t for p in document.pages for t in p.tables}

    root_pairs: dict[str, list[_DcPair]] = {}
    for page in document.pages:
        for table in page.tables:
            if not table.dc_cols or table.continuation_of is not None:
                continue
            pairs = _find_dc_pairs(table)
            _assign_pair_roles(table, pairs, companies)
            if not pairs:
                logger.error(f"таблица {table.id}: не удалось определить пары дебет/кредит")
                raise DcExtractionError(table_id=table.id, details="не удалось определить пары дебет/кредит")
            root_pairs[table.id] = pairs
            logger.debug(f"таблица {table.id}: пары {[(p.debit_col, p.credit_col, p.role) for p in pairs]}")

    all_debit: list[LedgerEntry] = []
    all_credit: list[LedgerEntry] = []

    for page in document.pages:
        for table in page.tables:
            if not table.dc_cols:
                continue
            if table.continuation_of is not None:
                root_id = table.continuation_of
                while root_id and all_tables.get(root_id, table).continuation_of:
                    root_id = all_tables[root_id].continuation_of
                source = root_pairs.get(root_id)
                pairs = (
                    [_DcPair(p.debit_col, p.credit_col, p.role) for p in source] if source else _find_dc_pairs(table)
                )
                if not source:
                    _assign_pair_roles(table, pairs, companies)
                if not pairs:
                    logger.error(f"таблица {table.id}: не удалось определить пары дебет/кредит")
                    raise DcExtractionError(table_id=table.id, details="не удалось определить пары дебет/кредит")
                logger.debug(
                    f"таблица {table.id}: наследует пары {[(p.debit_col, p.credit_col, p.role) for p in pairs]} "
                    f"из таблицы {root_id}"
                )
            else:
                pairs = root_pairs.get(table.id, [])

            if not pairs:
                logger.error(f"таблица {table.id}: не удалось определить пары дебет/кредит")
                raise DcExtractionError(table_id=table.id, details="не удалось определить пары дебет/кредит")

            debit, credit = _extract_entries(table, pairs, "seller")
            buyer_debit, buyer_credit = _extract_entries(table, pairs, "buyer")
            enriched_debit = [
                replace(s, row_reference=replace(s.row_reference, buyer_col=b.row_reference.id_col))
                for s, b in zip(debit, buyer_debit, strict=False)
            ]
            enriched_credit = [
                replace(s, row_reference=replace(s.row_reference, buyer_col=b.row_reference.id_col))
                for s, b in zip(credit, buyer_credit, strict=False)
            ]
            all_debit.extend(enriched_debit)
            all_credit.extend(enriched_credit)

    logger.info(f"dc: {len(all_debit)} дебет, {len(all_credit)} кредит")
    return all_debit, all_credit
