"""Определяет доменные сущности записей акта сверки."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RowReference:
    """Описывает ссылку на строку табличной структуры документа.

    Attributes:
        id_table: Идентификатор таблицы в документе.
        id_row: Идентификатор строки в таблице.
        id_col: Индекс целевой колонки в строке, если он определён.
    """

    id_table: str
    id_row: str
    id_col: int | None = None


@dataclass(slots=True, frozen=True)
class LedgerEntry:
    """Описывает одну бухгалтерскую запись акта сверки.

    Attributes:
        record: Текст записи.
        value: Денежное значение записи.
        date: Дата записи в строковом формате.
        row_reference: Ссылка на строку документа, если она определена.
    """

    record: str
    value: float
    date: str | None = None
    row_reference: RowReference | None = None
