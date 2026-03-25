"""Определяет доменную модель извлечённых данных акта сверки."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.entities.ledger_entry import LedgerEntry
from app.domain.value_objects.period import Period


@dataclass(slots=True)
class ReconciliationData:
    """Хранит извлечённые бизнес-данные акта сверки.

    Attributes:
        seller: Наименование продавца.
        buyer: Наименование покупателя.
        period: Период сверки.
        debit: Записи дебета.
        credit: Записи кредита.
        message: Служебное сообщение о результате извлечения.
    """

    seller: str = ""
    buyer: str = ""
    period: Period = field(default_factory=Period)
    debit: list[LedgerEntry] = field(default_factory=list)
    credit: list[LedgerEntry] = field(default_factory=list)
    message: str = ""
