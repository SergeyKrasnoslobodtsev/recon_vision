"""Определяет DTO сценария заполнения акта сверки."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.entities.ledger_entry import LedgerEntry


@dataclass(slots=True, frozen=True)
class FillReconciliationActCommand:
    """Хранит команду заполнения акта сверки."""

    process_id: str
    comments: str | None = None
    debit: list[LedgerEntry] = field(default_factory=list)
    credit: list[LedgerEntry] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class FillReconciliationActResult:
    """Хранит результат заполнения PDF документа."""

    document_base64: str
