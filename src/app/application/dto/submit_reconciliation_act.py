"""Определяет DTO сценария отправки акта сверки."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class SubmitReconciliationActCommand:
    """Хранит команду отправки PDF на обработку."""

    document_base64: str


@dataclass(slots=True, frozen=True)
class SubmitReconciliationActResult:
    """Хранит результат создания процесса обработки."""

    process_id: str
