"""Определяет порт извлечения бизнес-данных из документа."""

from __future__ import annotations

from typing import Any, Protocol

from app.domain.entities.reconciliation_data import ReconciliationData


class StructuredDataExtractor(Protocol):
    """Описывает контракт извлечения данных акта сверки из документа."""

    async def extract(self, document_payload: Any) -> ReconciliationData:
        """Извлекает бизнес-данные из канонического документа."""
