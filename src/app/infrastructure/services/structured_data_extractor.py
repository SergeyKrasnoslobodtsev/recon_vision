"""Предоставляет временную реализацию извлечения данных акта сверки."""

from __future__ import annotations

from typing import Any

from app.domain.entities.reconciliation_data import ReconciliationData
from app.domain.value_objects.period import Period


class StubStructuredDataExtractor:
    """Возвращает временный результат извлечения, пока semantic-слой не реализован."""

    async def extract(self, document_payload: Any) -> ReconciliationData:
        """Извлекает временные данные из построенного документа.

        Args:
            document_payload: Каноническое представление документа.

        Returns:
            ReconciliationData: Временный результат извлечения.
        """
        metadata = (
            document_payload.get("metadata", {})
            if isinstance(document_payload, dict)
            else {}
        )
        return ReconciliationData(
            seller="",
            buyer="",
            period=Period(),
            debit=[],
            credit=[],
            message=metadata.get("message", "done"),
        )