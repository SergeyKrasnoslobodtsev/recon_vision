"""Предоставляет временную реализацию извлечения данных акта сверки."""

from __future__ import annotations

from app.application.dto.semantic_input import SemanticInput
from app.domain.entities.reconciliation_data import ReconciliationData
from app.domain.value_objects.period import Period


class StubStructuredDataExtractor:
    """Возвращает временный результат извлечения, пока semantic-слой не реализован."""

    async def extract(self, semantic_input: SemanticInput) -> ReconciliationData:
        """Извлекает временные данные из semantic input.

        Args:
            semantic_input: Нормализованный вход semantic analysis.

        Returns:
            ReconciliationData: Временный результат извлечения.
        """
        return ReconciliationData(
            seller="",
            buyer="",
            period=Period(),
            debit=[],
            credit=[],
            message=semantic_input.document_metadata.get("message", "done"),
        )
