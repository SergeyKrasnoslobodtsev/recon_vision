"""Определяет порт извлечения бизнес-данных из документа."""

from __future__ import annotations

from typing import Protocol

from app.application.dto.semantic_input import SemanticInput
from app.domain.entities.reconciliation_data import ReconciliationData


class StructuredDataExtractor(Protocol):
    """Описывает контракт извлечения данных акта сверки из semantic input."""

    async def extract(self, semantic_input: SemanticInput) -> ReconciliationData:
        """Извлекает бизнес-данные из semantic input."""
