"""Определяет порт извлечения бизнес-данных из документа."""

from __future__ import annotations

from typing import Protocol

from app.domain.entities.reconciliation_data import ReconciliationData
from vision_core.entities.document import Document


class StructuredDataExtractor(Protocol):
    """Описывает контракт извлечения данных акта сверки из Document."""

    async def extract(self, document: Document) -> ReconciliationData:
        """Извлекает бизнес-данные из канонического документа."""
