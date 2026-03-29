"""Определяет порт построения semantic input из канонического документа."""

from __future__ import annotations

from typing import Any, Protocol

from app.application.dto.semantic_input import SemanticInput


class SemanticInputProjector(Protocol):
    """Описывает контракт проекции канонического документа в semantic input."""

    async def build(self, document_payload: Any) -> SemanticInput:
        """Строит SemanticInput из канонического документа."""
