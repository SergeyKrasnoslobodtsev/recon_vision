"""Определяет порт построения канонического документа."""

from __future__ import annotations

from typing import Any, Protocol


class DocumentBuilder(Protocol):
    """Описывает контракт построения документа из PDF.

    Note:
        До введения полноценной сущности Document используется Any,
        чтобы не блокировать поэтапную миграцию архитектуры.
    """

    async def build(self, pdf_bytes: bytes) -> Any:
        """Строит каноническое представление документа из PDF."""
