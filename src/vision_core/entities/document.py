"""Определяет каноническую сущность документа."""

from __future__ import annotations

from hashlib import sha1
from typing import Any

from pydantic import BaseModel, Field

from vision_core.entities.page import Page


class Document(BaseModel):
    """Хранит каноническое представление документа.

    Attributes:
        pages: Страницы документа после структурного анализа.
        source_hash: Хеш исходного PDF для идентификации документа.
        metadata: Дополнительные служебные данные документа.
    """

    pages: list[Page] = Field(default_factory=list)
    source_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_pages(self) -> int:
        """Возвращает количество страниц документа."""
        return len(self.pages)

    @classmethod
    def from_pdf_bytes(
        cls,
        pdf_bytes: bytes,
        pages: list[Page],
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Создаёт документ из PDF-байтов и набора страниц.

        Args:
            pdf_bytes: Исходный PDF-файл.
            pages: Список проанализированных страниц.
            metadata: Дополнительные служебные метаданные.

        Returns:
            Document: Каноническое представление документа.
        """
        return cls(
            pages=pages,
            source_hash=sha1(pdf_bytes).hexdigest(),
            metadata=metadata or {},
        )
