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

    def to_markdown(self) -> str:
        pages_md = [self._page_to_markdown(page) for page in self.pages]
        return "\n\n---\n\n".join(pages_md).strip()

    def _page_to_markdown(self, page: Page) -> str:
        header = f"## Страница {page.page_number + 1}"
        blocks: list[tuple[float, str]] = []
        for paragraph in page.paragraphs:
            md = paragraph.to_markdown()
            if md:
                blocks.append((paragraph.bbox.y_min, md))
        for table in page.tables:
            md = table.to_markdown()
            if md:
                blocks.append((table.bbox.y_min, md))
        blocks.sort(key=lambda b: b[0])
        content = "\n\n".join(text for _, text in blocks)
        return f"{header}\n\n{content}" if content else header

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
