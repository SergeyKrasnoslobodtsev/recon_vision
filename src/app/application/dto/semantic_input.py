"""Определяет входной контракт для semantic analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SemanticParagraph:
    """Описывает текстовый блок для semantic analysis.

    Attributes:
        id: Идентификатор блока.
        page_number: Номер страницы.
        text: Текст блока.
        block_type: Тип блока, например header/footer/text.
        bbox: Геометрия блока в координатах страницы.
    """

    id: str
    page_number: int
    text: str
    block_type: str
    bbox: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class SemanticCell:
    """Описывает содержимое одной ячейки таблицы.

    Attributes:
        row: Номер строки.
        col: Номер столбца.
        text: Текст ячейки.
        rowspan: Объединение по строкам.
        colspan: Объединение по столбцам.
    """

    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1


@dataclass(slots=True)
class SemanticTable:
    """Описывает таблицу для semantic analysis.

    Attributes:
        id: Идентификатор таблицы.
        page_number: Номер страницы, на которой находится таблица.
        cells: Структурированное представление ячеек таблицы.
    """

    id: str
    page_number: int
    cells: list[SemanticCell] = field(default_factory=list)


@dataclass(slots=True)
class SemanticInput:
    """Хранит нормализованный вход для semantic analysis.

    Attributes:
        paragraphs: Связный текст параграфов с минимальной метаинформацией.
        tables: Структурированное представление таблиц и текста ячеек.
        linearized_text: Производный линейный текст документа для NLP/LLM сценариев.
        document_metadata: Служебные метаданные документа.
    """

    paragraphs: list[SemanticParagraph] = field(default_factory=list)
    tables: list[SemanticTable] = field(default_factory=list)
    linearized_text: str = ""
    document_metadata: dict[str, Any] = field(default_factory=dict)