from pydantic import BaseModel, Field
from typing import Any

from vision_core.entities.table import Table
from vision_core.entities.paragraph import Paragraph


class Page(BaseModel):
    """Хранит структурные данные одной страницы документа.

    Attributes:
        page_number: Номер страницы в документе, начиная с 0.
        tables: Таблицы, найденные на странице.
        paragraphs: Параграфы, найденные на странице.
        metadata: Дополнительные служебные данные страницы.
    """

    page_number: int = 0
    tables: list[Table] = Field(default_factory=list)
    paragraphs: list[Paragraph] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
