from pydantic import BaseModel
from typing import Any

from vision_core.entities.table import Table
from vision_core.entities.paragraph import Paragraph


class Page(BaseModel):
    tables: list[Table]
    paragraphs: list[Paragraph]
    metadata: dict[str, Any] = {}
