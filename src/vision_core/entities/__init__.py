"""Содержит канонические сущности vision_core."""

from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.entities.paragraph import Paragraph
from vision_core.entities.table import Table

__all__ = [
    "BBox",
    "Cell",
    "Document",
    "Page",
    "Paragraph",
    "Table",
]
