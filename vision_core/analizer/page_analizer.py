import numpy as np
from typing import Optional

from vision_core.entities.bbox import BBox
from vision_core.entities.page import Page
from vision_core.entities.table import Table
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.preprocessor.paragraph_preprocessor import ParagraphPreprocessor
from vision_core.detector.table_detector import TableDetector
from vision_core.detector.table_cell_detector import TableCellDetector
from vision_core.detector.paragraph_detector import ParagraphDetector


class PageAnalizer:
    """Анализатор страницы для извлечения таблиц и абзацев"""

    def __init__(self):
        self.image_preprocessor = ImagePreprocessor()
        self.table_preprocessor = TablePreprocessor()
        self.paragraph_preprocessor = ParagraphPreprocessor()
        self.table_detector = TableDetector()
        self.table_cell_detector = TableCellDetector()
        self.paragraph_detector = ParagraphDetector()

    def analyze_page(self, image: np.ndarray) -> Page:
        """Анализирует страницу и извлекает таблицы и абзацы"""
        # Предобработка изображения
        processed_image = self.image_preprocessor.process(image)

        # Создание маски таблиц
        table_mask = self.table_preprocessor.create_table_mask(processed_image)
        table_bboxes = self.table_detector.extract_raw_tables(table_mask)

        # Создание маски абзацев
        paragraph_mask = self.paragraph_preprocessor.create_paragraph_mask(
            processed_image, table_bboxes
        )

        analysis_result = Page(
            image=image,
            tables=[],
            paragraphs=[],
            metadata={},
        )
        return analysis_result

    def _extract_tables(self, image: np.ndarray, merge_mode: Optional[str] = None):
        """Извлекает таблицы из изображения"""
        table_mask = self.table_preprocessor.create_table_mask(image)
        table_bboxes: list[BBox] = self.table_detector.extract_raw_tables(table_mask)
        tables: list[Table] = []

        for table_bbox in table_bboxes:
            cells = self.table_cell_detector.extract_cells(
                table_mask,
                table_bbox,
                merge_mode=merge_mode,
            )
            table = Table(bbox=table_bbox, cells=cells)
            if table.is_valid() or table.validate_structure():
                continue
            tables.append(table)

        return tables
