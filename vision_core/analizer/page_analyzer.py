import numpy as np

from vision_core.entities.bbox import BBox
from vision_core.entities.page import Page
from vision_core.entities.table import Table
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.paragraph_preprocessor import ParagraphPreprocessor
from vision_core.detector.table_detector import TableDetector
from vision_core.detector.paragraph_detector import ParagraphDetector


class PageAnalyzer:
    """Анализатор страницы для извлечения таблиц и абзацев"""

    def __init__(self):
        self.image_preprocessor = ImagePreprocessor()
        self.paragraph_preprocessor = ParagraphPreprocessor()
        self.table_detector = TableDetector()
        self.paragraph_detector = ParagraphDetector()

    def analyze_page(self, image: np.ndarray) -> Page:
        """Анализирует страницу и извлекает таблицы и абзацы"""
        # Предобработка изображения
        processed_image = self.image_preprocessor.process(image)

        table_bboxes = self.table_detector.detect_tables(processed_image)

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
