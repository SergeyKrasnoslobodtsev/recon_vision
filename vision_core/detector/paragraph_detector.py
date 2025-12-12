import numpy as np
import cv2
from vision_core.entities.bbox import BBox
from vision_core.preprocessor.paragraph_preprocessor import ParagraphPreprocessor


class ParagraphDetector:
    """Детектор абзацев"""

    def __init__(self):
        self.preprocessor = ParagraphPreprocessor()

    def extract_paragraphs(self, image: np.ndarray) -> list[BBox]:
        """Извлекает абзацы из изображения

        Args:
            image (np.ndarray): Входное изображение

        Returns:
            list[BBox]: Список ограничивающих рамок абзацев
        """
        paragraph_mask = self.preprocessor.create_paragraph_mask(image)
        raw_bboxes = self._find_paragraphs(paragraph_mask)
        merged_bboxes = self._merge_paragraphs(raw_bboxes)
        return merged_bboxes

    def _merge_paragraphs(self, bboxes: list[BBox]) -> list[BBox]:
        # объединить близко расположенные абзацы в один
        if not bboxes:
            return []

        bboxes = sorted(bboxes, key=lambda box: (box.y_min, box.x_min))
        merged_bboxes: list[BBox] = []
        current_box = bboxes[0]

        for box in bboxes[1:]:
            if (box.x_min - current_box.x_max < 20) and (
                box.y_min - current_box.y_max < 20
            ):
                current_box = BBox(
                    x_min=min(current_box.x_min, box.x_min),
                    y_min=min(current_box.y_min, box.y_min),
                    x_max=max(current_box.x_max, box.x_max),
                    y_max=max(current_box.y_max, box.y_max),
                )
            else:
                merged_bboxes.append(current_box)
                current_box = box

        merged_bboxes.append(current_box)
        return merged_bboxes

    def _find_paragraphs(self, paragraph_mask: np.ndarray) -> list[BBox]:
        # устранить обрезание абзацев по краям изображения

        contours = cv2.findContours(
            paragraph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]

        bboxes: list[BBox] = []
        for cnt in contours:
            c_poly = cv2.approxPolyDP(cnt, 4, True)
            x, y, w, h = cv2.boundingRect(c_poly)
            if w < 20 or h < 20:
                continue
            bboxes.append(BBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h))
        return bboxes
