import numpy as np
import cv2
from vision_core.entities.bbox import BBox
from loguru import logger


class ParagraphDetector:
    """Детектор абзацев"""

    def extract_paragraphs(self, paragraph_mask: np.ndarray) -> list[BBox]:
        raw_bboxes = self._find_paragraphs(paragraph_mask)

        return raw_bboxes

    def _find_paragraphs(self, paragraph_mask: np.ndarray) -> list[BBox]:
        contours = cv2.findContours(
            paragraph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]

        bboxes: list[BBox] = []
        for cnt in contours:
            c_poly = cv2.approxPolyDP(cnt, 4, True)
            x, y, w, h = cv2.boundingRect(c_poly)
            if w < 10 or h < 10:
                continue
            bboxes.append(BBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h))
        return bboxes
