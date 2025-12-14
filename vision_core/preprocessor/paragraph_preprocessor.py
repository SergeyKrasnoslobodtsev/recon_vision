import numpy as np
import cv2

from vision_core.entities.bbox import BBox


class ParagraphPreprocessor:
    """Предобработчик для абзацев"""

    def create_paragraph_mask(self, image: np.ndarray, bboxes: list[BBox]):
        processed = self._processing(image)

        paragraph_mask = self._detect_paragraphs(processed)

        for bbox in bboxes:
            bbox = bbox.padding(5)
            pt1 = (int(bbox.x_min), int(bbox.y_min))
            pt2 = (int(bbox.x_max), int(bbox.y_max))
            cv2.rectangle(
                paragraph_mask,
                pt1,
                pt2,
                (0, 0, 0),
                thickness=-1,
            )

        return paragraph_mask

    def _processing(self, image: np.ndarray):
        blur = cv2.GaussianBlur(image, (7, 7), 0)
        binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return binary

    def _detect_paragraphs(self, processed: np.ndarray):
        # Используем морфологические операции для выделения абзацев
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
        paragraph_mask = cv2.morphologyEx(
            processed,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=4,
        )

        return paragraph_mask
