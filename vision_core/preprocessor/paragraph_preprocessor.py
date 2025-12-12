import numpy as np
import cv2


class ParagraphPreprocessor:
    """Предобработчик для абзацев"""

    def __init__(self):
        pass

    def create_paragraph_mask(self, image: np.ndarray):
        processed = self._processing(image)

        paragraph_mask = self._detect_paragraphs(processed)
        return paragraph_mask

    def _processing(self, image: np.ndarray):
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return binary

    def _detect_paragraphs(self, processed: np.ndarray):
        # Используем морфологические операции для выделения абзацев
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        paragraph_mask = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        return paragraph_mask
