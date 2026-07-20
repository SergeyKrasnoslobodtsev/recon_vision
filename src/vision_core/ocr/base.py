import enum
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Ocr(enum.Enum):
    PADDLEOCR = 0
    EASYOCR = 1
    TESSERACT = 2


@dataclass
class OcrResult:
    """Результат распознавания текста из одного региона изображения.

    Содержит распознанный текст, уровень уверенности и пространственные координаты
    региона, из которого был извлечен текст.

    Attributes:
        text: Распознанный текст.
        confidence: Уровень уверенности распознавания в диапазоне [0, 1].
        bbox: Ограничивающий прямоугольник в формате (x_min, y_min, x_max, y_max).
    """

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


class OcrEngine(Protocol):
    """Протокол для OCR-движков.

    Определяет интерфейс, который должны реализовать все OCR-движки в системе.
    Позволяет легко подключать различные реализации распознавания текста.
    """

    def predict(self, images: np.ndarray | list[np.ndarray]) -> list[list[OcrResult]]: ...
