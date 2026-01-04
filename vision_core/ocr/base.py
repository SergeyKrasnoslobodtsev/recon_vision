from dataclasses import dataclass
import enum
from abc import ABC, abstractmethod


class Ocr(enum.Enum):
    PADDLEOCR = 0
    EASYOCR = 1
    TESSERACT = 2


@dataclass
class OcrResult:
    text: str
    confidence: float
    bboxes: tuple[int, int, int, int]


class OcrEngine(ABC):
    @abstractmethod
    def predict(self, images: list) -> list[list[OcrResult]]: ...
