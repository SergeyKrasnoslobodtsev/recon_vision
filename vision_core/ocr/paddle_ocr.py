from vision_core.ocr.base import OcrEngine, OcrResult
from vision_core.config import VisionCoreConfig
from paddleocr import PaddleOCR
from pathlib import Path
import numpy as np
from typing import Union, Optional


class PaddleOcrEngine(OcrEngine):
    """Реализует OCR-движок на базе PaddleOCR.

    Использует модели PaddleOCR для распознавания текста на изображениях.
    Поддерживает пакетную обработку и итеративный режим для больших объёмов данных.

    Attributes:
        cfg: Конфигурация PaddleOCR из VisionCoreConfig.
        ocr: Экземпляр PaddleOCR для выполнения распознавания.
    """

    def __init__(self, config: Optional[VisionCoreConfig]):
        """Инициализирует движок PaddleOCR.

        Args:
            config: Конфигурация VisionCore. Если None, используется конфигурация по умолчанию.

        Raises:
            FileNotFoundError: Если директории с моделями распознавания или детекции не найдены.
        """
        if config is None:
            config = VisionCoreConfig()

        self.cfg = config.paddleocr

        if not Path(self.cfg.text_detection_model_dir).exists():
            raise FileNotFoundError(
                f"Директория модели детекции текста не найдена: {self.cfg.text_detection_model_dir}"
            )

        if not Path(self.cfg.text_recognition_model_dir).exists():
            raise FileNotFoundError(
                f"Директория модели распознавания текста не найдена: {self.cfg.text_recognition_model_dir}"
            )

        self.ocr = PaddleOCR(
            text_recognition_model_name=self.cfg.text_recognition_model_name,
            text_recognition_model_dir=self.cfg.text_recognition_model_dir,
            text_detection_model_name=self.cfg.text_detection_model_name,
            text_detection_model_dir=self.cfg.text_detection_model_dir,
            use_doc_orientation_classify=self.cfg.use_doc_orientation_classify,
            use_doc_unwarping=self.cfg.use_doc_unwarping,
            use_textline_orientation=self.cfg.use_textline_orientation,
            device=self.cfg.device,
        )

    def predict_iter(self, images: Union[np.ndarray, list[np.ndarray]]):
        """Распознаёт текст на изображениях в итеративном режиме.

        Обрабатывает изображения по одному и возвращает результаты через генератор.
        Полезно для больших наборов данных, когда нежелательно загружать
        все результаты в память одновременно.

        Args:
            images: Одно изображение (numpy array) или список изображений для распознавания.
                Если передан одиночный массив, автоматически оборачивается в список.

        Yields:
            list[OcrResult]: Список результатов распознавания для каждого изображения.
                Каждый элемент содержит текст, уверенность (0.0–1.0) и координаты
                ограничивающего прямоугольника (x1, y1, x2, y2).

        Examples:
            >>> engine = PaddleOcrEngine(config)
            >>> images = [image1, image2]
            >>> for results in engine.predict_iter(images):
            ...     for result in results:
            ...         print(f"Текст: {result.text}, Уверенность: {result.confidence}")
        """
        if isinstance(images, np.ndarray):
            images = [images]

        for res in self.ocr.predict_iter(images):
            boxes = res.get("rec_boxes", []) or []
            texts = res.get("rec_texts", []) or []
            scores = res.get("rec_scores", []) or []

            out: list[OcrResult] = []
            for box, text, score in zip(boxes, texts, scores):
                out.append(
                    OcrResult(
                        text=str(text),
                        confidence=float(score),
                        bboxes=(box[0], box[1], box[2], box[3]),
                    )
                )

            yield out

    def predict(self, images: Union[np.ndarray, list[np.ndarray]]):
        """Распознаёт текст на изображениях и возвращает все результаты сразу.

        Args:
            images: Одно изображение (numpy array) или список изображений для распознавания.

        Returns:
            list[list[OcrResult]]: Список списков результатов распознавания для каждого изображения.
        """
        return list(self.predict_iter(images))
