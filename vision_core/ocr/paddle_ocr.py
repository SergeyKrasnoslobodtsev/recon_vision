from vision_core.ocr.base import OcrEngine, OcrResult
from vision_core.config import VisionCoreConfig
from paddleocr import PaddleOCR
from pathlib import Path
import numpy as np
from typing import Union, Optional


class PaddleOcrEngine(OcrEngine):
    def __init__(self, config: Optional[VisionCoreConfig]):
        if config is None:
            config = VisionCoreConfig()

        self.cfg = config.paddleocr

        if not Path(self.cfg.text_detection_model_dir).exists():
            raise FileNotFoundError(
                f"Text detection model directory not found: {self.cfg.text_detection_model_dir}"
            )

        if not Path(self.cfg.text_recognition_model_dir).exists():
            raise FileNotFoundError(
                f"Text recognition model directory not found: {self.cfg.text_recognition_model_dir}"
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
        """Распознает текст на изображениях и возвращает результаты по одному изображению за раз."""
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
        """Распознает текст на изображениях и возвращает результаты для всех изображений."""
        return list(self.predict_iter(images))
