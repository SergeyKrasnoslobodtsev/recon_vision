"""Определяет ориентацию страницы и исправляет мелкий наклон."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from paddleocr import DocImgOrientationClassification

from vision_core.config import PageOrientationPreprocessorConfig
from vision_core.utils.image_processing import rotate_image, to_grayscale


class PageOrientationPreprocessor:
    """Классифицирует ориентацию страницы в градусах: 0, 90, 180, 270."""

    def __init__(self, config: PageOrientationPreprocessorConfig | None = None) -> None:
        """Инициализирует классификатор ориентации страницы.

        Args:
            config: Конфигурация модели ориентации.

        Raises:
            FileNotFoundError: Если директория модели не найдена.
        """
        self.cfg = config or PageOrientationPreprocessorConfig()
        if not Path(self.cfg.model_dir).exists():
            raise FileNotFoundError(f"Директория модели ориентации документа не найдена: {self.cfg.model_dir}")
        self.model = DocImgOrientationClassification(
            model_name=self.cfg.model_name,
            model_dir=self.cfg.model_dir,
        )

    def process(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Выравнивает страницу по ориентации и наклону.

        Args:
            image: Исходное изображение страницы.

        Returns:
            tuple[np.ndarray, dict[str, float]]: Выравненное изображение и metadata шага.
        """
        metadata: dict[str, float] = {
            "orientation_deg": 0.0,
            "orientation_score": 0.0,
            "deskew_angle_deg": 0.0,
        }

        aligned_image = image.copy()

        orientation_deg, orientation_score = self.classify(image)
        logger.debug(f"Ориентация страницы: {orientation_deg}° с точностью {orientation_score:.4f}")
        metadata["orientation_deg"] = orientation_deg
        metadata["orientation_score"] = orientation_score

        if orientation_score >= self.cfg.min_orientation_score:
            aligned_image = self._rotate_by_orientation(image, orientation_deg)

        deskew_angle_deg = self.deskew(aligned_image)
        aligned_image = rotate_image(aligned_image, deskew_angle_deg)
        logger.debug(f"Угол наклона страницы: {deskew_angle_deg}°")
        metadata["deskew_angle_deg"] = deskew_angle_deg

        return aligned_image, metadata

    def classify(self, image: np.ndarray) -> tuple[int, float]:
        """Возвращает угол ориентации страницы и score модели.

        Args:
            image: Изображение страницы.

        Returns:
            tuple[int, float]: Угол ориентации в градусах и score уверенности.
        """
        results = list(self.model.predict(image))
        if not results:
            return 0, 0.0

        payload = self._extract_payload(results[0])
        labels = payload.get("label_names", [])
        scores = payload.get("scores", [])
        if not labels:
            return 0, 0.0

        angle_deg = int(labels[0])
        score = float(scores[0]) if scores else 0.0
        return angle_deg, score

    def _extract_payload(self, result) -> dict:
        if isinstance(result, dict):
            return result.get("res", result)

        json_payload = getattr(result, "json", None)
        if isinstance(json_payload, dict):
            return json_payload.get("res", json_payload)
        if callable(json_payload):
            payload = json_payload()
            if isinstance(payload, dict):
                return payload.get("res", payload)

        raise TypeError(f"Неподдерживаемый формат результата ориентации: {type(result)!r}")

    def deskew(self, image: np.ndarray) -> float:
        """Исправляет мелкий наклон изображения.

        Args:
            image: Исходное изображение страницы.

        Returns:
            float: Угол наклона изображения страницы.
        """
        gray = to_grayscale(image)
        height, width = gray.shape
        im_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        coarse_best_angle = 0.0
        coarse_best_score = -1.0
        coarse_step_10x = max(1, int(round(self.cfg.coarse_step_deg * 10)))
        max_skew_10x = int(round(self.cfg.max_skew_deg * 10))

        for angle_10x in range(-max_skew_10x, max_skew_10x + 1, coarse_step_10x):
            angle = angle_10x / 10.0
            score = self._projection_score(im_bw, angle, width, height)
            if score > coarse_best_score:
                coarse_best_score = score
                coarse_best_angle = angle

        fine_best_angle = coarse_best_angle
        fine_best_score = coarse_best_score
        fine_window_10x = int(round(self.cfg.fine_window_deg * 10))
        fine_step_10x = max(1, int(round(self.cfg.fine_step_deg * 10)))
        fine_start = int(round((coarse_best_angle * 10) - fine_window_10x))
        fine_end = int(round((coarse_best_angle * 10) + fine_window_10x))

        for angle_10x in range(fine_start, fine_end + 1, fine_step_10x):
            angle = angle_10x / 10.0
            score = self._projection_score(im_bw, angle, width, height)
            if score > fine_best_score:
                fine_best_score = score
                fine_best_angle = angle

        if abs(fine_best_angle) < self.cfg.min_abs_deskew_angle_deg:
            return 0.0

        return fine_best_angle

    def _projection_score(
        self,
        im_bw: np.ndarray,
        angle: float,
        width: int,
        height: int,
    ) -> float:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated = cv2.warpAffine(im_bw, matrix, (width, height), flags=cv2.INTER_NEAREST)
        projection = np.sum(rotated, axis=1, dtype=np.float64)
        return float(np.var(projection))

    def _rotate_by_orientation(self, image: np.ndarray, orientation_deg: int) -> np.ndarray:
        if orientation_deg not in (0, 90, 180, 270):
            raise ValueError(f"Недопустимый угол ориентации: {orientation_deg}. Ожидаются 0, 90, 180 или 270.")
        return rotate_image(image, orientation_deg)
