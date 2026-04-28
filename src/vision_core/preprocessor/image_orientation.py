"""Определяет ориентацию страницы и исправляет мелкий наклон."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from paddleocr import DocImgOrientationClassification

from vision_core.config import PageOrientationPreprocessorConfig, TablePreprocessorConfig
from vision_core.utils.image_utils import (
    binary_threshold,
    compute_horizontal_line_mask,
    find_contours,
    fit_line,
    gamma_correction,
    get_median_line_height,
    rotate_image,
)


class PageOrientationPreprocessor:
    """Классифицирует ориентацию страницы в градусах: 0, 90, 180, 270."""

    def __init__(self, config: PageOrientationPreprocessorConfig | None = None, debug_image=None) -> None:
        self.cfg = config or PageOrientationPreprocessorConfig()
        self._debug = debug_image
        if not Path(self.cfg.model_dir).exists():
            raise FileNotFoundError(f"Директория модели ориентации документа не найдена: {self.cfg.model_dir}")
        self.model = DocImgOrientationClassification(
            model_name=self.cfg.model_name,
            model_dir=self.cfg.model_dir,
        )

    def process(self, image: np.ndarray, *, page_number: int = 0) -> tuple[np.ndarray, dict[str, float]]:
        """Выравнивает страницу по ориентации и наклону."""
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
            aligned_image = _rotate_by_orientation(image, orientation_deg)

        # gray = to_grayscale(aligned_image)

        deskew_angle = self.compute_deskew_angle(aligned_image)
        logger.debug(f"Вычисленный угол наклона страницы: {deskew_angle:.4f}°")
        metadata["deskew_angle_deg"] = deskew_angle

        aligned_image = rotate_image(aligned_image, deskew_angle)

        if self._debug:
            self._debug.on_debug_image(
                src_image=aligned_image, stage="3_aligned", prefix="page", page_number=page_number
            )

        return aligned_image, metadata

    def classify(self, image: np.ndarray) -> tuple[int, float]:
        """Возвращает угол ориентации страницы и score модели."""
        results = list(self.model.predict(image))
        if not results:
            return 0, 0.0

        payload = self._extract_payload(results[0])
        labels = payload.get("label_names", [])
        scores = payload.get("scores", [])
        if not labels:
            return 0, 0.0

        return int(labels[0]), float(scores[0]) if scores else 0.0

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

    def compute_deskew_angle(self, image: np.ndarray) -> float:
        """Вычисляет угол наклона страницы в градусах на основе анализа горизонтальных линий."""
        preprocessed = self._preprocess_image_for_orientation(image)
        contours = find_contours(preprocessed)
        median_height = get_median_line_height(contours)

        if median_height == 0:
            return 0.0

        filtered_contours = self._filtered_contours(contours, median_height)

        if not filtered_contours:
            return 0.0

        angles = [self._angle_from_contour(cnt) for cnt in filtered_contours]

        return np.median(angles)

    def _angle_from_contour(self, contour: np.ndarray) -> float:
        """Вычисляет угол наклона в градусах для данного контура."""
        vx, vy, _, _ = fit_line(contour)
        angle = np.degrees(np.arctan2(vy, vx))
        return angle

    def _filtered_contours(self, contours: list[np.ndarray], height: float) -> list[np.ndarray]:
        """Фильтрует контуры, оставляя только те, которые имеют высоту, близкую к медианной высоте линий."""
        filtered_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(h - height) < height * 0.5:
                filtered_contours.append(cnt)
        return filtered_contours

    def _preprocess_image_for_orientation(self, image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """Применяет предобработку к изображению перед классификацией ориентации."""
        # Применяем гамма-коррекцию для улучшения контраста
        gamma_corrected = gamma_correction(image, gamma=gamma)
        binary = binary_threshold(gamma_corrected)
        horizontal_lines = compute_horizontal_line_mask(binary, scale=50)

        return horizontal_lines


# ---------------------------------------------------------------------------
# Модульные функции определения угла наклона
# ---------------------------------------------------------------------------


def _rotate_by_orientation(image: np.ndarray, orientation_deg: int) -> np.ndarray:
    if orientation_deg not in (0, 90, 180, 270):
        raise ValueError(f"Недопустимый угол ориентации: {orientation_deg}. Ожидаются 0, 90, 180 или 270.")
    return rotate_image(image, orientation_deg)
