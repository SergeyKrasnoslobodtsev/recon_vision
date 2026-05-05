"""Определяет ориентацию страницы и исправляет мелкий наклон."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from paddleocr import DocImgOrientationClassification

from vision_core.config import PageOrientationPreprocessorConfig
from vision_core.entities.bbox import BBox
from vision_core.utils import geometry_utils, image_utils


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

        deskew_angle = self.compute_deskew_angle(aligned_image)
        metadata["deskew_angle_deg"] = deskew_angle

        if deskew_angle == 0.0:
            return aligned_image, metadata

        aligned_image = image_utils.rotate_image(aligned_image, deskew_angle)

        if self._debug:
            self._debug.on_debug_image(
                src_image=aligned_image,
                stage="3_aligned",
                prefix="page",
                page_number=page_number,
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
        binary_image = self._preprocess_image(image)
        raw_tables = self._detected_raw_tables(binary_image)

        angles: list[float] = []
        weights: list[float] = []
        for bbox in raw_tables:
            angle, length = _get_angles(bbox, binary_image)
            angles.append(angle)
            weights.append(length)

        if not angles:
            return 0.0

        a = np.array(angles, dtype=np.float32)
        w = np.array(weights, dtype=np.float32)

        angle = _weighted_median(a, w)
        logger.debug(f"Углы наклона: {angles}, веса: {weights}, итоговый угол: {angle:.4f}°")
        return angle

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Применяет предобработку к изображению перед классификацией ориентации."""
        gamma_corrected = image_utils.gamma_correction(image)
        binary = image_utils.binary_threshold(gamma_corrected, block_size=11, C=5)
        return binary

    def _detected_raw_tables(self, image: np.ndarray) -> list[BBox]:
        """Извлекает bounding boxes таблиц из изображения."""
        h_line_mask = image_utils.compute_horizontal_line_mask(image, scale=40)
        v_line_mask = image_utils.compute_vertical_line_mask(image, median_height=40)
        table_mask = image_utils.get_mask(h_line_mask, v_line_mask)
        return _extract_raw_tables(table_mask)


# ---------------------------------------------------------------------------
# Модульные функции определения угла наклона
# ---------------------------------------------------------------------------


def _rotate_by_orientation(image: np.ndarray, orientation_deg: int) -> np.ndarray:
    if orientation_deg not in (0, 90, 180, 270):
        raise ValueError(f"Недопустимый угол ориентации: {orientation_deg}. Ожидаются 0, 90, 180 или 270.")
    return image_utils.rotate_image(image, orientation_deg)


def _get_angles(bbox: BBox, bin_img: np.ndarray) -> tuple[float, float]:
    box_bin = bbox.roi(bin_img)
    mask = image_utils.compute_horizontal_line_mask(box_bin, scale=10)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 30:
            continue

        comp = (labels == lbl).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue

        cnt = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 80 or bw < 6 * max(1, bh):
            continue

        vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        angle = _normalize_angle_deg(float(np.degrees(np.arctan2(vy, vx))))

        # Убираем явные выбросы: для deskew таблиц угол обычно малый
        if abs(angle) > 3.0:
            continue

        x_pts = cnt[:, 0, 0]
        x1, x2 = int(x_pts.min()), int(x_pts.max())
        if abs(vx) < 1e-6 or x2 <= x1:
            continue
        y1 = int(y0 + (x1 - x0) * (vy / vx))
        y2 = int(y0 + (x2 - x0) * (vy / vx))

        line_img = np.zeros_like(mask)
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 1)
        line_pts = int(np.count_nonzero(line_img))
        if line_pts == 0:
            continue

        support = np.count_nonzero((line_img > 0) & (mask > 0)) / line_pts
        length = float(np.hypot(x2 - x1, y2 - y1))

        if support >= 0.9 and length >= 120:
            return angle, length
    return 0.0, 0.0


def _normalize_angle_deg(angle: float) -> float:
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return angle


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    csum = np.cumsum(weights)
    idx = np.searchsorted(csum, 0.5 * csum[-1])
    return float(values[min(idx, len(values) - 1)])


def _extract_raw_tables(
    table_mask: np.ndarray,
    border_tol: int = 8,
    scale_width: float = 0.35,
    scale_height: float = 0.08,
    min_density: float = 0.01,
    intersection_over_min_thr: float = 0.2,
) -> list[BBox]:
    """Извлекает bounding boxes таблиц из маски таблицы.

    Args:
        table_mask: Бинарная маска, где белые пиксели соответствуют линиям таблицы.
        border_tol: Толерантность к границам изображения в пикселях. Таблицы, касающиеся границ, будут отфильтрованы.
        scale_width: Минимальная ширина таблицы относительно ширины изображения для фильтрации
        scale_height: Минимальная высота таблицы относительно ширины изображения для фильтрации
        min_density: Минимальная плотность белых пикселей внутри bounding box для таблиц, касающихся границ
        intersection_over_min_thr: Порог для объединения перекрывающихся bounding boxes (IoM > threshold)
    Returns:
        list[BBox]: Список bounding boxes таблиц, представленных в виде объектов BBox.
    """
    contours = geometry_utils.find_contours(table_mask)
    candidates: list[tuple[int, int, int, int]] = []
    H, W = table_mask.shape[:2]
    min_width = int(W * scale_width)
    min_height = int(H * scale_height)

    for contour in contours:
        x, y, w, h = geometry_utils.bounding_rect(contour)

        if w < min_width or h < 20:
            continue

        if w > W * 0.95 and h > H * 0.95:
            continue

        touches_border = x <= border_tol or y <= border_tol or x + w >= W - border_tol or y + h >= H - border_tol
        density = geometry_utils.count_non_zero_pixels(geometry_utils.roi_from_xywh(table_mask, (x, y, w, h)))

        if touches_border and density < min_density:
            continue

        candidates.append((x, y, w, h))

    bboxes: list[BBox] = []
    for x, y, w, h in candidates:
        box = BBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h)
        if any(
            kept.contains(box) or box.contains(kept) or kept.intersection_over_min(box) > intersection_over_min_thr
            for kept in bboxes
        ):
            continue
        bboxes.append(box)

    return bboxes
