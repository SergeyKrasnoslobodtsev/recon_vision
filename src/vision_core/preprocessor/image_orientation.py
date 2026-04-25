"""Определяет ориентацию страницы и исправляет мелкий наклон."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from paddleocr import DocImgOrientationClassification

from vision_core.config import PageOrientationPreprocessorConfig, TablePreprocessorConfig
from vision_core.utils.image_utils import compute_raw_line_mask, rotate_image


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

        gray = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2GRAY)

        corrected = _correct_perspective_from_table(gray, aligned_image)
        if corrected is not None:
            aligned_image = corrected
            logger.debug("Коррекция перспективы по таблице применена")
        else:
            angle_rad = _detect_rotation(gray, self.cfg)
            angle_deg = math.degrees(angle_rad)
            aligned_image = rotate_image(aligned_image, angle_deg)
            logger.debug(f"Угол наклона (edge fallback): {angle_deg:.3f}°")
            metadata["deskew_angle_deg"] = angle_deg

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


# ---------------------------------------------------------------------------
# Модульные функции определения угла наклона
# ---------------------------------------------------------------------------


def _order_corner_points(points: np.ndarray) -> np.ndarray | None:
    """Возвращает 4 угловые точки в порядке TL, TR, BR, BL.

    Использует суммы и разности координат для определения углов.
    """
    if len(points) < 4:
        return None
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]
    tl = points[np.argmin(sums)]
    br = points[np.argmax(sums)]
    tr = points[np.argmax(diffs)]
    bl = points[np.argmin(diffs)]
    return np.float32([tl, tr, br, bl])


def _correct_perspective_from_table(gray: np.ndarray, image: np.ndarray) -> np.ndarray | None:
    """Корректирует перспективу страницы по 4 угловым точкам таблицы.

    Находит пересечения h/v линий таблицы, выбирает 4 крайние точки (TL, TR, BR, BL)
    и применяет getPerspectiveTransform. Если таблица не найдена — возвращает None.
    """
    table_cfg = TablePreprocessorConfig()
    h_mask, v_mask = compute_raw_line_mask(gray, table_cfg.scale)
    raw_mask = cv2.bitwise_or(h_mask, v_mask)
    _, w = gray.shape

    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt, best_area = None, 0
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < table_cfg.min_table_width_ratio * w:
            continue
        if bh < table_cfg.min_table_height_ratio * bw:
            continue
        area = bw * bh
        if area > best_area:
            best_area, best_cnt = area, cnt

    if best_cnt is None:
        logger.debug("Таблица не найдена — пропуск коррекции перспективы")
        return None

    bx, by, bw, bh = cv2.boundingRect(best_cnt)

    intersections = cv2.bitwise_and(h_mask, v_mask)
    roi = intersections[by : by + bh, bx : bx + bw]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(roi, kernel)

    n_labels, _, _, centroids = cv2.connectedComponentsWithStats(dilated)
    if n_labels < 5:  # фон + минимум 4 угла
        logger.debug("Недостаточно точек пересечения для коррекции перспективы")
        return None

    points = centroids[1:] + np.array([[bx, by]])
    pts1 = _order_corner_points(points)
    if pts1 is None:
        return None

    pts2 = np.float32(
        [
            [bx, by],
            [bx + bw, by],
            [bx + bw, by + bh],
            [bx, by + bh],
        ]
    )

    logger.debug(f"Перспектива: {pts1.tolist()} -> [{bx},{by}] {bw}x{bh}")
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def _detect_rotation(
    gray: np.ndarray,
    cfg: PageOrientationPreprocessorConfig,
    edges: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> float:
    """Определяет угол наклона по четырём краям изображения.

    Args:
        gray: Изображение в оттенках серого.
        cfg: Конфигурация препроцессора.
        edges: Какие края использовать (левый, верхний, правый, нижний).

    Returns:
        Угол наклона в радианах.
    """
    left, top, right, bottom = edges
    rotations = []

    if left:
        rotations.append(_detect_edge_rotation(gray, cfg, (1, 0)))
    if top:
        rotations.append(-_detect_edge_rotation(gray, cfg, (0, 1)))
    if right:
        rotations.append(_detect_edge_rotation(gray, cfg, (-1, 0)))
    if bottom:
        rotations.append(-_detect_edge_rotation(gray, cfg, (0, -1)))

    if not rotations:
        return 0.0

    rotations = np.array(rotations)
    logger.debug(f"edge rotations: {[round(math.degrees(r), 3) for r in rotations]}°")

    # фильтруем выбросы: убираем значения дальше 2*MAD от медианы
    median = np.median(rotations)
    mad = np.median(np.abs(rotations - median))
    scan_step_rad = math.radians(cfg.edge_scan_step_deg)
    mask = np.abs(rotations - median) <= max(2 * mad, scan_step_rad)
    filtered = rotations[mask] if mask.any() else rotations

    logger.debug(f"после фильтрации: {[round(math.degrees(r), 3) for r in filtered]}°")

    # Если есть валидные ненулевые углы, нулевые оценки не участвуют в усреднении.
    # Это уменьшает смещение к 0°, когда один из краёв возвращает 0 из-за слабого сигнала.
    non_zero_mask = np.abs(filtered) > scan_step_rad
    filtered_for_average = filtered[non_zero_mask] if non_zero_mask.any() else filtered
    filtered_not_zero = filtered_for_average[np.abs(filtered_for_average) != 0]
    logger.debug(f"для усреднения используются: {[round(math.degrees(r), 3) for r in filtered_for_average]}°")
    # if filtered_for_average.size != filtered.size:
    #     filtered_for_average = [round(math.degrees(r), 3) for r in filtered if abs(r) <= scan_step_rad]
    #     logger.debug(f"исключены нулевые углы из усреднения: {filtered_for_average}°")

    average = float(np.mean(filtered_not_zero))
    deviation = float(
        np.sqrt(np.sum((filtered - average) ** 2))
    )  # считаем по всем filtered (включая нули) — нули голосуют за "край прямой"
    scan_deviation_rad = math.radians(cfg.edge_scan_deviation_deg)

    logger.debug(f"average: {math.degrees(average):.3f}°  deviation: {math.degrees(deviation):.3f}°")

    if deviation <= scan_deviation_rad:
        return average

    logger.debug("deviation too large -- no rotation")
    return 0.0


def _detect_edge_rotation(
    gray: np.ndarray,
    cfg: PageOrientationPreprocessorConfig,
    shift: tuple[int, int],
) -> float:
    """Определяет угол поворота по одному краю изображения."""
    scan_range_rad = math.radians(cfg.edge_scan_range_deg)
    scan_step_rad = math.radians(cfg.edge_scan_step_deg)

    max_peak = 0
    best_rotation = 0.0
    rotation = 0.0

    while rotation <= scan_range_rad:
        for r in [rotation] if rotation == 0.0 else [rotation, -rotation]:
            m = math.tan(r)
            peak = _detect_edge_rotation_peak(gray, cfg, shift, m)
            if peak > max_peak:
                max_peak = peak
                best_rotation = r
        rotation += scan_step_rad

    return best_rotation


def _detect_edge_rotation_peak(
    gray: np.ndarray,
    cfg: PageOrientationPreprocessorConfig,
    shift: tuple[int, int],
    m: float,
) -> int:
    """Двигает наклонную виртуальную линию от края к центру.

    Возвращает максимальный скачок черноты между шагами.

    Args:
        gray: Изображение в оттенках серого.
        cfg: Конфигурация с параметрами сканирования.
        shift: Направление движения линии (dx, dy).
        m: Наклон линии (тангенс угла).
    """
    h, w = gray.shape
    dx, dy = shift

    if dy == 0:  # горизонтальное сканирование (левый/правый край)
        scan_size = min(cfg.edge_scan_size if cfg.edge_scan_size != -1 else h, 10000, h)
        max_depth = w // 2
        half = scan_size // 2
        outer_offset = int(abs(m) * half)
        mid = h // 2
        side_x = -outer_offset if dx > 0 else w + outer_offset
        X = side_x + half * m
        Y = mid - half
        step_x, step_y = -m, 1.0

    else:  # вертикальное сканирование (верхний/нижний край)
        scan_size = min(cfg.edge_scan_size if cfg.edge_scan_size != -1 else w, 10000, w)
        max_depth = h // 2
        half = scan_size // 2
        outer_offset = int(abs(m) * half)
        mid = w // 2
        side_y = -outer_offset if dy > 0 else h + outer_offset
        X = mid - half
        Y = side_y - half * m
        step_x, step_y = 1.0, -m

    steps = np.arange(scan_size)
    px = np.round(X + steps * step_x).astype(np.int32)
    py = np.round(Y + steps * step_y).astype(np.int32)

    max_blackness_abs = 255 * scan_size * cfg.edge_scan_depth
    last_blackness = 0
    max_diff = 0
    accumulated = 0

    for dep in range(int(max_depth)):
        cur_px = px + dx * dep
        cur_py = py + dy * dep

        valid = (cur_px >= 0) & (cur_px < w) & (cur_py >= 0) & (cur_py < h)
        blackness = int((255 - gray[cur_py[valid], cur_px[valid]]).sum()) if valid.any() else 0

        diff = blackness - last_blackness
        last_blackness = blackness
        if diff > max_diff:
            max_diff = diff

        accumulated += blackness
        if accumulated >= max_blackness_abs:
            break
    else:
        return 0  # дошли до середины — ненадёжно

    return max_diff


def _rotate_by_orientation(image: np.ndarray, orientation_deg: int) -> np.ndarray:
    if orientation_deg not in (0, 90, 180, 270):
        raise ValueError(f"Недопустимый угол ориентации: {orientation_deg}. Ожидаются 0, 90, 180 или 270.")
    return rotate_image(image, orientation_deg)
