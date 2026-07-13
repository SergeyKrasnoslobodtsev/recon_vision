"""Модуль `geometry_utils` содержит функции для работы с геометрическими объектами,
такими как контуры, прямоугольники и другие формы.
Эти функции могут включать в себя операции по нахождению контуров на изображении,
вычислению геометрических характеристик объектов и другие полезные инструменты
для обработки изображений.
"""

from __future__ import annotations

import cv2
import numpy as np


def find_contours(binary_image: np.ndarray) -> list[np.ndarray]:
    """Находит контуры на бинаризованном изображении.

    Args:
        binary_image: Входное бинаризованное изображение, на котором нужно найти контуры.
    Returns:
        list[np.ndarray]: Список найденных контуров, представленных в виде массивов точек.
    """
    return cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def bounding_rect(contour: np.ndarray) -> tuple[int, int, int, int]:
    """Вычисляет ограничивающий прямоугольник для данного контура.

    Args:
        contour: Контур, для которого нужно вычислить ограничивающий прямоугольник.
    Returns:
        tuple[int, int, int, int]: Координаты ограничивающего прямоугольника в формате (x, y, w, h),
        где (x, y) - координаты верхнего левого угла прямоугольника, а (w, h) - его ширина и высота.
    """
    return cv2.boundingRect(contour)


def find_hough_lines(
    binary_image: np.ndarray,
    threshold: int = 100,
    min_line_length: int = 50,
    max_line_gap: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Находит линии на бинаризованном изображении с помощью преобразования Хафа.

    Args:
        binary_image: Входное бинаризованное изображение, на котором нужно найти линии.
        threshold: Порог для функции cv2.HoughLinesP. Чем выше значение, тем меньше линий будет найдено.
        min_line_length: Минимальная длина линии в пикселях. Линии короче этого значения будут отфильтрованы.
        max_line_gap: Максимальный разрыв между сегментами линии, чтобы они считались одной линией.
    Returns:
        list[tuple[int, int, int, int]]: Список отрезков (x1, y1, x2, y2),
        найденных преобразованием Хафа.
    """
    lines = cv2.HoughLinesP(
        binary_image,
        1,
        np.pi / 180,
        threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    return [(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines] if lines is not None else []


def count_non_zero_pixels(image: np.ndarray) -> int:
    """Подсчитывает количество ненулевых пикселей в изображении.

    Args:
        image: Входное изображение, на котором нужно подсчитать ненулевые пиксели.
    Returns:
        int: Количество ненулевых пикселей в изображении.
    """
    return cv2.countNonZero(image)


def iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    """Вычисляет Intersection over Union (IoU) для двух прямоугольников.

    Args:
        boxA: Координаты первого прямоугольника в формате (x_min, y_min, x_max, y_max).
        boxB: Координаты второго прямоугольника в формате (x_min, y_min, x_max, y_max).
    Returns:
        float: Значение IoU между двумя прямоугольниками.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou_value = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

    return iou_value


def roi_from_xyxy(image: np.ndarray, box_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Извлекает регион интереса (ROI) из изображения на основе координат прямоугольника.
    Args:
        image: Входное изображение, из которого нужно извлечь ROI.
        box_xyxy: Координаты прямоугольника в формате (x_min, y_min, x_max, y_max).
    Returns:
        np.ndarray: Извлеченный регион интереса (ROI) из изображения.
    """
    x_min, y_min, x_max, y_max = box_xyxy
    h, w = image.shape[:2]

    x_min = max(0, min(x_min, w))
    x_max = max(0, min(x_max, w))
    y_min = max(0, min(y_min, h))
    y_max = max(0, min(y_max, h))

    if x_max <= x_min or y_max <= y_min:
        return np.empty((0, 0), dtype=image.dtype)

    return image[y_min:y_max, x_min:x_max]


def roi_from_xywh(image: np.ndarray, box_xywh: tuple[int, int, int, int]) -> np.ndarray:
    """Извлекает регион интереса (ROI) из изображения на основе координат прямоугольника в формате (x, y, w, h).
    Args:
        image: Входное изображение, из которого нужно извлечь ROI.
        box_xywh: Координаты прямоугольника в формате (x, y, w, h), где (x, y) - координаты верхнего левого угла,
                   а (w, h) - ширина и высота прямоугольника.
    Returns:
        np.ndarray: Извлеченный регион интереса (ROI) из изображения.
    """
    x, y, w, h = box_xywh
    return roi_from_xyxy(image, (x, y, x + w, y + h))


def median_axis_step(axes: np.ndarray, default: int = 0, min_step: int = 1) -> int:
    """Вычисляет устойчивый медианный шаг между соседними осями.

    Args:
        axes (np.ndarray): Координаты осей линий (например, X или Y).
        default (int): Значение по умолчанию, если шаг вычислить нельзя.
        min_step (int): Минимально допустимый шаг в пикселях.

    Returns:
        median (int): Устойчивый медианный шаг между соседними осями.
    """
    axes = np.asarray(axes, dtype=np.int32)
    if axes.size < 2:
        return default

    axes = np.unique(np.sort(axes))
    diffs = np.diff(axes)
    diffs = diffs[diffs >= min_step]
    if diffs.size == 0:
        return default

    # IQR-фильтрация выбросов
    q1, q3 = np.percentile(diffs, [25, 75])
    iqr = q3 - q1

    if iqr > 0:
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        inliers = diffs[(diffs >= lo) & (diffs <= hi)]
        if inliers.size > 0:
            diffs = inliers

    return int(np.median(diffs))
