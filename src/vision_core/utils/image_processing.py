"""Содержит функции для обработки изображений, используемые в различных модулях."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Преобразует изображение в оттенки серого, если оно цветное.

    Args:
        image: Входное изображение.

    Returns:
        np.ndarray: Изображение в оттенках серого.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Поворачивает изображение на заданный угол.

    Args:
        image: Входное изображение.
        angle_deg: Угол поворота в градусах. Положительный угол задаёт поворот
            против часовой стрелки.

    Returns:
        np.ndarray: Повернутое изображение.
    """
    normalized_angle = angle_deg % 360
    if normalized_angle == 0:
        return image.copy()
    if normalized_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if normalized_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized_angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
