"""Содержит функции для обработки изображений, используемые в различных модулях."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Преобразует изображение в оттенки серого, если оно цветное.

    Args:
        image: Входное изображение RGB.
    Returns:
        np.ndarray: Изображение в оттенках серого.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()


def unsharp_mask(image: np.ndarray, kernel_size=(5, 5), sigma=1.5, amount=1.0) -> np.ndarray:
    """Применяет unsharp masking для повышения резкости изображения.
    Args:
        image: Входное изображение в формате BGR или RGB.
        kernel_size: Размер ядра для размытия при unsharp masking.
        sigma: Стандартное отклонение для гауссова размытия при unsharp masking.
        amount: Коэффициент усиления для unsharp masking.
    Returns:
        np.ndarray: Изображение после применения unsharp masking.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def gamma_correction(image: np.ndarray, gamma: float = 10.0) -> np.ndarray:
    """Применяет гамма-коррекцию для улучшения видимости линий на изображении.

    Args:
        image: Входное изображение в формате BGR или RGB.
        gamma: Параметр гамма-коррекции. Значение > 1 делает изображение темнее, < 1 - светлее.
    Returns:
        np.ndarray: Изображение после применения гамма-коррекции.

        Note:
            Функция преобразует изображение в цветовое пространство LAB, применяет гамма-коррекцию
            только к каналу L (яркость) и возвращает результат в grayscale.
            Этот метод используется только для улучшения видимости линий
            на изображении и не сохраняет цветовую информацию.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l_channel, _, _ = cv2.split(lab)

    blur = cv2.GaussianBlur(l_channel, (1, 1), 0)
    lut = (np.arange(256) / 255) ** gamma * 255
    gamma_img = cv2.LUT(blur, lut.astype(np.uint8))
    return gamma_img


def binary_threshold(gray: np.ndarray, block_size: int = 15, C: int = 5) -> np.ndarray:
    """Преобразует изображение в бинарное с помощью адаптивного порога.

    Args:
        gray: Входное изображение в оттенках серого.
        block_size: Размер блока для адаптивного порога.
        C: Константа, вычитаемая из среднего значения.
    Returns:
        np.ndarray: Бинаризованное изображение, где белые пиксели соответствуют линиям и другим элементам.
    """
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    dilate_img = cv2.dilate(bin_img, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)), iterations=1)
    return dilate_img


def compute_horizontal_line_mask(binary_image: np.ndarray, scale: int = 50, iterations: int = 1) -> np.ndarray:
    """Вычисляет маску горизонтальных линий на изображении.

    Args:
        binary_image: Входное бинаризованное изображение.
        scale: Масштаб для определения размера структурного элемента при морфологической обработке.
    Returns:
        np.ndarray: Бинарная маска, где белые пиксели соответствуют горизонтальным линиям на изображении.
    """

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(binary_image.shape[1] / scale), 1))
    horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horiz_kernel, iterations=iterations)

    return horizontal_lines


def dilate_image(image: np.ndarray, kernel_size: tuple[int, int] = (1, 1), iterations: int = 1) -> np.ndarray:
    """Применяет операцию дилатации к изображению.

    Args:
        image: Входное бинаризованное изображение.
        kernel_size: Размер структурного элемента для дилатации.
        iterations: Количество итераций дилатации.
    Returns:
        np.ndarray: Изображение после применения дилатации.
    """
    return cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size), iterations=iterations)


def compute_vertical_line_mask(binary_image: np.ndarray, median_height: int = 0) -> np.ndarray:
    """Вычисляет маску вертикальных линий на изображении.

    Args:
        binary_image: Входное бинаризованное изображение.
        median_height: Медианная высота линий, используемая для определения размера структурного
        элемента при морфологической обработке.
    Returns:
        np.ndarray: Бинарная маска, где белые пиксели соответствуют вертикальным линиям на изображении.
    """

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, median_height))
    vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    return vertical_lines


def get_mask(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Объединяет две маски с помощью логического ИЛИ.

    Args:
        mask1: Первая бинарная маска.
        mask2: Вторая бинарная маска.
    Returns:
        np.ndarray: Объединённая бинарная маска, где белые пиксели соответствуют элементам,
            присутствующим в обеих масках.
    """
    return cv2.bitwise_or(mask1, mask2)


def find_contours(binary_image: np.ndarray) -> list[np.ndarray]:
    """Находит контуры на бинаризованном изображении.

    Args:
        binary_image: Входное бинаризованное изображение, на котором нужно найти контуры.
    Returns:
        list[np.ndarray]: Список найденных контуров, представленных в виде массивов точек.
    """
    return cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]


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
