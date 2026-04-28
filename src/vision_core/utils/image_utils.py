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


def binary_threshold(gray: np.ndarray) -> np.ndarray:
    """Преобразует изображение в бинарное с помощью адаптивного порога.

    Args:
        gray: Входное изображение в оттенках серого.
    Returns:
        np.ndarray: Бинаризованное изображение, где белые пиксели соответствуют линиям и другим элементам.
    """
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 50)


def compute_horizontal_line_mask(binary_image: np.ndarray, scale: int = 50) -> np.ndarray:
    """Вычисляет маску горизонтальных линий на изображении.

    Args:
        binary_image: Входное бинаризованное изображение.
        scale: Масштаб для определения размера структурного элемента при морфологической обработке.
    Returns:
        np.ndarray: Бинарная маска, где белые пиксели соответствуют горизонтальным линиям на изображении.
    """

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(binary_image.shape[1] / scale), 1))
    horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

    return horizontal_lines


def find_contours(binary_image: np.ndarray) -> list[np.ndarray]:
    """Находит контуры на бинаризованном изображении.

    Args:
        binary_image: Входное бинаризованное изображение, на котором нужно найти контуры.
    Returns:
        list[np.ndarray]: Список найденных контуров, представленных в виде массивов точек.
    """
    return cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]


def get_median_line_height(contours) -> float:
    """Вычисляет медианную высоту линий по контурам.

    Args:
        contours: Список контуров, найденных на изображении.
    Returns:
        float: Медианная высота линий, вычисленная на основе высот ограничивающих прямоугольников контуров.
    """

    heights = [cv2.boundingRect(cnt)[3] for cnt in contours]

    median_height = int(np.median(heights))

    return median_height


def fit_line(cnt) -> tuple[float, float, float, float]:
    """Применяет функцию cv2.fitLine для получения параметров линии, аппроксимирующей контур.

    Args:
        cnt: Контур, представленный в виде массива точек.
    Returns:
        tuple: Параметры линии (vx, vy, x0, y0), где (vx, vy) - вектор направления линии,
               (x0, y0) - точка на линии.
    """
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)

    return vx.item(), vy.item(), x0.item(), y0.item()


# TODO: дублирует логику TablePreprocessor._detect_lines — рефакторить при оптимизации
def compute_raw_line_mask(
    gray: np.ndarray,
    scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Вычисляет сырые маски горизонтальных и вертикальных линий.

    Returns:
        (h_mask, v_mask)
    """
    binary = binary_threshold(gray)
    h_img = binary.copy()
    v_img = binary.copy()

    h_size = int(h_img.shape[1] / scale)
    h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))

    h_erode_img = cv2.erode(h_img, h_structure, 2)
    h_dilate_img = cv2.dilate(h_erode_img, h_structure, 2)

    v_size = int(v_img.shape[0] / scale)
    v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    v_erode_img = cv2.erode(v_img, v_structure, 2)
    v_dilate_img = cv2.dilate(v_erode_img, v_structure, 2)

    return h_dilate_img, v_dilate_img


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
