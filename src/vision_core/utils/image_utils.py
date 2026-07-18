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


def apply_gamma_correction(rgb: np.ndarray, gamma: float = 1.2):
    def get_table(gamma):
        inv = 1.0 / gamma
        table = (np.linspace(0, 1, 256) ** inv) * 255.0
        return table.astype("uint8")

    table = get_table(gamma)
    return cv2.LUT(rgb, table)


def binary_masked(gray: np.ndarray, k_gauss: int = 5, block_size: int = 11, c: int = 2):
    """_summary_

    Args:
        gray (np.ndarray): _description_
        k_gauss (int, optional): _description_. Defaults to 5.
        block_size (int, optional): _description_. Defaults to 11.
        c (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    blur = cv2.GaussianBlur(gray, (k_gauss, k_gauss), 0)

    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)
    return binary


def extract_lines_mask(bin_img: np.ndarray, v_scale: int = 10, h_scale: int = 40):
    """Шаг 2: MORPH_OPEN длинными осевыми ядрами -> убираем текст, оставляем линии сетки."""
    h, w = bin_img.shape

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 40))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))

    h_opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horizontal_kernel)
    v_opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vertical_kernel)

    mask = cv2.bitwise_or(h_opened, v_opened)
    return mask


def erase_lines(mask, lines, thickness=3):
    mask_clean = mask.copy()
    if lines.size == 0:
        return mask_clean
    pts = lines.reshape(-1, 2, 2)
    cv2.polylines(mask_clean, list(pts), isClosed=False, color=0, thickness=thickness)
    return mask_clean


def repair_mask(mask, close_size=7):
    """MORPH_CLOSE -> срастить микроразрывы."""
    close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, 1))
    close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_size))

    mask_fixed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_h)
    mask_fixed = cv2.morphologyEx(mask_fixed, cv2.MORPH_CLOSE, close_v)
    return mask_fixed


def subtract_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image, cv2.bitwise_not(mask))


def dilate_image(image: np.ndarray, kernel_size: tuple[int, int] = (1, 1), iterations: int = 1) -> np.ndarray:
    return cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size), iterations=iterations)


def filter_long_intersecting(
    mask: np.ndarray, dilated_other: np.ndarray, min_length: int, is_vertical: bool
) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    stat_col = cv2.CC_STAT_HEIGHT if is_vertical else cv2.CC_STAT_WIDTH
    long_enough = stats[:, stat_col] >= min_length
    touching = _components_touching(labels, dilated_other > 0, n)

    keep_labels = np.flatnonzero(long_enough & touching)
    keep_labels = keep_labels[keep_labels != 0]  # 0 — фон

    return _mask_from_labels(labels, keep_labels)


def _components_touching(labels: np.ndarray, other_mask: np.ndarray, n_labels: int) -> np.ndarray:
    return np.bincount(labels[other_mask], minlength=n_labels) > 0


def _mask_from_labels(labels: np.ndarray, keep_labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, keep_labels).astype(np.uint8) * 255


def morph_open(mask: np.ndarray, k_size=(1, 1)) -> np.ndarray:
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            ksize=k_size,
        ),
    )


def has_missing_text(content_mask: np.ndarray, min_area: int = 15, min_side: int = 4) -> bool:
    """True, если в content_mask есть остаточный регион (нераспознанный текст)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(content_mask)
    if n <= 1:
        return False

    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]

    return bool(np.any((areas >= min_area) & (widths >= min_side) & (heights >= min_side)))


# ------------------- legacy --------------------------
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
