import numpy as np
import cv2
from typing import List
from vision_core.entities.bbox import BBox


class TableDetector:
    """Детектор таблиц на изображении"""

    DEFAULT_MIN_TABLE_AREA = 10000
    DEFAULT_APPROX_EPS = 0.02
    DEFAULT_MIN_ASPECT_RATIO = 0.3

    def __init__(
        self,
        min_table_area: int = DEFAULT_MIN_TABLE_AREA,
        approx_poly_eps: float = DEFAULT_APPROX_EPS,
        min_aspect_ratio: float = DEFAULT_MIN_ASPECT_RATIO,
        min_vertices: int = 3,
        max_vertices: int = 8,
    ):
        """
        Args:
            min_table_area: Минимальная площадь таблицы
            approx_poly_eps: Коэффициент для аппроксимации контура
            min_aspect_ratio: Минимальное соотношение сторон таблицы
            min_vertices: Минимальное количество вершин в контуре
            max_vertices: Максимальное количество вершин в контуре
        """
        self.min_table_area = min_table_area
        self.approx_poly_eps = approx_poly_eps
        self.min_aspect_ratio = min_aspect_ratio
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices

    def extract_raw_tables(self, table_mask: np.ndarray) -> List[BBox]:
        """Извлекает BBox'ы таблиц из маски"""
        return self._find_tables(table_mask)

    def _find_tables(self, table_mask: np.ndarray) -> List[BBox]:
        """Находит таблицы на маске"""
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes: List[BBox] = []

        for cnt in contours:
            # Аппроксимируем контур
            arclen = cv2.arcLength(cnt, True)
            eps = self.approx_poly_eps * arclen
            c_poly = cv2.approxPolyDP(cnt, eps, True)

            # Проверяем количество вершин
            vertices = len(c_poly)
            if not (self.min_vertices <= vertices <= self.max_vertices):
                continue

            # Проверяем площадь
            area = cv2.contourArea(c_poly)
            if area < self.min_table_area:
                continue

            # Получаем bounding box
            if vertices == 4:
                # Для четырехугольника упорядочиваем точки
                pts = c_poly.reshape(4, 2)
                ordered_pts = self._order_points(pts)
                x_min, y_min = ordered_pts[:, 0].min(), ordered_pts[:, 1].min()
                x_max, y_max = ordered_pts[:, 0].max(), ordered_pts[:, 1].max()
            else:
                # Для других форм берем bounding rect
                x, y, w, h = cv2.boundingRect(c_poly)
                x_min, y_min = x, y
                x_max, y_max = x + w, y + h

            # Проверяем соотношение сторон
            width = x_max - x_min
            height = y_max - y_min

            if width == 0 or height == 0:
                continue

            aspect_ratio = min(width, height) / max(width, height)
            if aspect_ratio < self.min_aspect_ratio:
                continue

            bboxes.append(BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))

        return bboxes

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Упорядочивает точки четырехугольника"""
        # Сортируем точки по сумме координат (самая левая верхняя имеет наименьшую сумму)
        sums = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        # Левый верхний - минимальная сумма
        tl_idx = np.argmin(sums)
        # Правый нижний - максимальная сумма
        br_idx = np.argmax(sums)

        # Левый нижний - максимальная разность (y - x)
        bl_idx = np.argmax(diff)
        # Правый верхний - минимальная разность (y - x)
        tr_idx = np.argmin(diff)

        return np.array(
            [
                pts[tl_idx],  # Левый верхний
                pts[tr_idx],  # Правый верхний
                pts[br_idx],  # Правый нижний
                pts[bl_idx],  # Левый нижний
            ],
            dtype="float32",
        )
