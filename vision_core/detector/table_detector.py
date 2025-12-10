import numpy as np
import cv2
from loguru import logger
from vision_core.entities.table import Table
from vision_core.entities.cell import Cell
from vision_core.entities.bbox import BBox


class TableDetector:
    """Детектор таблиц на изображении"""

    def __init__(
        self,
        min_line_length: int = 50,
        max_line_gap: int = 10,
        min_table_area: int = 10000,
    ):
        """
        Args:
            min_line_length: Минимальная длина линии
            max_line_gap: Максимальный разрыв в линии
            min_table_area: Минимальная площадь таблицы
        """
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.min_table_area = min_table_area

    def extract_tables(self, image: np.ndarray):
        """
        Детектирует таблицы на изображении

        Args:
            image: Grayscale изображение

        Returns:
            Список детектированных таблиц
        """
        logger.info("Начало детекции таблиц на изображении")
        tables = []

        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[
            1
        ]
        # Детектируем линии
        h_mask = self._detect_horizontal_lines(binary)
        v_mask = self._detect_vertical_lines(binary)

        # Создаём маску таблицы
        table_mask = self._create_table_mask(h_mask, v_mask)
        # Находим контуры таблиц
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_table_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            table_bbox = BBox(x, y, x + w, y + h)
            table = Table(
                bbox=table_bbox, cells=[]
            )  # Cells можно детектировать отдельно
            tables.append(table)
            logger.debug(f"Таблица обнаружена с площадью {area} и BBox {table_bbox}")

        logger.info(f"Детекция завершена. Найдено таблиц: {len(tables)}")
        return tables

    def _find_tables_components(self, table_mask: np.ndarray):
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes: list[BBox] = []
        for cnt in contours:
            c_poly = cv2.approxPolyDP(cnt, 3, True)
            if len(c_poly) < 4:
                continue
            x, y, w, h = cv2.boundingRect(c_poly)
            if w < 500 or h < 50:
                continue
            bbox = BBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h)
            bboxes.append(bbox)
        return bboxes

    def _create_table_mask(self, h_mask: np.ndarray, v_mask: np.ndarray):
        # удаляем шумы
        intersec = cv2.bitwise_and(h_mask, v_mask)
        cleaned_h_mask = self._clean_mask(h_mask, intersec)
        cleaned_v_mask = self._clean_mask(v_mask, intersec)
        cleaned_mask = cv2.add(cleaned_h_mask, cleaned_v_mask)

        return cleaned_mask

    def _clean_mask(
        self,
        mask: np.ndarray,
        intersec: np.ndarray,
        min_length: int = 120,
        min_intersections: int = 1,
    ):
        num_mask_labels, labels_mask, stats_mask, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )

        output_mask = np.zeros_like(mask, dtype=np.uint8)

        for lbl in range(
            1, num_mask_labels
        ):  # Начинаем с 1, чтобы пропустить фон (метка 0)
            length = stats_mask[lbl, cv2.CC_STAT_AREA]
            if length < min_length:
                continue

            comp_boolean_mask = labels_mask == lbl

            current_comp_intersections_map_bool = comp_boolean_mask & (intersec > 0)

            if not np.any(current_comp_intersections_map_bool):
                crosses = 0
            else:
                num_intersection_blobs, _ = cv2.connectedComponents(
                    current_comp_intersections_map_bool.astype(np.uint8), connectivity=8
                )
                crosses = num_intersection_blobs - 1

            if crosses > min_intersections:
                output_mask[comp_boolean_mask] = 255

        return output_mask

    def _detect_horizontal_lines(self, binary: np.ndarray):
        h, w = binary.shape
        kernel_width = max(10, min(50, w // 50))  # не меньше 10, не больше 50
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        reconstructed_horizontal = cv2.dilate(horizontal, kernel_dilate, iterations=2)

        return reconstructed_horizontal

    def _detect_vertical_lines(self, binary: np.ndarray):
        h, w = binary.shape
        kernel_height = max(5, min(50, h // 250))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        reconstructed_vertical = cv2.dilate(vertical, kernel_dilate, iterations=2)

        return reconstructed_vertical
