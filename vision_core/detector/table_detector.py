import numpy as np
import cv2
from typing import Optional
from vision_core.entities.bbox import BBox
from vision_core.entities.table import Table
from vision_core.detector.table_cell_detector import TableCellDetector
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.config import VisionCoreConfig
from loguru import logger


class TableDetector:
    """Детектор таблиц на изображении"""

    def __init__(self, config: Optional[VisionCoreConfig] = None):
        """
        Args:
            config: Конфигурация для всех компонентов распознавания таблиц
        """
        if config is None:
            config = VisionCoreConfig()

        self.config = config
        cfg = config.detector

        self.min_table_area = cfg.min_table_area
        self.approx_poly_eps = cfg.approx_poly_eps
        self.mode_merge_cells = cfg.mode_merge_cells

        self._table_mask: Optional[np.ndarray] = None
        self.preprocessor = TablePreprocessor(config.preprocessor)
        self.table_cell_detector = TableCellDetector(config.cell_detector)

    def detect_tables(self, image: np.ndarray) -> list[Table]:
        # Создаем маску таблицы
        self._table_mask = self.preprocessor.create_table_mask(image)

        # Извлекаем bounding boxes таблиц
        table_bboxes = self.extract_raw_tables(self._table_mask)

        tables: list[Table] = []

        for idx, bbox in enumerate(table_bboxes):
            # Детектируем ячейки внутри таблицы

            cells = self.table_cell_detector.extract_cells(
                self._table_mask,
                bbox,
                merge_mode=self.mode_merge_cells,
            )
            table = Table(
                id=f"table_{idx}",
                bbox=bbox,
                cells=cells,
            )

            if cells:
                table.num_rows = max(cell.row + cell.rowspan for cell in cells)
                table.num_cols = max(cell.col + cell.colspan for cell in cells)

            if not table.is_valid() or not table.validate_structure():
                continue
            logger.debug(
                f"Найдена таблица {table.id}: c {table.num_rows} строк и {table.num_cols} столбцов"
            )
            tables.append(table)

        return tables

    def get_table_line_mask(self, table: Table, padding: int = 0):
        """Возвращает маску линий таблицы внутри заданной таблицы.
        Мы можем использовать эту маску для удаления линий перед распознаванием текста.
        Также можно указать padding вокруг линий, чтобы захватить больше пространства.
        """
        if self._table_mask is None:
            raise ValueError(
                "Маска таблиц не доступна. Сначала вызовите detect_tables()"
            )
        roi_mask = table.bbox.padding(padding).roi(self._table_mask)
        return roi_mask

    def remove_table_lines(
        self,
        image: np.ndarray,
        table: Table,
        fill_value: int = 255,
        padding: int = 2,
    ) -> np.ndarray:
        """
        Удаляет линии таблицы из изображения.
        Используется перед распознаванием текста.
        Используйте в любом случае, если вы распознаете текст внутри таблицы или текст только из ячеек.
        Эффективность достигает еще с примененением расширения (dilation) маски линий таблицы.
        Вы можете применить дополнительную обработку к изображению после удаления линий, если это необходимо.
        Например, можно применить морфологическое закрытие (morphological closing) для восстановления текста.
        Либо удалить шум с помощью медианного фильтра (median filter) .

        Args:
            image: Исходное изображение (BGR или Grayscale)
            table: Объект таблицы
            fill_value: Значение для заполнения удаленных линий
            padding: Отступ вокруг линий для удаления

        Returns:
            Изображение с удаленными линиями таблицы
        """
        result = image.copy()
        line_mask = self.get_table_line_mask(table)
        roi = table.bbox.roi(result)

        if roi.shape[:2] != line_mask.shape:
            line_mask = cv2.resize(
                line_mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        # Расширяем маску если нужен padding
        if padding > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (padding * 2 + 1, padding * 2 + 1)
            )
            line_mask = cv2.dilate(line_mask, kernel, iterations=1)

        roi[line_mask > 0] = fill_value

        return result

    def extract_raw_tables(self, table_mask: np.ndarray) -> list[BBox]:
        """Извлекает BBox'ы таблиц из маски"""
        return self._find_tables(table_mask)

    def _find_tables(self, table_mask: np.ndarray):
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes: list[BBox] = []
        for cnt in contours:
            arclen = cv2.arcLength(cnt, True)
            eps = self.approx_poly_eps * arclen
            c_poly = cv2.approxPolyDP(cnt, eps, True)
            if len(c_poly) != 4:
                continue
            if cv2.contourArea(c_poly) < self.min_table_area:
                continue
            pts = c_poly.reshape(4, 2)
            ordered_pts = self._order_points(pts)
            x_min, y_min = ordered_pts[:, 0].min(), ordered_pts[:, 1].min()
            x_max, y_max = ordered_pts[:, 0].max(), ordered_pts[:, 1].max()
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
