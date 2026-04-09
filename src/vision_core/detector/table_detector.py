import cv2
import numpy as np
from loguru import logger

from vision_core.config import TableCellDetectorConfig, TableDetectorConfig, TablePreprocessorConfig
from vision_core.debug_image_observer import DebugImageObserver
from vision_core.detector.table_cell_detector import TableCellDetector
from vision_core.entities.bbox import BBox
from vision_core.entities.table import Table
from vision_core.preprocessor.table_preprocessor import TablePreprocessor


class TableDetector:
    """Детектор таблиц на изображении"""

    def __init__(
        self,
        preprocessor_config: TablePreprocessorConfig | None = None,
        table_detector_config: TableDetectorConfig | None = None,
        cell_detector_config: TableCellDetectorConfig | None = None,
        debug_image: DebugImageObserver | None = None,
    ):
        """
        Args:
            preprocessor_config: Конфигурация для предобработки таблиц.
                Если None, используется конфигурация по умолчанию.
            table_detector_config: Конфигурация для детектора таблиц.
                Если None, используется конфигурация по умолчанию.
            cell_detector_config: Конфигурация для детектора ячеек. Если None, используется конфигурация по умолчанию.
        """
        self.cfg = table_detector_config or TableDetectorConfig()
        self.preprocessor = TablePreprocessor(preprocessor_config, debug_image)
        self.table_cell_detector = TableCellDetector(cell_detector_config)
        self._debug_image = debug_image

    def create_table_mask(self, image: np.ndarray) -> np.ndarray:
        """Создаёт маску таблиц для текущего изображения.

        Args:
            image: Изображение страницы.

        Returns:
            np.ndarray: Бинарная маска таблиц.
        """
        return self.preprocessor.create_table_mask(image)

    def detect_tables(self, image: np.ndarray) -> list[Table]:
        """Детектирует таблицы на изображении и извлекает их ячейки.
        Args:
            image: Изображение страницы.
        Returns:
            list[Table]: Список найденных таблиц с их ячейками.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        # Создаем маску таблицы локально для текущего изображения.
        table_mask = self.create_table_mask(gray_image)

        if self._debug_image:
            self._debug_image.on_debug_image(
                src_image=table_mask,
                stage="3_table_mask",
                prefix="page",
                page_number=0,
            )

        # Извлекаем bounding boxes таблиц
        table_bboxes = self.extract_raw_tables(table_mask)

        tables: list[Table] = []

        for idx, bbox in enumerate(table_bboxes):
            # Детектируем ячейки внутри таблицы
            roi_mask = bbox.roi(table_mask)
            cells = self.table_cell_detector.extract_cells(
                roi_mask,
                bbox.to_tuple(),
                merge_mode=self.cfg.mode_merge_cells,
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
            logger.debug(f"Найдена таблица {table.id}: c {table.num_rows} строк и {table.num_cols} столбцов")
            tables.append(table)

        return tables

    def get_table_line_mask(self, table: Table, table_mask: np.ndarray, padding: int = 0) -> np.ndarray:
        """Возвращает маску линий таблицы внутри заданной таблицы.

        Мы можем использовать эту маску для удаления линий перед распознаванием текста.
        Также можно указать padding вокруг линий, чтобы захватить больше пространства.

        Args:
            table: Таблица, для которой нужно получить маску линий.
            table_mask: Общая маска таблиц для текущего изображения.
            padding: Дополнительный отступ вокруг bbox таблицы.

        Returns:
            np.ndarray: Маска линий внутри области таблицы.
        """
        roi_mask = table.bbox.padding(padding).roi(table_mask)
        return roi_mask

    def remove_table_lines(
        self,
        image: np.ndarray,
        table: Table,
        fill_value: int = 255,
        padding: int = 2,
        table_mask: np.ndarray | None = None,
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
            table_mask: Предвычисленная маска таблиц для текущего изображения.

        Returns:
            Изображение с удаленными линиями таблицы
        """
        result = image.copy()
        current_table_mask = table_mask if table_mask is not None else self.create_table_mask(image)
        line_mask = self.get_table_line_mask(table, current_table_mask)

        if self._debug_image:
            self._debug_image.on_debug_image(
                src_image=line_mask,
                stage="2_table_line_mask",
                prefix="page",
                page_number=0,
            )

        roi = table.bbox.roi(result)

        if roi.shape[:2] != line_mask.shape:
            line_mask = cv2.resize(line_mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Расширяем маску если нужен padding
        if padding > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (padding * 2 + 1, padding * 2 + 1))
            line_mask = cv2.dilate(line_mask, kernel, iterations=1)

        roi[line_mask > 0] = fill_value

        return result

    def extract_raw_tables(self, table_mask: np.ndarray) -> list[BBox]:
        """Извлекает BBox'ы таблиц из маски"""
        return self._find_tables(table_mask)

    def _find_tables(self, table_mask: np.ndarray):
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes: list[BBox] = []
        for cnt in contours:
            arclen = cv2.arcLength(cnt, True)
            eps = self.cfg.approx_poly_eps * arclen
            c_poly = cv2.approxPolyDP(cnt, eps, True)
            if len(c_poly) != 4:
                continue
            if cv2.contourArea(c_poly) < self.cfg.min_table_area:
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
