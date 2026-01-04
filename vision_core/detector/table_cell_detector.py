import cv2
import numpy as np
from typing import Optional
from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.config import TableCellDetectorConfig

from loguru import logger


class TableCellDetector:
    """Извлекатель ячеек из таблицы"""

    def __init__(self, cfg: Optional[TableCellDetectorConfig] = None):
        """
        Args:
            cfg: Конфигурация детектора ячеек
        """
        if cfg is None:
            cfg = TableCellDetectorConfig()

        self.min_cell = cfg.min_cell
        self.padding = cfg.padding
        self.threshold_line = cfg.threshold_line
        self.coverage_thr = cfg.coverage_thr
        self.hough_threshold = cfg.hough_threshold
        self.min_line_length = cfg.min_line_length
        self.max_line_gap = cfg.max_line_gap
        self.line_angle_threshold = cfg.line_angle_threshold

    def extract_cells(
        self,
        table_mask: np.ndarray,
        table_bbox: tuple[int, int, int, int],
        merge_mode: Optional[str] = None,
    ) -> list[Cell]:
        """Извлекает ячейки из маски сырых таблиц. Под сырыми подразумевается
           все квадраты, попавшие под условия.

        Args:
            table_mask (np.ndarray): маска таблицы
            table_bbox (tuple[int, int, int, int]): ограничивающий прямоугольник таблицы
            merge_mode (Optional[str], optional): Режим объединения ячеек:
                - None: без объединения
                - "cols": объединение по столбцам
                - "rows": объединение по строкам
                - "all": полное объединение

        Returns:
            list[Cell]: Список ячеек структурой Cell
        """

        y_coords, x_coords = self._build_grid_coordinates_from_mask(table_mask)

        return self._extract_cells_by_mode(
            table_mask, x_coords, y_coords, table_bbox, merge_mode
        )

    def _build_grid_coordinates_from_mask(
        self,
        roi_mask: np.ndarray,
    ) -> tuple[list[int], list[int]]:
        """
        Строит координаты сетки напрямую из бинарной маски линий.
        Идея: берём профили (доля белых пикселей) по X и по Y, находим "полосы" линий,
        схлопываем в центры и фильтруем близкие координаты.
        """
        if roi_mask.ndim != 2:
            raise ValueError("roi_mask must be 2D (single-channel)")

        h, w = roi_mask.shape[:2]
        bin_img = (roi_mask > 0).astype(np.uint8)

        # Профили: сколько белого в каждой колонке/строке
        x_profile = bin_img.mean(axis=0)  # shape: (w,)
        y_profile = bin_img.mean(axis=1)  # shape: (h,)

        # Адаптивные пороги: не фиксированные магические числа
        x_thr = max(0.003, 0.25 * float(x_profile.max()))
        y_thr = max(0.003, 0.25 * float(y_profile.max()))

        x_idxs = np.where(x_profile >= x_thr)[0]
        y_idxs = np.where(y_profile >= y_thr)[0]

        def _runs_to_centers(idxs: np.ndarray) -> list[int]:
            if idxs.size == 0:
                return []
            centers: list[int] = []
            start = int(idxs[0])
            prev = int(idxs[0])
            for v in idxs[1:]:
                v = int(v)
                if v == prev + 1:
                    prev = v
                else:
                    centers.append((start + prev) // 2)
                    start = prev = v
            centers.append((start + prev) // 2)
            return centers

        x_coords = _runs_to_centers(x_idxs)
        y_coords = _runs_to_centers(y_idxs)

        # Границы ROI (чтобы сетка была замкнутой даже если крайняя линия не попала в профиль)
        if 0 not in x_coords:
            x_coords = [0] + x_coords
        if (w - 1) not in x_coords:
            x_coords = x_coords + [w - 1]

        if 0 not in y_coords:
            y_coords = [0] + y_coords
        if (h - 1) not in y_coords:
            y_coords = y_coords + [h - 1]

        x_coords = sorted(set(int(x) for x in x_coords))
        y_coords = sorted(set(int(y) for y in y_coords))

        # Фильтруем слишком близкие координаты
        min_dist = max(2, int(self.min_cell))
        x_coords = self._filter_close_coordinates(x_coords, min_dist)
        y_coords = self._filter_close_coordinates(y_coords, min_dist)

        return y_coords, x_coords

    def _filter_close_coordinates(
        self,
        coords: list[int],
        min_distance: int,
    ) -> list[int]:
        """Фильтрует слишком близкие координаты"""
        filtered = []

        for i, coord in enumerate(coords):
            if i == 0 or (coord - filtered[-1]) >= min_distance:
                filtered.append(coord)

        return filtered

    def _extract_cells_by_mode(
        self,
        roi_mask: np.ndarray,
        x_coords: list[int],
        y_coords: list[int],
        table_bbox: tuple[int, int, int, int],
        merge_mode: Optional[str],
    ) -> list[Cell]:
        """Выбирает метод извлечения ячеек в зависимости от режима"""
        if merge_mode is None:
            return self._extract_base_cells(x_coords, y_coords, table_bbox)

        v_gaps, h_gaps = self._build_gap_matrices(roi_mask, x_coords, y_coords)

        if merge_mode == "cols":
            return self._extract_merged_cols(x_coords, y_coords, table_bbox, v_gaps)
        elif merge_mode == "rows":
            return self._extract_merged_rows(x_coords, y_coords, table_bbox, h_gaps)
        else:  # "all"
            return self._extract_fully_merged(
                x_coords, y_coords, table_bbox, v_gaps, h_gaps
            )

    def _extract_base_cells(
        self,
        x_coords: list[int],
        y_coords: list[int],
        table_bbox: tuple[int, int, int, int],
    ) -> list[Cell]:
        """Извлекает базовые ячейки без объединения"""
        cells = []

        for i in range(len(y_coords) - 1):
            for j in range(len(x_coords) - 1):
                bbox = BBox(
                    x_min=x_coords[j] + table_bbox[0],
                    y_min=y_coords[i] + table_bbox[1],
                    x_max=x_coords[j + 1] + table_bbox[0],
                    y_max=y_coords[i + 1] + table_bbox[1],
                )

                cells.append(Cell(row=i, col=j, bbox=bbox, colspan=1, rowspan=1))

        return cells

    def _build_gap_matrices(
        self,
        mask: np.ndarray,
        x_coords: list[int],
        y_coords: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Строит матрицы вертикальных и горизонтальных разделителей между ячейками таблицы.
        Метод анализирует бинарную маску с линиями таблицы и определяет, где присутствуют
        физические границы между ячейками. Использует морфологические операции для выделения
        горизонтальных и вертикальных линий, а затем проверяет их наличие в окрестности
        заданных координат сетки.

        Args:
            mask (np.ndarray): Бинарная маска изображения с линиями таблицы (H x W).
            x_coords (list[int]): Список x-координат вертикальных линий сетки таблицы,
                отсортированный по возрастанию. Определяет границы столбцов.
            y_coords (list[int]): Список y-координат горизонтальных линий сетки таблицы,
                отсортированный по возрастанию. Определяет границы строк.

        Returns:
            tuple[np.ndarray, np.ndarray]: Кортеж из двух булевых матриц:
            - vertical_gaps: Матрица размером (n_rows, n_cols-1), где vertical_gaps[i, j]=True
                означает наличие вертикальной границы между столбцами j и j+1 в строке i.
            - horizontal_gaps: Матрица размером (n_rows-1, n_cols), где horizontal_gaps[i, j]=True
                означает наличие горизонтальной границы между строками i и i+1 в столбце j.
            Если сетка пустая (n_rows<=0 или n_cols<=0), возвращаются пустые матрицы (0, 0).
        """

        n_rows = len(y_coords) - 1
        n_cols = len(x_coords) - 1

        if n_rows <= 0 or n_cols <= 0:
            return np.zeros((0, 0), dtype=bool), np.zeros((0, 0), dtype=bool)

        vertical_gaps = np.zeros((n_rows, max(0, n_cols - 1)), dtype=bool)
        horizontal_gaps = np.zeros((max(0, n_rows - 1), n_cols), dtype=bool)

        h, w = mask.shape[:2]
        bin_img = (mask > 0).astype(np.uint8) * 255

        # Оценим типичный размер ячейки
        dx = np.diff(np.array(x_coords, dtype=np.int32))
        dy = np.diff(np.array(y_coords, dtype=np.int32))
        med_dx = int(np.median(dx[dx > 0])) if np.any(dx > 0) else max(10, w // 20)
        med_dy = int(np.median(dy[dy > 0])) if np.any(dy > 0) else max(10, h // 20)

        # Ядра для выделения ориентированных линий
        k_h = max(10, int(0.6 * med_dx))
        k_v = max(10, int(0.6 * med_dy))

        # Сначала слегка соединяем разрывы
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # Затем выделяем горизонтали/вертикали
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_v))

        h_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, h_kernel, iterations=1)
        v_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, v_kernel, iterations=1)

        # Допуск вокруг координаты линии
        tol = max(2, int(self.padding), int(self.threshold_line))

        # Вертикальные границы
        for row in range(n_rows):
            y1 = max(int(y_coords[row]), 0)
            y2 = min(int(y_coords[row + 1]), h)
            if y2 <= y1:
                continue

            for col in range(n_cols - 1):
                x_pos = int(x_coords[col + 1])
                x1 = max(x_pos - tol, 0)
                x2 = min(x_pos + tol + 1, w)
                if x2 <= x1:
                    continue

                roi = v_mask[y1:y2, x1:x2] > 0
                rows_with_line = np.any(roi, axis=1)

                if (
                    rows_with_line.size
                    and float(rows_with_line.mean()) >= self.coverage_thr
                ):
                    vertical_gaps[row, col] = True

        # Горизонтальные границы
        for row in range(n_rows - 1):
            y_pos = int(y_coords[row + 1])
            y1 = max(y_pos - tol, 0)
            y2 = min(y_pos + tol + 1, h)
            if y2 <= y1:
                continue

            for col in range(n_cols):
                x1 = max(int(x_coords[col]), 0)
                x2 = min(int(x_coords[col + 1]), w)
                if x2 <= x1:
                    continue

                roi = h_mask[y1:y2, x1:x2] > 0
                cols_with_line = np.any(roi, axis=0)

                if (
                    cols_with_line.size
                    and float(cols_with_line.mean()) >= self.coverage_thr
                ):
                    horizontal_gaps[row, col] = True

        return vertical_gaps, horizontal_gaps

    def _extract_merged_cols(
        self,
        x_coords: list[int],
        y_coords: list[int],
        table_bbox: tuple[int, int, int, int],
        vertical_gaps: np.ndarray,
    ) -> list[Cell]:
        """Извлекает ячейки с объединением по столбцам"""
        cells = []
        n_rows = len(y_coords) - 1

        for row in range(n_rows):
            col = 0
            while col < len(x_coords) - 1:
                colspan = 1

                # Определяем сколько столбцов можно объединить
                while (
                    col + colspan < len(x_coords) - 1
                    and not vertical_gaps[row, col + colspan - 1]
                ):
                    colspan += 1

                bbox = BBox(
                    x_min=x_coords[col] + table_bbox[0],
                    y_min=y_coords[row] + table_bbox[1],
                    x_max=x_coords[col + colspan] + table_bbox[0],
                    y_max=y_coords[row + 1] + table_bbox[1],
                )

                cells.append(
                    Cell(row=row, col=col, bbox=bbox, colspan=colspan, rowspan=1)
                )

                col += colspan

        return cells

    def _extract_merged_rows(
        self,
        x_coords: list[int],
        y_coords: list[int],
        table_bbox: tuple[int, int, int, int],
        horizontal_gaps: np.ndarray,
    ) -> list[Cell]:
        """Извлекает ячейки с объединением по строкам"""
        cells = []
        n_cols = len(x_coords) - 1

        for col in range(n_cols):
            row = 0
            while row < len(y_coords) - 1:
                rowspan = 1

                # Определяем сколько строк можно объединить
                while (
                    row + rowspan < len(y_coords) - 1
                    and not horizontal_gaps[row + rowspan - 1, col]
                ):
                    rowspan += 1

                bbox = BBox(
                    x_min=x_coords[col] + table_bbox[0],
                    y_min=y_coords[row] + table_bbox[1],
                    x_max=x_coords[col + 1] + table_bbox[0],
                    y_max=y_coords[row + rowspan] + table_bbox[1],
                )

                cells.append(
                    Cell(row=row, col=col, bbox=bbox, colspan=1, rowspan=rowspan)
                )

                row += rowspan

        return cells

    def _extract_fully_merged(
        self,
        x_coords: list[int],
        y_coords: list[int],
        table_bbox: tuple[int, int, int, int],
        vertical_gaps: np.ndarray,
        horizontal_gaps: np.ndarray,
    ) -> list[Cell]:
        """Извлекает полностью объединенные ячейки"""
        # Извлекаем отдельно по столбцам и строкам
        col_cells = self._extract_merged_cols(
            x_coords, y_coords, table_bbox, vertical_gaps
        )
        row_cells = self._extract_merged_rows(
            x_coords, y_coords, table_bbox, horizontal_gaps
        )

        # Создаем словарь для быстрого доступа к ячейкам по столбцам
        col_cells_dict = {(cell.row, cell.col): cell for cell in col_cells}

        # Объединяем результаты
        merged_cells = []

        for row_cell in row_cells:
            key = (row_cell.row, row_cell.col)

            if key in col_cells_dict:
                col_cell = col_cells_dict[key]

                merged_cells.append(
                    Cell(
                        row=row_cell.row,
                        col=row_cell.col,
                        bbox=BBox(
                            x_min=col_cell.bbox.x_min,
                            y_min=row_cell.bbox.y_min,
                            x_max=col_cell.bbox.x_max,
                            y_max=row_cell.bbox.y_max,
                        ),
                        colspan=col_cell.colspan,
                        rowspan=row_cell.rowspan,
                    )
                )

        return merged_cells
