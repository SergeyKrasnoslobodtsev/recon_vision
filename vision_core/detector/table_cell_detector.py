import cv2
import numpy as np
from typing import Optional
from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.config import TableCellDetectorConfig


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
        self.hough_threshold = cfg.hough_threshold
        self.min_line_length = cfg.min_line_length
        self.max_line_gap = cfg.max_line_gap
        self.line_angle_threshold = cfg.line_angle_threshold

    def extract_cells(
        self,
        table_mask: np.ndarray,
        table_bbox: BBox,
        merge_mode: Optional[str] = None,
    ) -> list[Cell]:
        """
        Извлекает ячейки из таблицы.

        Args:
            merge_mode: Режим объединения ячеек:
                - None: без объединения
                - "cols": объединение по столбцам
                - "rows": объединение по строкам
                - "all": полное объединение
        """
        roi_mask = table_bbox.roi(table_mask)

        # 1. Находим линии
        h_lines, v_lines = self._detect_lines(roi_mask)

        # 2. Группируем близкие линии
        h_lines = self._group_lines(h_lines, axis="y")
        v_lines = self._group_lines(v_lines, axis="x")

        # 3. Получаем координаты сетки
        y_coords, x_coords = self._build_grid_coordinates(
            h_lines, v_lines, roi_mask.shape
        )

        # 4. Извлекаем ячейки в зависимости от режима
        return self._extract_cells_by_mode(
            roi_mask, x_coords, y_coords, table_bbox, merge_mode
        )

    def _detect_lines(self, mask: np.ndarray) -> tuple[list, list]:
        """Обнаруживает горизонтальные и вертикальные линии на маске"""
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return [], []

        h_lines, v_lines = [], []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # Корректное определение горизонтальных/вертикальных линий
            if (
                angle <= self.line_angle_threshold
                or angle >= 180 - self.line_angle_threshold
            ):
                h_lines.append((x1, y1, x2, y2))
            else:
                v_lines.append((x1, y1, x2, y2))

        return h_lines, v_lines

    def _group_lines(
        self,
        lines: list[tuple[float, float, float, float]],
        axis: str,
    ) -> list[tuple[float, float, float, float]]:
        """Группирует близкие линии по указанной оси"""
        if not lines:
            return []

        # Определяем ключ сортировки в зависимости от оси
        if axis == "y":
            lines.sort(key=lambda line: (line[1] + line[3]) / 2)
            positions = [(line[1] + line[3]) / 2 for line in lines]
        else:  # 'x'
            lines.sort(key=lambda line: (line[0] + line[2]) / 2)
            positions = [(line[0] + line[2]) / 2 for line in lines]

        # Группируем линии
        groups = []
        current_group = [lines[0]]

        for i in range(1, len(lines)):
            if abs(positions[i] - positions[i - 1]) <= self.threshold_line:
                current_group.append(lines[i])
            else:
                groups.append(current_group)
                current_group = [lines[i]]

        if current_group:
            groups.append(current_group)

        # Усредняем линии в группах
        return self._average_lines_in_groups(groups, axis)

    def _average_lines_in_groups(
        self,
        groups: list[list[tuple]],
        axis: str,
    ) -> list[tuple]:
        """Усредняет линии в группах"""
        result = []

        for group in groups:
            if axis == "y":
                avg_y = np.mean([(line[1] + line[3]) / 2 for line in group])
                min_x = min(min(line[0], line[2]) for line in group)
                max_x = max(max(line[0], line[2]) for line in group)
                result.append((min_x, avg_y, max_x, avg_y))
            else:  # 'x'
                avg_x = np.mean([(line[0] + line[2]) / 2 for line in group])
                min_y = min(min(line[1], line[3]) for line in group)
                max_y = max(max(line[1], line[3]) for line in group)
                result.append((avg_x, min_y, avg_x, max_y))

        return result

    def _build_grid_coordinates(
        self,
        h_lines: list,
        v_lines: list,
        roi_shape: tuple[int, int],
    ) -> tuple[list[int], list[int]]:
        """Строит координаты сетки ячеек"""
        # Извлекаем Y-координаты из горизонтальных линий
        y_coords = sorted(set(int(line[1]) for line in h_lines))
        x_coords = sorted(set(int(line[0]) for line in v_lines))

        # Добавляем границы ROI если нет крайних линий
        if not y_coords or roi_shape[0] - 1 > y_coords[-1]:
            y_coords.append(roi_shape[0] - 1)
        if not x_coords or roi_shape[1] - 1 > x_coords[-1]:
            x_coords.append(roi_shape[1] - 1)

        # Фильтруем слишком близкие координаты
        y_coords = self._filter_close_coordinates(y_coords, self.min_cell)
        x_coords = self._filter_close_coordinates(x_coords, self.min_cell)

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
        table_bbox: BBox,
        merge_mode: Optional[str],
    ) -> list[Cell]:
        """Выбирает метод извлечения ячеек в зависимости от режима"""
        if merge_mode is None:
            return self._extract_base_cells(x_coords, y_coords, table_bbox)

        # Строим матрицу границ один раз для всех режимов
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
        table_bbox: BBox,
    ) -> list[Cell]:
        """Извлекает базовые ячейки без объединения"""
        cells = []

        for i in range(len(y_coords) - 1):
            for j in range(len(x_coords) - 1):
                bbox = BBox(
                    x_min=x_coords[j] + table_bbox.x_min,
                    y_min=y_coords[i] + table_bbox.y_min,
                    x_max=x_coords[j + 1] + table_bbox.x_min,
                    y_max=y_coords[i + 1] + table_bbox.y_min,
                )

                cells.append(Cell(row=i, col=j, bbox=bbox, colspan=1, rowspan=1))

        return cells

    def _build_gap_matrices(
        self,
        mask: np.ndarray,
        x_coords: list[int],
        y_coords: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Строит матрицы наличия вертикальных и горизонтальных границ"""
        n_rows = len(y_coords) - 1
        n_cols = len(x_coords) - 1

        # Матрицы: True если есть граница между ячейками
        vertical_gaps = np.zeros((n_rows, n_cols - 1), dtype=bool)
        horizontal_gaps = np.zeros((n_rows - 1, n_cols), dtype=bool)

        # Заполняем матрицу вертикальных границ
        for row in range(n_rows):
            for col in range(n_cols - 1):
                x_pos = x_coords[col + 1]
                y_min = max(y_coords[row] + self.padding, 0)
                y_max = min(y_coords[row + 1] - self.padding, mask.shape[0])
                x_min = max(x_pos - self.padding, 0)
                x_max = min(x_pos + self.padding, mask.shape[1])

                if y_min < y_max and x_min < x_max:
                    roi = mask[y_min:y_max, x_min:x_max]
                    if roi.size > 0:
                        vertical_gaps[row, col] = np.any(
                            np.sum(roi, axis=0) > self.threshold_line
                        )

        # Заполняем матрицу горизонтальных границ
        for row in range(n_rows - 1):
            for col in range(n_cols):
                y_pos = y_coords[row + 1]
                y_min = max(y_pos - self.padding, 0)
                y_max = min(y_pos + self.padding, mask.shape[0])
                x_min = max(x_coords[col] + self.padding, 0)
                x_max = min(x_coords[col + 1] - self.padding, mask.shape[1])

                if y_min < y_max and x_min < x_max:
                    roi = mask[y_min:y_max, x_min:x_max]
                    if roi.size > 0:
                        horizontal_gaps[row, col] = np.any(
                            np.sum(roi, axis=1) > self.threshold_line
                        )

        return vertical_gaps, horizontal_gaps

    def _extract_merged_cols(
        self,
        x_coords: list[int],
        y_coords: list[int],
        table_bbox: BBox,
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
                    x_min=x_coords[col] + table_bbox.x_min,
                    y_min=y_coords[row] + table_bbox.y_min,
                    x_max=x_coords[col + colspan] + table_bbox.x_min,
                    y_max=y_coords[row + 1] + table_bbox.y_min,
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
        table_bbox: BBox,
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
                    x_min=x_coords[col] + table_bbox.x_min,
                    y_min=y_coords[row] + table_bbox.y_min,
                    x_max=x_coords[col + 1] + table_bbox.x_min,
                    y_max=y_coords[row + rowspan] + table_bbox.y_min,
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
        table_bbox: BBox,
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
