import cv2
import numpy as np
from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell


class TableCellDetector:
    """Извлекатель ячеек из таблицы"""

    def __init__(
        self,
        min_cell: int = 15,
        padding: int = 10,
        threshold_line: int = 10,
    ):
        """
        Args:
            dist_group_lines: Максимальное расстояние между линиями для группировки
            min_cell: Минимальный размер ячейки
            padding: Отступ при проверке на наличие линии между ячейками
            threshold_line: Порог для определения наличия линии между ячейками
        """
        self.min_cell = min_cell
        self.padding = padding
        self.threshold_line = threshold_line

    def extract_cells(self, table_mask: np.ndarray, table_bbox: BBox, merge_mode=None):
        cells: list[Cell] = []
        roi_mask_table = table_bbox.roi(table_mask)
        h_lines, v_lines = self._find_lines_from_mask(roi_mask_table)
        h_lines = self._group_lines(h_lines, is_horizontal=True)
        v_lines = self._group_lines(v_lines, is_horizontal=False)
        y_coords, x_coords = self._get_grid(h_lines, v_lines)
        if merge_mode is None:
            cells = self._base_extract_cells(x_coords, y_coords, table_bbox)
        elif merge_mode == "cols":
            cells = self._merge_cols_cells(
                roi_mask_table, x_coords, y_coords, table_bbox
            )
        elif merge_mode == "rows":
            cells = self._merge_rows_cells(
                roi_mask_table, x_coords, y_coords, table_bbox
            )
        else:  # all
            cells = self._merge_all_cells(
                roi_mask_table, x_coords, y_coords, table_bbox
            )

        return cells

    def _base_extract_cells(
        self, x_coords: list[int], y_coords: list[int], table_bbox: BBox
    ):
        cells: list[Cell] = []
        for i in range(len(y_coords) - 1):
            for j in range(len(x_coords) - 1):
                y1, y2 = (
                    y_coords[i] + table_bbox.y_min,
                    y_coords[i + 1] + table_bbox.y_min,
                )
                x1, x2 = (
                    x_coords[j] + table_bbox.x_min,
                    x_coords[j + 1] + table_bbox.x_min,
                )
                cells.append(
                    Cell(
                        row=i,
                        col=j,
                        bbox=BBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                        colspan=1,
                        rowspan=1,
                    )
                )

        return cells

    def _merge_all_cells(
        self,
        line_mask_table: np.ndarray,
        x_coords: list[int],
        y_coords: list[int],
        table_bbox: BBox,
    ):
        col_cells = self._merge_cols_cells(
            line_mask_table, x_coords, y_coords, table_bbox
        )
        row_cells = self._merge_rows_cells(
            line_mask_table, x_coords, y_coords, table_bbox
        )

        col_cells_dict = {(cell.row, cell.col): cell for cell in col_cells}
        merged_cells = []
        for row_cell in row_cells:
            row_cell: Cell = row_cell
            key = (row_cell.row, row_cell.col)
            if key in col_cells_dict:
                col_cell = col_cells_dict[key]
                col_cell: Cell = col_cell
                # Итоговые координаты: X — из col_cell, Y — из row_cell
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

    def _merge_cols_cells(
        self,
        line_mask_table: np.ndarray,
        x_coords: list[int],
        y_coords: list[int],
        table_bbox: BBox,
    ):
        """Объединяет ячейки по колонкам"""
        merged_cells = []

        for i in range(len(y_coords) - 1):
            j = 0
            while j < len(x_coords) - 1:
                # Координаты текущей строки
                y1 = y_coords[i]
                y2 = y_coords[i + 1]

                x1 = x_coords[j]
                colspan = 1
                current_j = j

                while current_j < len(x_coords) - 2:  # до предпоследнего столбца
                    right_x = x_coords[current_j + 1]

                    box_y1 = max(int(y1 + self.padding), 0)
                    box_y2 = min(int(y2 - self.padding), line_mask_table.shape[0])
                    box_x1 = max(int(right_x - self.padding), 0)
                    box_x2 = min(int(right_x + self.padding), line_mask_table.shape[1])

                    # Вырезаем область
                    line_roi = line_mask_table[
                        int(box_y1) : int(box_y2),
                        int(box_x1) : int(box_x2),
                    ]
                    # if i == 0 and j == 1:
                    #     from PIL import Image

                    #     Image.fromarray(line_roi).show()
                    if line_roi.size == 0:
                        break
                    # Суммируем по строкам
                    row_sums = np.sum(line_roi, axis=1)

                    # Порог для определения линии
                    if np.any(row_sums > self.threshold_line):
                        break
                    else:
                        colspan += 1
                        current_j += 1

                x2 = x_coords[j + colspan]
                merged_cells.append(
                    Cell(
                        row=i,
                        col=j,
                        bbox=BBox(
                            x_min=x1 + table_bbox.x_min,
                            y_min=y1 + table_bbox.y_min,
                            x_max=x2 + table_bbox.x_min,
                            y_max=y2 + table_bbox.y_min,
                        ),
                        colspan=colspan,
                        rowspan=1,
                    )
                )

                j += colspan

        return merged_cells

    def _merge_rows_cells(
        self,
        line_mask_table: np.ndarray,
        x_coords: list[int],
        y_coords: list[int],
        table_bbox: BBox,
    ):
        """Объединяет ячейки по строкам"""
        merged_cells = []

        for j in range(len(x_coords) - 1):
            i = 0
            while i < len(y_coords) - 1:
                # Координаты текущей колонки
                x1 = x_coords[j]
                x2 = x_coords[j + 1]

                y1 = y_coords[i]
                rowspan = 1
                current_i = i

                while current_i < len(y_coords) - 2:  # до предпоследней строки
                    bottom_y = y_coords[current_i + 1]

                    box_y1 = max(int(bottom_y - self.padding), 0)
                    box_y2 = min(int(bottom_y + self.padding), line_mask_table.shape[0])
                    box_x1 = max(int(x1 + self.padding), 0)
                    box_x2 = min(int(x2 - self.padding), line_mask_table.shape[1])

                    # Вырезаем область
                    line_roi = line_mask_table[
                        int(box_y1) : int(box_y2),
                        int(box_x1) : int(box_x2),
                    ]

                    if line_roi.size == 0:
                        break
                    # Суммируем по столбцам
                    col_sums = np.sum(line_roi, axis=0)

                    # Порог для определения линии
                    if np.any(col_sums > self.threshold_line):
                        break
                    else:
                        rowspan += 1
                        current_i += 1

                y2 = y_coords[i + rowspan]
                merged_cells.append(
                    Cell(
                        row=i,
                        col=j,
                        bbox=BBox(
                            x_min=x1 + table_bbox.x_min,
                            y_min=y1 + table_bbox.y_min,
                            x_max=x2 + table_bbox.x_min,
                            y_max=y2 + table_bbox.y_min,
                        ),
                        colspan=1,
                        rowspan=rowspan,
                    )
                )

                i += rowspan

        return merged_cells

    def _get_grid(self, h_lines, v_lines, roi_shape=None):
        """Получает координаты сетки ячеек из линий"""
        y_coords = sorted(set(int(line[1]) for line in h_lines))
        x_coords = sorted(set(int(line[0]) for line in v_lines))

        # Если нет нижней горизонтальной линии, добавляем максимальный Y из вертикальных линий или низ ROI
        if roi_shape is not None:
            max_y = roi_shape[0] - 1
        else:
            max_y = max([int(line[3]) for line in v_lines], default=None)
        if max_y is not None and (not y_coords or max_y > y_coords[-1]):
            y_coords.append(max_y)

        # Аналогично для x_coords (если нет правой вертикальной линии)
        if roi_shape is not None:
            max_x = roi_shape[1] - 1
        else:
            max_x = max([int(line[2]) for line in h_lines], default=None)
        if max_x is not None and (not x_coords or max_x > x_coords[-1]):
            x_coords.append(max_x)

        x_coords = [
            x_coords[i]
            for i in range(len(x_coords))
            if i == 0 or (x_coords[i] - x_coords[i - 1]) >= self.min_cell
        ]

        y_coords = [
            y_coords[i]
            for i in range(len(y_coords))
            if i == 0 or (y_coords[i] - y_coords[i - 1]) >= self.min_cell
        ]

        return y_coords, x_coords

    def _group_lines(
        self,
        lines: list[tuple[float, float, float, float]],
        is_horizontal=True,
    ):
        """Группирует близкие линии в одну группу по среднему положению"""
        if not lines:
            return []

        if is_horizontal:
            lines.sort(key=lambda s: (s[1] + s[3]) / 2)
            positions = [(s[1] + s[3]) / 2 for s in lines]

        else:
            lines.sort(key=lambda s: (s[0] + s[2]) / 2)
            positions = [(s[0] + s[2]) / 2 for s in lines]

        groups = []
        current_group = [lines[0]]
        # logger.debug(f"Mean distance for grouping lines: {median_distance:.2f}")
        for i in range(1, len(lines)):
            if abs(positions[i] - positions[i - 1]) <= self.threshold_line:
                current_group.append(lines[i])
            else:
                groups.append(current_group)
                current_group = [lines[i]]
        groups.append(current_group)

        result: list[tuple[float, float, float, float]] = []
        for group in groups:
            if is_horizontal:
                avg_y = np.mean([(s[1] + s[3]) / 2 for s in group])
                min_x = min(min(s[0], s[2]) for s in group)
                max_x = max(max(s[0], s[2]) for s in group)
                result.append((min_x, avg_y, max_x, avg_y))
            else:
                avg_x = np.mean([(s[0] + s[2]) / 2 for s in group])
                min_y = min(min(s[1], s[3]) for s in group)
                max_y = max(max(s[1], s[3]) for s in group)
                result.append((avg_x, min_y, avg_x, max_y))

        return result

    def _find_lines_from_mask(self, mask_table: np.ndarray):
        """Находит горизонтальные и вертикальные линии на маске таблицы"""
        edges = cv2.Canny(mask_table, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=5
        )

        if lines is None:
            return [], []

        h_lines: list[tuple[float, float, float, float]] = []
        v_lines: list[tuple[float, float, float, float]] = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle)

            if angle < 45 or angle > 135:  # горизонтальная
                h_lines.append((x1, y1, x2, y2))
            else:  # вертикальная
                v_lines.append((x1, y1, x2, y2))

        return h_lines, v_lines
