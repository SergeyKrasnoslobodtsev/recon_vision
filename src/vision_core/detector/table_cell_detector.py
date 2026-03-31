"""Детектор ячеек таблицы на основе анализа сетки линий."""

from __future__ import annotations

import cv2
import numpy as np

from vision_core.config import TableCellDetectorConfig
from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell

_MERGE_COLS = "cols"
_MERGE_ROWS = "rows"
_MERGE_ALL = "all"


class TableCellDetector:
    """Детектор ячеек таблицы.

    Разделяет маску на горизонтальные и вертикальные линии, строит сетку
    координат и извлекает ячейки методом Union-Find: для каждой пары
    соседних ячеек проверяет наличие разделителя в маске — если его нет,
    ячейки объединяются.
    """

    def __init__(self, cfg: TableCellDetectorConfig | None = None) -> None:
        if cfg is None:
            cfg = TableCellDetectorConfig()
        self.min_cell = cfg.min_cell
        self.padding = cfg.padding
        self.threshold_line = cfg.threshold_line
        self.coverage_thr = cfg.coverage_thr

    def extract_cells(
        self,
        table_mask: np.ndarray,
        table_bbox: tuple[int, int, int, int],
        merge_mode: str | None = None,
    ) -> list[Cell]:
        """Извлекает ячейки из маски таблицы.

        Args:
            table_mask: Бинарная маска таблицы (линии белые, фон чёрный).
            table_bbox: Координаты таблицы на исходном изображении (x_min, y_min, x_max, y_max).
            merge_mode: Режим объединения ячеек: "cols", "rows", "all" или None.

        Returns:
            Список ячеек, отсортированных по (row, col).
        """
        x, y = table_bbox[0], table_bbox[1]
        pure_h, pure_v = self._split_lines(table_mask)
        return self._build_cells(x, y, pure_h, pure_v, merge_mode)

    # ------------------------------------------------------------------
    # Разделение маски на горизонтальные и вертикальные линии
    # ------------------------------------------------------------------

    def _split_lines(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Разделяет маску таблицы на горизонтальные и вертикальные линии.

        Args:
            mask: Бинарная маска таблицы.

        Returns:
            Кортеж (pure_h, pure_v) — маски горизонтальных и вертикальных линий.
        """
        h, w = mask.shape[:2]
        k_h = max(10, w // 20)
        k_v = max(10, h // 20)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_v))

        pure_h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)
        pure_v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel)

        return pure_h, pure_v

    # ------------------------------------------------------------------
    # Нахождение позиций линий
    # ------------------------------------------------------------------

    def _find_line_positions(self, mask: np.ndarray, axis: int) -> list[int]:
        """Находит координаты центров линий по бинарной проекции маски.

        Args:
            mask: Бинарная маска линий.
            axis: 0 — ищем позиции по X (вертикальные линии),
                  1 — ищем позиции по Y (горизонтальные линии).

        Returns:
            Отсортированный список координат центров линий.
        """
        # np.any — линия считается присутствующей если хотя бы один пиксель ненулевой
        proj = np.any(mask > 0, axis=axis).astype(np.uint8)

        diff = np.diff(proj, prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        centers = list(((starts + ends - 1) // 2).astype(int))
        return self._filter_close_positions(centers)

    def _filter_close_positions(self, positions: list[int]) -> list[int]:
        """Убирает позиции ближе чем min_cell друг к другу (артефакт толстой линии)."""
        result: list[int] = []
        for pos in positions:
            if not result or pos - result[-1] >= self.min_cell:
                result.append(pos)
        return result

    # ------------------------------------------------------------------
    # Построение ячеек: Union-Find
    # ------------------------------------------------------------------

    def _build_cells(
        self,
        x: int,
        y: int,
        pure_h: np.ndarray,
        pure_v: np.ndarray,
        merge_mode: str | None,
    ) -> list[Cell]:
        """Строит ячейки таблицы методом Union-Find.

        Args:
            x: X-смещение таблицы на исходном изображении.
            y: Y-смещение таблицы на исходном изображении.
            pure_h: Маска горизонтальных линий.
            pure_v: Маска вертикальных линий.
            merge_mode: Режим объединения: "cols", "rows", "all" или None.

        Returns:
            Список ячеек, отсортированных по (row, col).
        """
        xs = self._find_line_positions(pure_v, axis=0)
        ys = self._find_line_positions(pure_h, axis=1)

        if len(xs) < 2:
            return []

        ys = self._add_missing_bottom_border(ys, pure_h.shape[0])

        if len(ys) < 2:
            return []

        n_rows = len(ys) - 1
        n_cols = len(xs) - 1

        parent = list(range(n_rows * n_cols))

        def cell_id(row: int, col: int) -> int:
            return row * n_cols + col

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        if merge_mode in (_MERGE_COLS, _MERGE_ALL):
            for r in range(n_rows):
                for c in range(n_cols - 1):
                    if not self._has_vertical_separator(pure_v, xs, ys, r, c):
                        union(cell_id(r, c), cell_id(r, c + 1))

        if merge_mode in (_MERGE_ROWS, _MERGE_ALL):
            for r in range(n_rows - 1):
                for c in range(n_cols):
                    if not self._has_horizontal_separator(pure_h, xs, ys, r, c):
                        union(cell_id(r, c), cell_id(r + 1, c))

        groups: dict[int, list[tuple[int, int]]] = {}
        for r in range(n_rows):
            for c in range(n_cols):
                groups.setdefault(find(cell_id(r, c)), []).append((r, c))

        cells: list[Cell] = []
        for members in groups.values():
            cells.extend(self._group_to_cells(members, xs, ys, x, y))

        cells.sort(key=lambda cell: (cell.row, cell.col))
        return cells

    def _add_missing_bottom_border(self, ys: list[int], roi_height: int) -> list[int]:
        """Добавляет нижнюю границу ROI если последняя строка не имеет нижней линии.

        Обрабатывает случай:
            |--------|------|-------|
            |        |      |       |   <- нет нижней линии

        Args:
            ys: Найденные Y-позиции горизонтальных линий.
            roi_height: Высота ROI.

        Returns:
            Обновлённый список Y-позиций.
        """
        if ys and ys[-1] < roi_height - self.min_cell:
            return ys + [roi_height]
        return ys

    def _group_to_cells(
        self,
        members: list[tuple[int, int]],
        xs: list[int],
        ys: list[int],
        offset_x: int,
        offset_y: int,
    ) -> list[Cell]:
        """Преобразует группу атомарных ячеек в одну или несколько Cell.

        Если группа прямоугольная — возвращает одну объединённую ячейку.
        Иначе — каждую атомарную ячейку отдельно (не прямоугольный colspan не поддерживается).

        Args:
            members: Список (row, col) атомарных ячеек в группе.
            xs: X-координаты сетки.
            ys: Y-координаты сетки.
            offset_x: X-смещение таблицы.
            offset_y: Y-смещение таблицы.

        Returns:
            Список Cell.
        """
        rows = sorted({r for r, _ in members})
        cols = sorted({c for _, c in members})
        is_rectangular = len(members) == len(rows) * len(cols)

        if not is_rectangular:
            return [
                Cell(
                    bbox=BBox(
                        x_min=xs[c] + offset_x,
                        y_min=ys[r] + offset_y,
                        x_max=xs[c + 1] + offset_x,
                        y_max=ys[r + 1] + offset_y,
                    ),
                    row=r,
                    col=c,
                    colspan=1,
                    rowspan=1,
                    text="",
                )
                for r, c in members
            ]

        min_r, max_r = rows[0], rows[-1]
        min_c, max_c = cols[0], cols[-1]

        return [
            Cell(
                bbox=BBox(
                    x_min=xs[min_c] + offset_x,
                    y_min=ys[min_r] + offset_y,
                    x_max=xs[max_c + 1] + offset_x,
                    y_max=ys[max_r + 1] + offset_y,
                ),
                row=min_r,
                col=min_c,
                colspan=len(cols),
                rowspan=len(rows),
                text="",
            )
        ]

    # ------------------------------------------------------------------
    # Проверка наличия разделителей между ячейками
    # ------------------------------------------------------------------

    def _has_vertical_separator(
        self,
        pure_v: np.ndarray,
        xs: list[int],
        ys: list[int],
        row: int,
        col: int,
    ) -> bool:
        """Проверяет наличие вертикальной линии между col и col+1 в строке row."""
        x_pos = xs[col + 1]
        y0 = ys[row] + self.padding
        y1 = ys[row + 1] - self.padding
        x0 = max(x_pos - self.threshold_line, 0)
        x1 = min(x_pos + self.threshold_line + 1, pure_v.shape[1])

        if y1 <= y0 or x1 <= x0:
            return True

        region = pure_v[y0:y1, x0:x1]
        min_length = max(1, int((y1 - y0) * 0.1))
        return self._has_continuous_line(region, min_length, axis=0, min_density=self.coverage_thr)

    def _has_horizontal_separator(
        self,
        pure_h: np.ndarray,
        xs: list[int],
        ys: list[int],
        row: int,
        col: int,
    ) -> bool:
        """Проверяет наличие горизонтальной линии между row и row+1 в столбце col."""
        y_pos = ys[row + 1]
        x0 = xs[col] + self.padding
        x1 = xs[col + 1] - self.padding
        y0 = max(y_pos - self.threshold_line, 0)
        y1 = min(y_pos + self.threshold_line + 1, pure_h.shape[0])

        if x1 <= x0 or y1 <= y0:
            return True

        region = pure_h[y0:y1, x0:x1]
        min_length = max(1, int((x1 - x0) * 0.1))
        return self._has_continuous_line(region, min_length, axis=1, min_density=self.coverage_thr)

    def _has_continuous_line(
        self,
        region: np.ndarray,
        min_length: int,
        axis: int,
        min_density: float = 0.6,
    ) -> bool:
        """Проверяет наличие непрерывной линии с достаточной плотностью.

        Args:
            region: Область для проверки.
            min_length: Минимальная длина непрерывного участка.
            axis: 0 — вертикальная линия, 1 — горизонтальная.
            min_density: Минимальная доля белых пикселей (0-1).

        Returns:
            True если найдена непрерывная линия с достаточной плотностью.
        """
        if region.size == 0:
            return False

        projection = np.max(region, axis=1) if axis == 0 else np.max(region, axis=0)
        binary = (projection > 127).astype(np.uint8)

        max_run = current_run = total = 0
        for px in binary:
            if px:
                current_run += 1
                total += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        if max_run < min_length:
            return False

        return total / len(binary) >= min_density
