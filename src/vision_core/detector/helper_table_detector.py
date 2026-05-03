from enum import Enum
from typing import TypeAlias
from uuid import uuid4

import numpy as np
import numpy.typing as npt

from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.table import Table
from vision_core.utils import geometry_utils, image_utils


class LineAxis(Enum):
    X = 0
    Y = 1


Int32Array: TypeAlias = npt.NDArray[np.int32]
SegmentsArray: TypeAlias = npt.NDArray[np.int32]


def axes_to_abs(box_lines: list[tuple[int, SegmentsArray]], offset: int) -> np.ndarray:
    if not box_lines:
        return np.empty(0, dtype=np.int32)

    axes = np.fromiter((axis for axis, _ in box_lines), dtype=np.int32, count=len(box_lines))
    return np.sort(axes + np.int32(offset))


def complete_grid_axes(
    axes_abs: np.ndarray,
    axis_min: int,
    axis_max: int,
    expected_step: int = 0,
) -> np.ndarray:
    axes = np.asarray(axes_abs, dtype=np.int32)
    if axes.size == 0:
        return np.array([axis_min, axis_max], dtype=np.int32)

    axes = np.unique(np.sort(axes))
    tol = max(8, expected_step // 2) if expected_step > 0 else 8

    if axes[0] > axis_min + tol:
        axes = np.r_[np.int32(axis_min), axes]

    if axes[-1] < axis_max - tol:
        axes = np.r_[axes, np.int32(axis_max)]

    if axes.size == 1:
        if abs(int(axes[0]) - axis_min) >= abs(axis_max - int(axes[0])):
            axes = np.r_[np.int32(axis_min), axes]
        else:
            axes = np.r_[axes, np.int32(axis_max)]

    return np.unique(np.sort(axes.astype(np.int32, copy=False)))


def extract_lines(
    lines: Int32Array | None,
    axis: LineAxis,
    axis_tol: int = 10,
    merge_gap: int = 5,
    min_len: int = 10,
) -> list[tuple[int, SegmentsArray]]:
    if lines is None:
        return []

    arr = np.asarray(lines, dtype=np.int32).reshape(-1, 4)
    if arr.size == 0:
        return []

    axis_values = find_line_axes(arr, axis=axis, axis_tol=axis_tol)

    out: list[tuple[int, SegmentsArray]] = []
    for axis_value in axis_values:
        segs = _collect_axis_segments(
            arr,
            axis_value=int(axis_value),
            axis=axis,
            axis_tol=axis_tol,
            merge_gap=merge_gap,
            min_len=min_len,
        )
        if segs.size > 0:
            out.append((int(axis_value), segs))
    return out


def _collect_axis_segments(
    lines: Int32Array,
    axis_value: int,
    axis: LineAxis,
    axis_tol: int = 10,
    merge_gap: int = 5,
    min_len: int = 10,
) -> SegmentsArray:
    arr = np.asarray(lines, dtype=np.int32).reshape(-1, 4)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    if axis is LineAxis.Y:
        # Горизонтальные линии: группируем по Y, сегменты по X.
        axis_ref = arr[:, 1]
        seg_start = np.minimum(arr[:, 0], arr[:, 2])
        seg_end = np.maximum(arr[:, 0], arr[:, 2])
    else:
        # Вертикальные линии: группируем по X, сегменты по Y.
        axis_ref = arr[:, 0]
        seg_start = np.minimum(arr[:, 1], arr[:, 3])
        seg_end = np.maximum(arr[:, 1], arr[:, 3])

    mask = np.abs(axis_ref - np.int32(axis_value)) <= np.int32(axis_tol)
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.int32)

    raw_segments: SegmentsArray = np.column_stack((seg_start[mask], seg_end[mask])).astype(np.int32, copy=False)
    return _normalize_segments(raw_segments, merge_gap=merge_gap, min_len=min_len)


def _normalize_segments(
    segs: SegmentsArray,
    merge_gap: int = 2,
    min_len: int = 8,
) -> SegmentsArray:
    segs = np.asarray(segs, dtype=np.int32).reshape(-1, 2)
    if segs.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    starts = np.minimum(segs[:, 0], segs[:, 1])
    ends = np.maximum(segs[:, 0], segs[:, 1])
    segs = np.column_stack((starts, ends))

    segs = segs[segs[:, 1] > segs[:, 0]]
    if segs.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    segs = np.unique(segs, axis=0)
    segs = segs[np.argsort(segs[:, 0], kind="stable")]

    merged = [segs[0].copy()]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s <= pe + merge_gap:
            merged[-1][1] = max(pe, e)
        else:
            merged.append(np.array([s, e], dtype=np.int32))

    out = np.vstack(merged)
    out = out[(out[:, 1] - out[:, 0]) >= min_len]
    return out.astype(np.int32, copy=False)


def find_line_axes(lines: np.ndarray, axis: LineAxis, axis_tol: int = 10) -> np.ndarray:
    if lines is None or len(lines) == 0:
        return np.array([], dtype=np.int32)
    arr = np.asarray(lines).reshape(-1, 4).astype(np.int32)

    axis_ref = arr[:, axis.value]

    if axis_ref.size == 0:
        return np.array([], dtype=np.int32)

    axes = _merge_close(axis_ref, gap=axis_tol)
    return axes.astype(np.int32)


def _merge_close(coords: np.ndarray, gap: int = 3) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int32)
    if coords.size == 0:
        return np.array([], dtype=np.int32)

    coords = np.sort(coords)

    # Границы новых групп там, где разница стала больше gap
    split_idx = np.flatnonzero(np.diff(coords) > gap) + 1

    # starts / ends групп (ends - exclusive)
    starts = np.r_[0, split_idx]
    ends = np.r_[split_idx, coords.size]

    # Индекс "середины" каждой группы:
    # start + (len(group)//2), полностью векторно
    mid_idx = starts + ((ends - starts) // 2)
    return coords[mid_idx]


def extract_raw_vertical_lines(
    roi_image: np.ndarray,
    median_height: int,
    min_line_length: int = 50,
    max_line_gap: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Извлекает вертикальные линии из бинаризованного изображения с помощью маски и преобразования Хафа.

    Args:
        roi_image: Входное бинаризованное изображение, на котором нужно найти линии.
        median_height: Медианная высота строк для определения размера
            структурного элемента при морфологической обработке.
        min_line_length: Минимальная длина линии в пикселях. Линии короче этого значения будут отфильтрованы.
        max_line_gap: Максимальный разрыв между сегментами линии, чтобы они считались одной линией.
    Returns:
        list[tuple[int, int, int, int]]: Список найденных вертикальных линий,
        представленных в виде кортежей (x1, y1, x2, y2),
        где (x1, y1) и (x2, y2) - координаты начала и конца линии.
    """

    vertical_lines = image_utils.compute_vertical_line_mask(roi_image, median_height=median_height)
    v_raw_lines = geometry_utils.find_hough_lines(
        vertical_lines,
        threshold=15,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
    )
    return v_raw_lines


def extract_raw_horizontal_lines(
    roi_image: np.ndarray,
    scale: int = 40,
    min_line_length: int = 50,
    max_line_gap: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Извлекает горизонтальные линии из бинаризованного изображения с помощью маски и преобразования Хафа.

    Args:
        roi_image: Входное бинаризованное изображение, на котором нужно найти линии.
        scale: Масштаб для определения размера структурного элемента при морфологической обработке.
        min_line_length: Минимальная длина линии в пикселях. Линии короче этого значения будут отфильтрованы.
        max_line_gap: Максимальный разрыв между сегментами линии, чтобы они считались одной линией.

    Returns:
        list[tuple[int, int, int, int]]: Список найденных горизонтальных линий,
        представленных в виде кортежей (x1, y1, x2, y2),
        где (x1, y1) и (x2, y2) - координаты начала и конца линии.
    """

    horizontal_lines = image_utils.compute_horizontal_line_mask(roi_image, scale=scale)
    h_raw_lines = geometry_utils.find_hough_lines(
        horizontal_lines,
        threshold=15,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
    )
    return h_raw_lines


def extract_raw_tables(
    table_mask: np.ndarray,
    border_tol: int = 8,
    scale_width: float = 0.35,
    scale_height: float = 0.08,
    min_density: float = 0.01,
    intersection_over_min_thr: float = 0.2,
) -> list[BBox]:
    """Извлекает bounding boxes таблиц из маски таблицы.

    Args:
        table_mask: Бинарная маска, где белые пиксели соответствуют линиям таблицы.
        border_tol: Толерантность к границам изображения в пикселях. Таблицы, касающиеся границ, будут отфильтрованы.
        scale_width: Минимальная ширина таблицы относительно ширины изображения для фильтрации
        scale_height: Минимальная высота таблицы относительно ширины изображения для фильтрации
        min_density: Минимальная плотность белых пикселей внутри bounding box для таблиц, касающихся границ
        intersection_over_min_thr: Порог для объединения перекрывающихся bounding boxes (IoM > threshold)
    Returns:
        list[BBox]: Список bounding boxes таблиц, представленных в виде объектов BBox.
    """
    contours = geometry_utils.find_contours(table_mask)
    candidates: list[tuple[int, int, int, int]] = []
    H, W = table_mask.shape[:2]
    min_width = int(W * scale_width)

    for contour in contours:
        x, y, w, h = geometry_utils.bounding_rect(contour)

        if w < min_width or h < 20:
            continue

        if w > W * 0.95 and h > H * 0.95:
            continue

        touches_border = x <= border_tol or y <= border_tol or x + w >= W - border_tol or y + h >= H - border_tol
        density = geometry_utils.count_non_zero_pixels(geometry_utils.roi_from_xywh(table_mask, (x, y, w, h)))

        if touches_border and density < min_density:
            continue

        candidates.append((x, y, w, h))

    bboxes: list[BBox] = []
    for x, y, w, h in candidates:
        box = BBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h)
        if any(
            kept.contains(box) or box.contains(kept) or kept.intersection_over_min(box) > intersection_over_min_thr
            for kept in bboxes
        ):
            continue
        bboxes.append(box)

    return bboxes


def _has_horizontal_separator(
    h_lines_box: list[tuple[int, SegmentsArray]],
    rel_y: int,
    x1: int,
    x2: int,
    tol: int = 8,
    min_cov: float = 0.5,
) -> bool:
    for y_c, segs in h_lines_box:
        if abs(y_c - rel_y) > tol:
            continue
        length = x2 - x1
        covered = sum(max(0, min(xe, x2) - max(xs, x1)) for xs, xe in segs)
        if length > 0 and covered / length >= min_cov:
            return True
    return False


def _has_vertical_separator(
    v_lines_box: list[tuple[int, SegmentsArray]],
    rel_x: int,
    y1: int,
    y2: int,
    tol: int = 8,
    min_cov: float = 0.5,
) -> bool:
    for x_c, segs in v_lines_box:
        if abs(x_c - rel_x) > tol:
            continue
        length = y2 - y1
        covered = sum(max(0, min(ye, y2) - max(ys, y1)) for ys, ye in segs)
        if length > 0 and covered / length >= min_cov:
            return True
    return False


def _build_separators(
    bbox: BBox,
    row_ys: np.ndarray,
    col_xs: np.ndarray,
    h_lines_box: list[tuple[int, SegmentsArray]],
    v_lines_box: list[tuple[int, SegmentsArray]],
    tol: int = 8,
    min_cov: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    x_min, y_min, _, _ = bbox.to_tuple()
    n_rows = len(row_ys) - 1
    n_cols = len(col_xs) - 1

    h_sep = np.zeros((max(n_rows - 1, 0), max(n_cols, 0)), dtype=bool)
    for i in range(n_rows - 1):
        rel_y = int(row_ys[i + 1] - y_min)
        for j in range(n_cols):
            h_sep[i, j] = _has_horizontal_separator(
                h_lines_box,
                rel_y,
                int(col_xs[j] - x_min),
                int(col_xs[j + 1] - x_min),
                tol=tol,
                min_cov=min_cov,
            )

    v_sep = np.zeros((max(n_rows, 0), max(n_cols - 1, 0)), dtype=bool)
    for j in range(n_cols - 1):
        rel_x = int(col_xs[j + 1] - x_min)
        for i in range(n_rows):
            v_sep[i, j] = _has_vertical_separator(
                v_lines_box,
                rel_x,
                int(row_ys[i] - y_min),
                int(row_ys[i + 1] - y_min),
                tol=tol,
                min_cov=min_cov,
            )

    return h_sep, v_sep


def _collect_union_pairs(
    n_rows: int,
    n_cols: int,
    h_sep: np.ndarray,
    v_sep: np.ndarray,
    merge_mode: str | None,
) -> np.ndarray:
    mode = (merge_mode or "none").lower()
    if mode == "none" or n_rows <= 0 or n_cols <= 0:
        return np.empty((0, 2), dtype=np.int32)

    pairs: list[np.ndarray] = []

    if mode in ("all", "cols") and n_cols > 1:
        r, c = np.where(~v_sep)
        left = r * n_cols + c
        right = left + 1
        if left.size:
            pairs.append(np.column_stack((left, right)).astype(np.int32, copy=False))

    if mode in ("all", "rows") and n_rows > 1:
        r, c = np.where(~h_sep)
        top = r * n_cols + c
        bottom = top + n_cols
        if top.size:
            pairs.append(np.column_stack((top, bottom)).astype(np.int32, copy=False))

    if not pairs:
        return np.empty((0, 2), dtype=np.int32)

    return np.vstack(pairs).astype(np.int32, copy=False)


def _dsu_build(size: int, pairs: np.ndarray) -> np.ndarray:
    parent = np.arange(size, dtype=np.int32)
    rank = np.zeros(size, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in pairs:
        union(int(a), int(b))

    for i in range(size):
        parent[i] = find(i)

    return parent


def build_table_from_grid(
    bbox: BBox,
    row_ys: np.ndarray,
    col_xs: np.ndarray,
    h_lines_box: list[tuple[int, SegmentsArray]],
    v_lines_box: list[tuple[int, SegmentsArray]],
    merge_mode: str | None = "all",
    table_id: str | None = None,
) -> Table:
    """Строит структуру таблицы по осям сетки и линиям-разделителям.

    Args:
        bbox: Габариты таблицы в абсолютных координатах.
        row_ys: Абсолютные Y-координаты осей строк.
        col_xs: Абсолютные X-координаты осей столбцов.
        h_lines_box: Горизонтальные линии в ROI-координатах таблицы.
        v_lines_box: Вертикальные линии в ROI-координатах таблицы.
        merge_mode: Режим объединения ячеек: all, rows, cols, none.
        table_id: Идентификатор таблицы. Если не задан, генерируется UUID.

    Returns:
        Table: Таблица с атомарными или объединенными ячейками.
    """
    x_min, y_min, x_max, y_max = bbox.to_tuple()
    n_rows = len(row_ys) - 1
    n_cols = len(col_xs) - 1
    total = n_rows * n_cols

    if total <= 0:
        return Table(
            id=table_id or str(uuid4()),
            bbox=BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            num_rows=max(n_rows, 0),
            num_cols=max(n_cols, 0),
            cells=[],
        )

    h_sep, v_sep = _build_separators(bbox, row_ys, col_xs, h_lines_box, v_lines_box)
    pairs = _collect_union_pairs(n_rows, n_cols, h_sep, v_sep, merge_mode)
    parent = _dsu_build(total, pairs)

    groups: dict[int, list[tuple[int, int]]] = {}
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            groups.setdefault(int(parent[idx]), []).append((i, j))

    cells: list[Cell] = []
    for indices in groups.values():
        rows = [i for i, _ in indices]
        cols = [j for _, j in indices]
        r0, r1 = min(rows), max(rows) + 1
        c0, c1 = min(cols), max(cols) + 1

        cells.append(
            Cell(
                row=r0,
                col=c0,
                rowspan=r1 - r0,
                colspan=c1 - c0,
                bbox=BBox(
                    x_min=int(col_xs[c0]),
                    y_min=int(row_ys[r0]),
                    x_max=int(col_xs[c1]),
                    y_max=int(row_ys[r1]),
                ),
                value="",
            )
        )

    cells.sort(key=lambda cell: (cell.row, cell.col))
    return Table(
        id=table_id or str(uuid4()),
        bbox=BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        num_rows=n_rows,
        num_cols=n_cols,
        cells=cells,
    )
