import numpy as np
from scipy.signal import find_peaks

from vision_core.config import TableDetectorConfig
from vision_core.debug_image_observer import DebugImageObserver
from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.table import Table
from vision_core.ocr.base import OcrEngine, OcrResult
from vision_core.utils import geometry_utils, image_utils


class TableDetector:
    """Детектор таблиц на изображении"""

    def __init__(
        self,
        table_detector_config: TableDetectorConfig | None = None,
        ocr_engine: OcrEngine = None,
        debug_image: DebugImageObserver | None = None,
    ):
        """
        Инициализация детектора таблиц.

        Args:
            preprocessor_config: Конфигурация для предобработки таблиц.
                Если None, используется конфигурация по умолчанию.
            table_detector_config: Конфигурация для детектора таблиц.
                Если None, используется конфигурация по умолчанию.
            debug_image: Наблюдатель для отладки изображений. Если None, отладка отключена.
        """
        self.cfg = table_detector_config or TableDetectorConfig()
        self._ocr = ocr_engine
        self._debug_image = debug_image

        self._sharpness_img = None
        self._binary_img = None
        self._mask_raw_lines = None
        self._mask_clean_lines = None
        self._binary_img_clean_lines = None
        self._mask_fixed = None
        self.ocr_bboxes = None

    def detect_tables(
        self,
        image: np.ndarray,
        ocr_results: list[OcrResult],
        page_number: int = 0,
    ) -> list[Table]:
        self._preprocess(image, page_number)

        self.ocr_bboxes = [BBox.from_tuple(r.bbox) for r in ocr_results]

        candidate_tables = self._find_table_candidates(self.ocr_bboxes)

        raw_tables: list[Table] = []
        median_heights: list[float] = []

        for i, tbl in enumerate(candidate_tables):
            bboxes = overlapping_bboxes(tbl, self.ocr_bboxes, min_overlap_ratio=0.6)
            heights = overlapping_heights(bboxes)

            if not heights:
                continue

            median_height = np.median(heights)

            median_heights.append(float(median_height))

            mask, content_mask = self._clean_mask_table(tbl, bboxes, median_height)

            row_ys, col_xs = self._build_grid(mask, median_height)

            is_vborder = detect_vertical_borders(mask, row_ys, col_xs)
            is_hborder = detect_horizontal_borders(mask, row_ys, col_xs)

            n_rows = len(row_ys) - 1
            n_cols = len(col_xs) - 1

            labels = build_merged_regions(is_vborder, is_hborder, n_rows, n_cols, mode=self.cfg.mode_merge_cells)

            table = build_table_entity(tbl, labels, row_ys, col_xs, f"{i}")

            cell_ocr = assign_blobs_to_cells(table.cells, ocr_results, min_overlap_ratio=0.1, confidence_threshold=0.4)
            reocr_cells_list = find_reocr_cells(table.cells, cell_ocr, min_containment_ratio=0.7, min_overlap_ratio=0.1)

            missing_cells = find_missing_text_cells(table.cells, content_mask, int(tbl.x_min), int(tbl.y_min))

            cells_to_reocr = list({id(c): c for c in reocr_cells_list + missing_cells}.values())

            fill_cell_values(table.cells, cell_ocr)
            reocr_cells(cells_to_reocr, image, self._ocr)

            raw_tables.append(table)

        return self._postprocess(raw_tables, median_heights)

    def _postprocess(self, tables: list[Table], median_heights: list[float]) -> list[Table]:
        valid_idx = [i for i, t in enumerate(tables) if t.is_valid()]
        tables = [tables[i] for i in valid_idx]
        median_heights = [median_heights[i] for i in valid_idx]

        merge_pairs = find_merge_candidates(tables, median_heights, x_overlap_thr=0.9, max_gap_in_medians=2.5)
        return merge_table_groups(tables, merge_pairs)

    def _build_grid(self, mask: np.ndarray, median_height: float) -> tuple[np.ndarray, np.ndarray]:
        row_pix = sum_pixel_axis(mask, axis=1)
        col_pix = sum_pixel_axis(mask, axis=0)

        row_peaks = find_lines_with_edges(row_pix, distance=20, prominence_frac=0.3)
        col_peaks = find_lines_with_edges(col_pix, distance=20, prominence_frac=0.3)

        row_peaks = merge_thin_lines(row_peaks, median_height * 0.8)
        col_peaks = merge_thin_lines(col_peaks, median_height * 0.8)

        row_ys = np.asarray(row_peaks, dtype=np.int32)
        col_xs = np.asarray(col_peaks, dtype=np.int32)

        return row_ys, col_xs

    def _clean_mask_table(
        self,
        candidate_tbl: BBox,
        bboxes: list[BBox],
        median_height: float,
    ) -> tuple[np.ndarray, np.ndarray]:

        text_mask = text_mask_inset(candidate_tbl, bboxes, pix=3)
        no_text_mask = image_utils.subtract_mask(candidate_tbl.roi(self._binary_img_clean_lines), text_mask)
        no_text_mask = image_utils.get_mask(no_text_mask, candidate_tbl.roi(self._mask_fixed))

        scale_factor = int(candidate_tbl.height / median_height)

        v_mask = image_utils.morph_open(no_text_mask, (1, 7))
        h_mask = image_utils.morph_open(no_text_mask, (int(candidate_tbl.width // 10), 1))

        mask_finaly = keep_lines_with_intersections(v_mask, h_mask, 1, median_height)

        content_mask = image_utils.subtract_mask(no_text_mask, mask_finaly)

        return (mask_finaly, content_mask)

    def _find_table_candidates(self, bboxes: list[BBox]) -> list[BBox]:
        cnts = geometry_utils.find_top_level_contours(self._mask_fixed)
        table_candidates = []
        for c in cnts:
            x, y, w, h = geometry_utils.bounding_rect(c)
            bbox = BBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h)

            if is_full_page_frame(bbox, self._w, self._h):
                continue
            if not is_valid_table_size(bbox, self._w, self._h, self.cfg.min_area_frac, self.cfg.min_w, self.cfg.min_h):
                continue
            if not has_text_inside(bbox, bboxes, self.cfg.min_overlap_ratio):
                continue

            table_candidates.append(bbox)

        return sorted(table_candidates, key=lambda b: (b.y_min, b.x_min))

    def _preprocess(self, image: np.ndarray, page_number: int = 0):
        self._h, self._w = image.shape[:2]
        self._sharpness_img = image_utils.apply_gamma_correction(image, gamma=0.3)
        gray_img = image_utils.to_grayscale(self._sharpness_img)
        self._binary_img = image_utils.binary_masked(gray_img, k_gauss=5, block_size=11, c=2)
        self._mask_raw_lines = image_utils.extract_lines_mask(self._binary_img, v_scale=10, h_scale=40)
        lines = geometry_utils.find_hough_lines(
            self._mask_raw_lines,
            threshold=10,
            min_line_length=int(self._h * 0.1),
            max_line_gap=30,
        )
        diag_lines = extract_inclined_lines(lines, angle_thresh=1.0)
        self._mask_clean_lines = image_utils.erase_lines(self._mask_raw_lines, diag_lines, thickness=3)
        # эта маска нужна для удаления текста чтобы получить явные линии внутри таблиц
        self._binary_img_clean_lines = image_utils.erase_lines(self._binary_img, diag_lines, thickness=3)
        # для поиска кандидатов таблиц заполним разрывы
        self._mask_fixed = image_utils.repair_mask(self._mask_clean_lines, close_size=7)


def find_merge_candidates(
    tables: list[Table],
    median_heights: list[float],
    x_overlap_thr: float = 0.9,
    max_gap_in_medians: float = 2.0,
) -> list[tuple[int, int]]:
    """Находит пары соседних по вертикали таблиц с совпадающим числом столбцов,
    высоким перекрытием по x и разрывом не больше max_gap_in_medians медиан высоты строки."""
    merge_pairs = []
    for i in range(len(tables) - 1):
        a, b = tables[i], tables[i + 1]

        if _x_overlap_ratio(a.bbox, b.bbox) < x_overlap_thr:
            continue
        if a.num_cols != b.num_cols:
            continue

        gap = b.bbox.y_min - a.bbox.y_max
        if gap > max_gap_in_medians * max(median_heights[i], median_heights[i + 1]):
            continue

        merge_pairs.append((i, i + 1))
    return merge_pairs


def _x_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    return overlap / min(a.width, b.width)


def merge_table_groups(tables: list[Table], merge_pairs: list[tuple[int, int]]) -> list[Table]:
    """Объединяет таблицы в группы через union-find, сдвигая row у ячеек
    нижних таблиц и склеивая их в одну Table."""
    n = len(tables)
    parent = list(range(n))
    for a, b in merge_pairs:
        _union(parent, a, b)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(_find_root(parent, idx), []).append(idx)

    merged = []
    for indices in groups.values():
        indices.sort(key=lambda idx: tables[idx].bbox.y_min)
        group = [tables[idx] for idx in indices]

        cells = []
        row_offset = 0
        for t in group:
            cells.extend(cell.model_copy(update={"row": cell.row + row_offset}) for cell in t.cells)
            row_offset += t.num_rows

        merged.append(
            Table(
                id=group[0].id,
                bbox=BBox(
                    x_min=min(t.bbox.x_min for t in group),
                    y_min=min(t.bbox.y_min for t in group),
                    x_max=max(t.bbox.x_max for t in group),
                    y_max=max(t.bbox.y_max for t in group),
                ),
                num_rows=sum(t.num_rows for t in group),
                num_cols=group[0].num_cols,
                cells=cells,
            )
        )
    return merged


def fill_cell_values(cells: list[Cell], cell_ocr: dict[int, list[OcrResult]]) -> None:
    """Заполняет value и blobs каждой ячейки из назначенных ей OCR-результатов."""
    for i, cell in enumerate(cells):
        items = cell_ocr[i]
        cell.value = "\n".join(r.text for r in items)
        cell.blobs = [BBox.from_tuple(r.bbox) for r in items]


def reocr_cells(cells: list[Cell], image: np.ndarray, ocr_engine: OcrEngine, confidence_threshold: float = 0.7) -> None:
    """Перезапускает OCR по кропу для переданных ячеек и перезаписывает их value."""
    for cell in cells:
        crop = cell.bbox.roi(image)
        if crop.size == 0:
            cell.value = ""
            cell.blobs = []
            continue

        results = ocr_engine.predict(crop)
        cell_results = results[0] if results else []
        texts = [r.text for r in cell_results if r.confidence >= confidence_threshold]

        cell.value = " ".join(texts)
        cell.blobs = []


def find_missing_text_cells(
    cells: list[Cell],
    content_mask: np.ndarray,
    origin_x: int,
    origin_y: int,
    min_area: int = 50,
    min_side: int = 7,
) -> list[Cell]:
    """Ячейки, где после вычитания OCR-текста и линий сетки остались
    нераспознанные пиксели — OCR пропустил текст в этой ячейке."""
    missing = []
    for cell in cells:
        cx0, cy0, cx1, cy1 = cell.bbox.to_tuple()
        crop = content_mask[cy0 - origin_y : cy1 - origin_y, cx0 - origin_x : cx1 - origin_x]
        if image_utils.has_missing_text(crop, min_area=min_area, min_side=min_side):
            missing.append(cell)
    return missing


def has_overflowing_text(cell_bbox: BBox, ocr_blobs: list[BBox], min_containment_ratio: float = 0.95) -> bool:
    """True, если среди блобов, назначенных ячейке, есть такой, что не помещается в неё целиком."""
    return any(blob.area > 0 and cell_bbox.intersect(blob) / blob.area < min_containment_ratio for blob in ocr_blobs)


def find_reocr_cells(
    cells: list[Cell],
    cell_ocr: dict[int, list[OcrResult]],
    min_containment_ratio: float = 0.8,
    min_overlap_ratio: float = 0.1,
) -> list[Cell]:
    """Ячейки, где OCR слепил текст нескольких ячеек, + соседи, которых заметно
    (не на пару случайных пикселей, а по-настоящему) задевает тот же блок."""
    reocr_idx = set()

    for i, cell in enumerate(cells):
        blobs = [BBox.from_tuple(r.bbox) for r in cell_ocr[i]]
        if not has_overflowing_text(cell.bbox, blobs, min_containment_ratio):
            continue
        reocr_idx.add(i)

        for blob in blobs:
            if blob.area == 0 or cell.bbox.intersect(blob) / blob.area >= min_containment_ratio:
                continue  # этот конкретный блок не переполняет — не он виноват
            for j, other in enumerate(cells):
                if j != i and other.bbox.intersect(blob) / blob.area >= min_overlap_ratio:
                    reocr_idx.add(j)

    return [cell for i, cell in enumerate(cells) if i in reocr_idx]


def assign_blobs_to_cells(
    cells: list[Cell],
    ocr_results: list[OcrResult],
    min_overlap_ratio: float = 0.1,
    confidence_threshold: float = 0.7,
) -> dict[int, list[OcrResult]]:
    """Распределяет OCR-результаты по ячейкам — каждый блок достаётся ячейке
    с максимальной долей перекрытия (BBox.intersect / ocr_area)."""
    cell_ocr: dict[int, list[OcrResult]] = {i: [] for i in range(len(cells))}

    for r in ocr_results:
        if r.confidence < confidence_threshold:
            continue

        ocr_bbox = BBox.from_tuple(r.bbox)
        if ocr_bbox.area == 0:
            continue

        best_idx, best_ratio = None, min_overlap_ratio
        for i, cell in enumerate(cells):
            ratio = ocr_bbox.intersect(cell.bbox) / ocr_bbox.area
            if ratio > best_ratio:
                best_idx, best_ratio = i, ratio

        if best_idx is not None:
            cell_ocr[best_idx].append(r)

    return cell_ocr


def build_table_entity(
    table_bbox: BBox,
    labels: np.ndarray,
    row_ys: np.ndarray,
    col_xs: np.ndarray,
    table_id: str,
) -> Table:
    """Строит Table/Cell из grid-разметки (row_ys/col_xs/labels). Координаты ячеек — абсолютные (страница)."""
    ox, oy = table_bbox.x_min, table_bbox.y_min
    n_rows, n_cols = labels.shape

    cells = []
    for label in np.unique(labels):
        if label < 0:
            continue

        rows, cols = np.where(labels == label)
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())

        cell_bbox = BBox(
            x_min=ox + col_xs[col_min],
            y_min=oy + row_ys[row_min],
            x_max=ox + col_xs[col_max + 1],
            y_max=oy + row_ys[row_max + 1],
        )

        cells.append(
            Cell(
                row=row_min,
                col=col_min,
                rowspan=row_max - row_min + 1,
                colspan=col_max - col_min + 1,
                bbox=cell_bbox,
            )
        )

    return Table(
        id=table_id,
        bbox=table_bbox,
        num_rows=n_rows,
        num_cols=n_cols,
        cells=cells,
    )


def build_merged_regions(
    is_vborder: np.ndarray,
    is_hborder: np.ndarray,
    n_rows: int,
    n_cols: int,
    mode: str | None = "cols",
) -> np.ndarray:
    """Строит labels-сетку по режиму объединения ячеек: cols/rows/all/none."""
    mode = (mode or "none").lower()
    merge_cols = mode in ("cols", "all")
    merge_rows = mode in ("rows", "all")

    segments = _row_segments(is_vborder, n_rows, n_cols) if merge_cols else _unit_segments(n_rows, n_cols)

    parent = list(range(len(segments)))
    if merge_rows:
        for a, b in _segment_pairs_to_merge(segments, is_hborder, n_rows):
            _union(parent, a, b)

    labels = np.full((n_rows, n_cols), -1, dtype=int)
    for idx, (row, col_start, col_end) in enumerate(segments):
        labels[row, col_start : col_end + 1] = _find_root(parent, idx)

    return labels


def _segment_pairs_to_merge(
    segments: list[tuple[int, int, int]],
    is_hborder: np.ndarray,
    n_rows: int,
) -> list[tuple[int, int]]:
    """Пары сегментов друг под другом: совпадают по столбцам и между ними нет hborder."""
    by_row: dict[int, list[int]] = {}
    for idx, (row, _, _) in enumerate(segments):
        by_row.setdefault(row, []).append(idx)

    pairs = []
    for idx, (row, col_start, col_end) in enumerate(segments):
        if row + 1 >= n_rows:
            continue
        for other_idx in by_row.get(row + 1, []):
            _, other_start, other_end = segments[other_idx]
            if other_start != col_start or other_end != col_end:
                continue
            if any(is_hborder[row + 1, c] for c in range(col_start, col_end + 1)):
                continue
            pairs.append((idx, other_idx))
    return pairs


def _row_segments(is_vborder: np.ndarray, n_rows: int, n_cols: int) -> list[tuple[int, int, int]]:
    """Разбивает каждую строку на горизонтальные сегменты по вертикальным границам."""
    segments = []
    for row in range(n_rows):
        col = 0
        while col < n_cols:
            col_end = col
            while col_end + 1 < n_cols and not is_vborder[row, col_end + 1]:
                col_end += 1
            segments.append((row, col, col_end))
            col = col_end + 1
    return segments


def _unit_segments(n_rows: int, n_cols: int) -> list[tuple[int, int, int]]:
    """Каждая позиция сетки — отдельный сегмент (без горизонтального объединения)."""
    return [(row, col, col) for row in range(n_rows) for col in range(n_cols)]


def _union(parent: list[int], a: int, b: int) -> None:
    ra, rb = _find_root(parent, a), _find_root(parent, b)
    if ra != rb:
        parent[ra] = rb


def _find_root(parent: list[int], idx: int) -> int:
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]
    return idx


def detect_vertical_borders(
    mask: np.ndarray,
    row_ys: np.ndarray,
    col_xs: np.ndarray,
    tolerance: int = 20,
    prominence_frac: float = 0.2,
) -> np.ndarray:
    return _detect_borders(mask, row_ys, col_xs, tolerance, prominence_frac, is_vertical=True)


def detect_horizontal_borders(
    mask: np.ndarray,
    row_ys: np.ndarray,
    col_xs: np.ndarray,
    tolerance: int = 20,
    prominence_frac: float = 0.2,
) -> np.ndarray:
    return _detect_borders(mask, col_xs, row_ys, tolerance, prominence_frac, is_vertical=False).T


def _detect_borders(
    mask: np.ndarray,
    band_edges: np.ndarray,
    check_positions: np.ndarray,
    tolerance: int,
    prominence_frac: float,
    is_vertical: bool,
) -> np.ndarray:
    n_bands = len(band_edges) - 1
    result = np.zeros((n_bands, len(check_positions)), dtype=bool)

    for band_idx in range(n_bands):
        start, end = band_edges[band_idx], band_edges[band_idx + 1]
        strip = mask[start:end, :].sum(axis=0) if is_vertical else mask[:, start:end].sum(axis=1)
        if strip.max() == 0:
            continue

        threshold = strip.max() * prominence_frac
        for pos_idx, pos in enumerate(check_positions):
            lo, hi = max(0, pos - tolerance), min(len(strip), pos + tolerance + 1)
            result[band_idx, pos_idx] = strip[lo:hi].max() >= threshold

    return result


def merge_thin_lines(peaks: list[int], min_gap: float) -> list[int]:
    """Убирает пики, если промежуток до соседнего меньше min_gap —
    сливает его с соседним. Первую и последнюю границу не трогает."""
    if len(peaks) < 3:
        return peaks

    merged = list(peaks)

    i = 1
    while i < len(merged) - 1:
        if merged[i] - merged[i - 1] < min_gap:
            del merged[i]
        else:
            i += 1

    while len(merged) > 2 and merged[-1] - merged[-2] < min_gap:
        del merged[-2]

    return merged


def _edge_peak(sums: np.ndarray, edge_check_width: int, threshold: float, at_start: bool) -> int:
    window = sums[:edge_check_width] if at_start else sums[-edge_check_width:]
    if window.max() <= threshold:
        return 0 if at_start else len(sums) - 1
    offset = 0 if at_start else len(sums) - edge_check_width
    return offset + int(np.argmax(window))


def find_lines_with_edges(sums, distance, prominence_frac, edge_check_width=5, edge_frac=0.5):
    candidate_peaks, _ = find_peaks(sums, distance=distance)
    baseline = np.median(sums[candidate_peaks]) if candidate_peaks.size > 0 else sums.max()
    peaks, _ = find_peaks(sums, distance=distance, prominence=baseline * prominence_frac)
    peaks = list(peaks)
    threshold = sums.max() * edge_frac

    if not peaks or peaks[0] > edge_check_width:
        peaks.insert(0, _edge_peak(sums, edge_check_width, threshold, at_start=True))

    if peaks[-1] < len(sums) - edge_check_width:
        peaks.append(_edge_peak(sums, edge_check_width, threshold, at_start=False))

    return sorted(peaks)


def sum_pixel_axis(mask: np.ndarray, axis: int = 0):
    return (mask > 0).sum(axis=axis)


def keep_lines_with_intersections(v_mask, h_mask, dilate_size=1, min_length=15):
    h_dilated = image_utils.dilate_image(h_mask, kernel_size=(dilate_size, dilate_size))
    v_dilated = image_utils.dilate_image(v_mask, kernel_size=(dilate_size, dilate_size))

    v_keep = image_utils.filter_long_intersecting(v_mask, h_dilated, min_length, is_vertical=True)
    h_keep = image_utils.filter_long_intersecting(h_mask, v_dilated, min_length, is_vertical=False)

    return image_utils.get_mask(v_keep, h_keep)


def text_mask_inset(table_bbox: BBox, bboxes: list[BBox], pix: int = 5) -> np.ndarray:
    h, w = int(table_bbox.height), int(table_bbox.width)
    mask = np.zeros((h, w), dtype=np.uint8)
    ox, oy = table_bbox.x_min, table_bbox.y_min

    for b in bboxes:
        margin_b = b.margin(pix)
        x0 = max(0, int(margin_b.x_min - ox))
        y0 = max(0, int(margin_b.y_min - oy))
        x1 = min(w, int(margin_b.x_max - ox))
        y1 = min(h, int(margin_b.y_max - oy))
        mask[y0:y1, x0:x1] = 255

    return mask


def overlapping_heights(bboxes: list[BBox]) -> list[float]:
    return [b.y_max - b.y_min for b in bboxes]


def overlapping_bboxes(table_bbox: BBox, bboxes: list[BBox], min_overlap_ratio: float = 0.6) -> list[BBox]:
    return [b for b in bboxes if table_bbox.intersect(b) / b.area >= min_overlap_ratio]


def is_full_page_frame(bbox: BBox, width: int, height: int, frac: float = 0.95) -> bool:
    return bbox.width > width * frac and bbox.height > height * frac


def is_valid_table_size(bbox: BBox, width: int, height: int, min_area_frac: float, min_w: float, min_h: float) -> bool:
    return bbox.area > min_area_frac * width * height and bbox.width > min_w and bbox.height > min_h


def has_text_inside(table_bbox: BBox, ocr_bboxes: list[BBox], min_overlap_ratio: float = 0.6) -> bool:
    return any(table_bbox.intersect(r) / r.area >= min_overlap_ratio for r in ocr_bboxes)


def extract_inclined_lines(lines, angle_thresh: float) -> np.ndarray:
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)

    lines_arr = np.asarray(lines, dtype=np.int32).reshape(-1, 4)
    x1, y1, x2, y2 = lines_arr[:, 0], lines_arr[:, 1], lines_arr[:, 2], lines_arr[:, 3]

    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
    is_axis_aligned = (angle < angle_thresh) | (angle > 180 - angle_thresh) | (np.abs(angle - 90) < angle_thresh)

    return lines_arr[~is_axis_aligned]
