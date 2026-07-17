import numpy as np
from loguru import logger

from vision_core.config import TableDetectorConfig
from vision_core.debug_image_observer import DebugImageObserver
from vision_core.entities.bbox import BBox
from vision_core.entities.table import Table
from vision_core.ocr.base import OcrResult
from vision_core.utils import geometry_utils, image_utils


class TableDetector:
    """Детектор таблиц на изображении"""

    def __init__(
        self,
        table_detector_config: TableDetectorConfig | None = None,
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
        self._debug_image = debug_image

        self._sharpness_img = None
        self._binary_img = None
        self._mask_raw_lines = None
        self._mask_clean_lines = None
        self._binary_img_clean_lines = None
        self._mask_fixed = None
        self.ocr_bboxes = None

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
        diag_lines = _extract_inclined_lines(lines, angle_thresh=1.0)
        self._mask_clean_lines = image_utils.erase_lines(self._mask_raw_lines, diag_lines, thickness=3)
        # эта маска нужна для удаления текста чтобы получить явные линии внутри таблиц
        self._binary_img_clean_lines = image_utils.erase_lines(self._binary_img, diag_lines, thickness=3)
        # для поиска кандидатов таблиц заполним разрывы
        self._mask_fixed = image_utils.repair_mask(self._mask_clean_lines, close_size=7)

    def detect_tables(
        self,
        image: np.ndarray,
        ocr_results: list[OcrResult],
        page_number: int = 0,
    ) -> list[Table]:
        self._preprocess(image, page_number)

        self.ocr_bboxes = [BBox.from_tuple(r.bbox) for r in ocr_results]

        return []

    def clean_mask_table(): ...

    def _find_table_candidates(self) -> list[BBox]:
        cnts = geometry_utils.find_top_level_contours(self._mask_fixed)
        table_candidates = []
        for c in cnts:
            x, y, w, h = geometry_utils.bounding_rect(c)
            bbox = BBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h)

            if _is_full_page_frame(bbox, self._w, self._h):
                continue
            if not _is_valid_table_size(bbox, self._w, self._h, self.cfg.min_area_frac, self.cfg.min_w, self.cfg.min_h):
                continue
            if not _has_text_inside(bbox, self.ocr_bboxes, self.cfg.min_overlap_ratio):
                continue

            table_candidates.append(bbox)

        return sorted(table_candidates, key=lambda b: (b.y_min, b.x_min))


def _is_full_page_frame(bbox: BBox, width: int, height: int, frac: float = 0.95) -> bool:
    return bbox.width > width * frac and bbox.height > height * frac


def _is_valid_table_size(bbox: BBox, width: int, height: int, min_area_frac: float, min_w: float, min_h: float) -> bool:
    return bbox.area > min_area_frac * width * height and bbox.width > min_w and bbox.height > min_h


def _has_text_inside(table_bbox: BBox, ocr_bboxes: list[BBox], min_overlap_ratio: float = 0.5) -> bool:
    return any(table_bbox.intersect(r) / r.area >= min_overlap_ratio for r in ocr_bboxes)


def _extract_inclined_lines(lines, angle_thresh: float) -> np.ndarray:
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)

    lines_arr = np.asarray(lines, dtype=np.int32).reshape(-1, 4)
    x1, y1, x2, y2 = lines_arr[:, 0], lines_arr[:, 1], lines_arr[:, 2], lines_arr[:, 3]

    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
    is_axis_aligned = (angle < angle_thresh) | (angle > 180 - angle_thresh) | (np.abs(angle - 90) < angle_thresh)

    return lines_arr[~is_axis_aligned]
