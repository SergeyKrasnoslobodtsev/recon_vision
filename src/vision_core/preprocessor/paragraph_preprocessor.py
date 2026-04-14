import cv2
import numpy as np
from loguru import logger

from vision_core.config import ParagraphPreprocessorConfig
from vision_core.entities.bbox import BBox


class ParagraphPreprocessor:
    """Выделяет сырые текстовые регионы на изображении страницы.

    Маскирует таблицы, бинаризует изображение и через морфологические операции
    получает контуры текстовых блоков.
    """

    def __init__(self, cfg: ParagraphPreprocessorConfig | None = None):
        self.cfg = cfg or ParagraphPreprocessorConfig()

    def extract_regions(
        self,
        image: np.ndarray,
        table_bboxes: list[BBox],
        median_row_height: float,
    ) -> list[BBox]:
        """Возвращает сырые bbox текстовых регионов, отсортированные по (y, x).

        Args:
            image: RGB или grayscale изображение страницы.
            table_bboxes: Области таблиц, которые нужно исключить.
            median_row_height: Медианная высота OCR-строки в пикселях.

        Returns:
            Список BBox текстовых регионов.
        """
        masked = self._mask_tables(image, table_bboxes)
        binary = self._binarize(masked)
        morphed = self._morphology(binary, median_row_height)
        regions = self._find_regions(morphed, page_height=image.shape[0])
        logger.debug(f"ParagraphPreprocessor: найдено {len(regions)} регионов")
        return regions

    def _mask_tables(self, image: np.ndarray, table_bboxes: list[BBox]) -> np.ndarray:
        result = image.copy()
        for bbox in table_bboxes:
            x1, y1 = int(bbox.x_min), int(bbox.y_min)
            x2, y2 = int(bbox.x_max), int(bbox.y_max)
            result[y1:y2, x1:x2] = 255
        return result

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.cfg.adaptive_block_size,
            self.cfg.adaptive_c,
        )

    def _morphology(self, binary: np.ndarray, median_row_height: float) -> np.ndarray:
        kw = max(self.cfg.dilate_kw_min, int(median_row_height * self.cfg.dilate_kw_ratio))
        kh = max(self.cfg.dilate_kh_min, int(median_row_height * self.cfg.dilate_kh_ratio))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        dilated = cv2.dilate(binary, dilate_kernel, iterations=1)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.cfg.erode_kw, 1))
        return cv2.erode(dilated, erode_kernel, iterations=1)

    def _find_regions(self, morphed: np.ndarray, page_height: int) -> list[BBox]:
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cutoff_y = page_height * self.cfg.page_bottom_cutoff
        regions: list[BBox] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if y > cutoff_y:
                continue
            regions.append(BBox(x_min=float(x), y_min=float(y), x_max=float(x + w), y_max=float(y + h)))
        regions.sort(key=lambda b: (b.y_min, b.x_min))
        return regions
