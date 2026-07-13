import numpy as np
from loguru import logger

from vision_core.config import ParagraphDetectorConfig, ParagraphPreprocessorConfig
from vision_core.entities.bbox import BBox
from vision_core.entities.paragraph import Paragraph, ParagraphType
from vision_core.entities.table import Table
from vision_core.ocr.base import OcrResult
from vision_core.preprocessor.paragraph_preprocessor import ParagraphPreprocessor


class ParagraphDetector:
    """Детектирует и классифицирует текстовые параграфы на изображении страницы.

    Алгоритм:
    1. Вычисляет median_row_height из OCR-результатов.
    2. Через ParagraphPreprocessor получает сырые текстовые регионы (list[BBox]).
    3. Маппит OCR-строки на регионы -> list[Paragraph].
    4. Сливает параграфы на одной горизонтальной строке.
    5. Классифицирует каждый параграф по позиции.
    6. Сортирует в порядке чтения (полосами).
    """

    def __init__(
        self,
        config: ParagraphDetectorConfig | None = None,
        preprocessor_config: ParagraphPreprocessorConfig | None = None,
        debug_image=None,
    ):
        self.cfg = config or ParagraphDetectorConfig()
        self._preprocessor = ParagraphPreprocessor(preprocessor_config)
        self._debug = debug_image

    def detect_paragraphs(
        self,
        image: np.ndarray,
        ocr_results: list[OcrResult],
        tables: list[Table],
        page_number: int = 0,
    ) -> list[Paragraph]:
        """Детектирует параграфы на странице.

        Args:
            image: RGB изображение страницы.
            ocr_results: OCR-строки вне таблиц.
            tables: Таблицы страницы (для маскирования и классификации подписей).

        Returns:
            Параграфы в порядке чтения.
        """
        if not ocr_results:
            logger.warning("OCR-результаты пусты, параграфы не обнаружены")
            return []

        median_row_height = self._median_row_height(ocr_results) * 0.5
        table_bboxes = [t.bbox for t in tables]

        regions = self._preprocessor.extract_regions(image, table_bboxes, median_row_height)
        paragraphs = self._map_ocr_to_regions(ocr_results, regions, median_row_height)

        page_shape = (image.shape[0], image.shape[1])
        paragraphs = [p.classify_by_position(page_shape, table_bboxes, median_row_height) for p in paragraphs]
        paragraphs = self._reading_order(paragraphs, median_row_height)

        if self._debug:
            self._debug.on_detected_boxes(
                image,
                boxes=[p.bbox.to_tuple() for p in paragraphs],
                stage="7_paragraphs",
                page_number=page_number,
                color="darkorange",
                fill=(255, 165, 0, 80),
            )

        logger.info(
            f"Параграфов: {len(paragraphs)} "
            f"(HEADER={sum(p.type == ParagraphType.PAGE_HEADER for p in paragraphs)}, "
            f"FOOTER={sum(p.type == ParagraphType.PAGE_FOOTER for p in paragraphs)}, "
            f"TITLE={sum(p.type == ParagraphType.SECTION_TITLE for p in paragraphs)}, "
            f"CAPTION={sum(p.type == ParagraphType.TABLE_CAPTION for p in paragraphs)}, "
            f"TEXT={sum(p.type == ParagraphType.BODY_TEXT for p in paragraphs)})"
        )
        return paragraphs

    # ------------------------------------------------------------------ #

    def _median_row_height(self, ocr_results: list[OcrResult]) -> float:
        heights = [float(r.bbox[3] - r.bbox[1]) for r in ocr_results]
        return float(np.median(heights)) if heights else 1.0

    def _map_ocr_to_regions(self, ocr_results: list[OcrResult], regions: list[BBox], line_tolerance) -> list[Paragraph]:
        paragraphs: list[Paragraph] = []
        for region in regions:
            matched = [r for r in ocr_results if region.contains_center(BBox.from_tuple(r.bbox))]

            if not matched:
                continue

            matched.sort(key=lambda l: (int(float(l.bbox[1]) // line_tolerance), float(l.bbox[0])))
            blobs = [BBox.from_tuple(r.bbox) for r in matched]
            p = Paragraph(
                id=str(len(paragraphs)),
                text=" ".join(r.text for r in matched),
                bbox=BBox(
                    x_min=min(b.x_min for b in blobs),
                    y_min=min(b.y_min for b in blobs),
                    x_max=max(b.x_max for b in blobs),
                    y_max=max(b.y_max for b in blobs),
                ),
                blobs=blobs,
            )

            paragraphs.append(p)
            logger.debug(f"region[{p.id}] строк={len(blobs)}")
            for l in matched:
                logger.debug(f" x={float(l.bbox[0]):.0f} y={float(l.bbox[1]):.0f}  '{l.text}'")

        return paragraphs

    def _merge_same_line(self, paragraphs: list[Paragraph], median_row_height: float) -> list[Paragraph]:
        result: list[Paragraph] = []
        for p in paragraphs:
            if result and result[-1].is_same_line(p, median_row_height):
                result[-1] = result[-1].merge_with(p)
            else:
                result.append(p)
        return result

    def _reading_order(
        self,
        paragraphs: list[Paragraph],
        median_row_height: float,
    ) -> list[Paragraph]:
        band_tolerance = median_row_height * self.cfg.reading_order_band_k
        sorted_para = sorted(paragraphs, key=lambda p: (p.bbox.y_min, p.bbox.x_min))

        band_id = 0
        band_y_start = -1.0
        band_ids: list[int] = []
        for p in sorted_para:
            if abs(p.bbox.y_min - band_y_start) > band_tolerance:
                band_id += 1
                band_y_start = p.bbox.y_min
            band_ids.append(band_id)

        return [
            p
            for _, p in sorted(
                zip(band_ids, sorted_para, strict=True),
                key=lambda x: (x[0], x[1].bbox.x_min),
            )
        ]
