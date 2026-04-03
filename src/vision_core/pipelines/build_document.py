"""Собирает канонический Document из PDF-файла."""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from vision_core.config import VisionCoreConfig
from vision_core.detector.paragraph_detector import ParagraphDetector
from vision_core.detector.table_detector import TableDetector
from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.ocr.base import OcrResult
from vision_core.ocr.paddle_ocr import PaddleOcrEngine
from vision_core.postprocessor.cell_text_filler import CellTextFiller
from vision_core.postprocessor.table_continuation_linker import TableContinuationLinker
from vision_core.postprocessor.table_id_assigner import TableIdAssigner
from vision_core.preprocessor.image_orientation import PageOrientationPreprocessor
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor


class DocumentBuildPipeline:
    """Строит каноническое представление документа из PDF.

    Шаги для каждой страницы:
        1. Коррекция ориентации и наклона.
        2. Предобработка изображения.
        3. Детекция таблиц и ячеек.
        4. OCR-распознавание текста.
        5. Заполнение ячеек текстом.
        6. Детекция абзацев из текста вне таблиц.
    """

    def __init__(self, config: VisionCoreConfig | None = None) -> None:
        cfg = config or VisionCoreConfig()
        self.orientation_preprocessor = PageOrientationPreprocessor(cfg.page_orientation_preprocessor)
        self.image_preprocessor = ImagePreprocessor(cfg.image_preprocessor)
        self.table_detector = TableDetector(
            preprocessor_config=cfg.table_preprocessor,
            table_detector_config=cfg.table_detector,
            cell_detector_config=cfg.cell_detector,
        )
        self.ocr_engine = PaddleOcrEngine(cfg.paddleocr)
        self.paragraph_detector = ParagraphDetector(cfg.paragraph_detector)
        self.cell_text_filler = CellTextFiller(cfg.ocr_confidence_threshold)
        self.continuation_linker = TableContinuationLinker()
        self.table_id_assigner = TableIdAssigner()
        self.dpi = cfg.dpi

    def build(self, pdf_bytes: bytes) -> Document:
        """Строит канонический документ из PDF.

        Args:
            pdf_bytes: Исходный PDF-файл в байтах.

        Returns:
            Каноническое представление документа.
        """
        pages: list[Page] = []

        with PDFLoader(pdf_bytes) as loader:
            for page_number in range(loader.num_pages):
                image = loader.get_page_image(page_number, dpi=self.dpi)
                page = self._process_page(image, page_number)
                page.page_number = page_number
                page.metadata.update(
                    {
                        "source_page_number": page_number,
                        "page_size": list(loader.get_page_size(page_number)),
                        "has_text_layer": loader.has_text_layer(page_number),
                    }
                )
                pages.append(page)

        self.table_id_assigner.assign(pages)
        self.continuation_linker.link(pages)

        return Document.from_pdf_bytes(
            pdf_bytes=pdf_bytes,
            pages=pages,
            metadata={"dpi": self.dpi, "num_pages": len(pages)},
        )

    def _process_page(self, image: np.ndarray, page_number: int) -> Page:
        """Обрабатывает одну страницу документа.

        Args:
            image: Изображение страницы.
            page_number: Номер страницы.

        Returns:
            Страница с таблицами, абзацами и метаданными.
        """
        image, alignment_metadata = self.orientation_preprocessor.process(image)
        image = self.image_preprocessor.process(image)

        tables = self.table_detector.detect_tables(image)
        logger.debug(f"Страница {page_number}: обнаружено таблиц: {len(tables)}")

        ocr_results = self._run_ocr(image)

        self.cell_text_filler.fill_cells(tables, ocr_results)
        filtered_ocr = self.cell_text_filler.exclude_table_text(ocr_results, tables)

        paragraphs = self.paragraph_detector.detect_paragraphs(
            filtered_ocr,
            image.shape[:2],
        )

        return Page(
            tables=tables,
            paragraphs=paragraphs,
            metadata={"image_shape": list(image.shape[:2]), **alignment_metadata},
        )

    def _run_ocr(self, image: np.ndarray) -> list[OcrResult]:
        """Запускает OCR на предобработанном изображении.

        Args:
            image: Предобработанное изображение страницы (grayscale).

        Returns:
            Список OCR-результатов.
        """
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        results = self.ocr_engine.predict(bgr_image)
        if not results or not results[0]:
            logger.warning("OCR не распознал текст на странице.")
            return []
        logger.debug(
            f"OCR: распознано {len(results[0])} блоков, "
            f"средняя уверенность: {np.mean([r.confidence for r in results[0]]):.2f}"
        )
        return results[0]
