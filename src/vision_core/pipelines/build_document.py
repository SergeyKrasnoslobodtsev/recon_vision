"""Собирает канонический Document из PDF-файла."""

from __future__ import annotations

import numpy as np
from loguru import logger

from vision_core.config import VisionCoreConfig
from vision_core.debug_image_observer import DebugImageObserver
from vision_core.detector.paragraph_detector import ParagraphDetector
from vision_core.detector.table_detector import TableDetector
from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.ocr.base import OcrResult
from vision_core.ocr.paddle_ocr import PaddleOcrEngine
from vision_core.postprocessor.cell_text_filler import CellTextFiller
from vision_core.postprocessor.dc_cols_resolver import DcColsResolver
from vision_core.postprocessor.debit_credit_processor import DebitCreditProcessor
from vision_core.postprocessor.row_splitter import RowSplitter
from vision_core.postprocessor.table_continuation_linker import TableContinuationLinker
from vision_core.postprocessor.table_id_assigner import TableIdAssigner
from vision_core.preprocessor.image_orientation import PageOrientationPreprocessor
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.utils.drawer import Drawer


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

    def __init__(
        self,
        config: VisionCoreConfig | None = None,
        debug_image: DebugImageObserver | None = None,
    ) -> None:
        cfg = config or VisionCoreConfig()
        self._debug_image = debug_image

        self.image_preprocessor = ImagePreprocessor(cfg.image_preprocessor, debug_image=debug_image)
        self.orientation_preprocessor = PageOrientationPreprocessor(
            cfg.page_orientation_preprocessor, debug_image=debug_image
        )
        self.table_detector = TableDetector(
            preprocessor_config=cfg.table_preprocessor,
            table_detector_config=cfg.table_detector,
            debug_image=debug_image,
        )
        self.ocr_engine = PaddleOcrEngine(cfg.paddleocr)
        self.cell_text_filler = CellTextFiller(
            cfg.ocr_confidence_threshold,
            debug_image=debug_image,
            ocr_engine=self.ocr_engine,
        )
        self.paragraph_detector = ParagraphDetector(
            cfg.paragraph_detector, cfg.paragraph_preprocessor, debug_image=debug_image
        )
        self.continuation_linker = TableContinuationLinker()
        self.dc_cols_resolver = DcColsResolver()
        self.row_splitter = RowSplitter()
        self.debit_credit_processor = DebitCreditProcessor()
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
        aligned_images: list[np.ndarray] = []

        with PDFLoader(pdf_bytes) as loader:
            for page_number in range(loader.num_pages):
                logger.info(f"Обработка страницы {page_number} с dpi {self.dpi}...")

                image = loader.get_page_image(page_number, dpi=self.dpi)
                page, aligned_image = self._process_page(image, page_number)
                page.page_number = page_number
                page.metadata.update(
                    {
                        "source_page_number": page_number,
                        "page_size": list(loader.get_page_size(page_number)),
                        "has_text_layer": loader.has_text_layer(page_number),
                    }
                )
                pages.append(page)
                aligned_images.append(aligned_image)

                logger.debug(
                    f"Страница {page_number}: "
                    f"{len(page.tables)} таблиц, "
                    f"{len(page.paragraphs)} абзацев, "
                    f"метаданные: {page.metadata}"
                )

                logger.info("Обработка страницы завершена.")

        self.table_id_assigner.assign(pages)
        self.continuation_linker.link(pages)
        self.dc_cols_resolver.resolve(pages)
        self.row_splitter.split(pages)
        self.debit_credit_processor.process(pages)

        if self._debug_image:
            table_color_map = self._build_table_color_map(pages)
            for page, aligned_image in zip(pages, aligned_images, strict=True):
                self._debug_image.on_document_page(
                    aligned_image,
                    page=page,
                    table_color_map=table_color_map,
                    stage="8_document",
                    page_number=page.page_number,
                )

        return Document.from_pdf_bytes(
            pdf_bytes=pdf_bytes,
            pages=pages,
            metadata={"dpi": self.dpi, "num_pages": len(pages)},
        )

    def _process_page(self, image: np.ndarray, page_number: int) -> tuple[Page, np.ndarray]:
        """Обрабатывает одну страницу документа.

        Args:
            image: Изображение страницы.
            page_number: Номер страницы.

        Returns:
            Кортеж (страница, выровненное изображение до препроцессора).
        """

        logger.info("Коррекция ориентации и наклона...")
        aligned_image, alignment_metadata = self.orientation_preprocessor.process(image, page_number=page_number)
        logger.info("Коррекция завершена.")

        logger.info("Предобработка изображения...")
        image_ocr = self.image_preprocessor.process(aligned_image, page_number=page_number)
        logger.info("Предобработка завершена.")

        logger.info("Детекция таблиц и ячеек...")
        tables = self.table_detector.detect_tables(aligned_image)
        logger.debug(f"Страница {page_number}: обнаружено таблиц: {len(tables)}")
        logger.info("Детекция таблиц и ячеек завершена.")

        logger.info("Распознавание текста OCR...")
        ocr_results = self._run_ocr(image_ocr)
        logger.info("Распознавание текста OCR завершено.")

        logger.info("Заполнение ячеек текстом...")
        self.cell_text_filler.fill_cells(tables, ocr_results, image=aligned_image, page_number=page_number)
        logger.info("Заполнение ячеек текстом завершено.")

        logger.info("Детекция абзацев...")
        filtered_ocr = self.cell_text_filler.exclude_table_text(ocr_results, tables)
        paragraphs = self.paragraph_detector.detect_paragraphs(
            aligned_image, filtered_ocr, tables, page_number=page_number
        )
        logger.info("Детекция абзацев завершена.")

        page = Page(
            tables=tables,
            paragraphs=paragraphs,
            metadata={
                "source_image_shape": list(image.shape[:2]),
                "image_shape": list(aligned_image.shape[:2]),
                **alignment_metadata,
            },
        )
        return page, aligned_image

    def _run_ocr(self, image: np.ndarray) -> list[OcrResult]:
        """Запускает OCR на предобработанном изображении.

        Args:
            image: Предобработанное изображение страницы (grayscale).

        Returns:
            Список OCR-результатов.
        """
        results = self.ocr_engine.predict(image)
        if not results or not results[0]:
            logger.warning("OCR не распознал текст на странице.")
            return []
        logger.debug(
            f"OCR: распознано {len(results[0])} блоков, "
            f"средняя уверенность: {np.mean([r.confidence for r in results[0]]):.2f}"
        )
        return results[0]

    @staticmethod
    def _build_table_color_map(
        pages: list[Page],
    ) -> dict[str, tuple[int, int, int, int]]:
        """Строит карту цветов для таблиц документа.

        Таблицы одной цепочки продолжений получают одинаковый цвет.
        Цвета берутся циклически из палитры Drawer.

        Args:
            pages: Все страницы документа после постпроцессоров.

        Returns:
            Словарь {table.id: RGBA-цвет}.
        """
        all_tables = {t.id: t for page in pages for t in page.tables}

        def find_root(table_id: str) -> str:
            t = all_tables.get(table_id)
            if t and t.continuation_of and t.continuation_of in all_tables:
                return find_root(t.continuation_of)
            return table_id

        roots: list[str] = []
        for page in pages:
            for table in page.tables:
                root = find_root(table.id)
                if root not in roots:
                    roots.append(root)

        palette = Drawer._CYCLIC_PALETTE
        root_colors = {root: palette[i % len(palette)] for i, root in enumerate(roots)}
        return {tid: root_colors[find_root(tid)] for tid in all_tables}
