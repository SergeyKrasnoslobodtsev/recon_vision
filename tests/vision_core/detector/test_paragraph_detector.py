from pathlib import Path

import numpy as np
from loguru import logger

from vision_core.debug_image_observer import DebugImageObserver
from vision_core.detector.paragraph_detector import ParagraphDetector
from vision_core.detector.table_detector import TableDetector
from vision_core.entities.paragraph import ParagraphType
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.ocr.paddle_ocr import PaddleOcrEngine
from vision_core.postprocessor.cell_text_filler import CellTextFiller
from vision_core.preprocessor.image_orientation import PageOrientationPreprocessor
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor


class TestParagraphDetector:
    def test_returns_empty_for_no_ocr(self):
        detector = ParagraphDetector()
        image = np.ones((800, 600, 3), dtype="uint8") * 255
        assert detector.detect_paragraphs(image, ocr_results=[], tables=[]) == []

    def test_integration(self, pdf_file: Path, output_dir: Path):
        observer = DebugImageObserver(output_dir=output_dir)

        with PDFLoader(pdf_file.read_bytes()) as loader:
            original = loader.get_page_image(0, dpi=300)

        processed = ImagePreprocessor(debug_image=observer).process(original)
        oriented, _ = PageOrientationPreprocessor(debug_image=observer).process(processed)
        tables = TableDetector(debug_image=observer).detect_tables(oriented)
        ocr_results = PaddleOcrEngine().predict([oriented])
        ocr_results = ocr_results[0] if ocr_results else []

        cell_text_filler = CellTextFiller(debug_image=observer)
        cell_text_filler.fill_cells(tables, ocr_results, image=oriented)
        filtered_ocr = cell_text_filler.exclude_table_text(ocr_results, tables)

        paragraphs = ParagraphDetector(debug_image=observer).detect_paragraphs(oriented, filtered_ocr, tables)

        assert isinstance(paragraphs, list)
        assert {p.type for p in paragraphs}.issubset(set(ParagraphType))
        logger.success(f"Детекция параграфов пройдена: {pdf_file.name}")
