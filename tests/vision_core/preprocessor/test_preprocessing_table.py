from pathlib import Path

import cv2
from loguru import logger

from vision_core.config import TablePreprocessorConfig
from vision_core.debug_image_observer import DebugImageObserver
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.preprocessor.image_orientation import PageOrientationPreprocessor
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.utils.image_utils import compute_raw_line_mask


class TestPreprocessingTable:
    def test_detected_lines(self, pdf_file: Path, output_dir: Path):
        """Тестирует детекцию сырых линий -- визуализация ручная."""
        observer = DebugImageObserver(output_dir=output_dir)
        cfg = TablePreprocessorConfig()

        with PDFLoader(pdf_file.read_bytes()) as loader:
            original = loader.get_page_image(0, dpi=300)

        oriented_img, _ = PageOrientationPreprocessor().process(original)
        processed = ImagePreprocessor().process(oriented_img)
        gray_img = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)

        h_lines, v_lines = compute_raw_line_mask(gray_img, cfg.scale)

        observer.on_debug_image(
            oriented_img, h_lines + v_lines, stage="table_lines_detection", prefix=pdf_file.stem, page_number=0
        )
        logger.success(f"Детекция линий пройдена: {pdf_file.name}")

    def test_create_table_mask(self, pdf_file: Path, output_dir: Path):
        observer = DebugImageObserver(output_dir=output_dir)

        with PDFLoader(pdf_file.read_bytes()) as loader:
            original = loader.get_page_image(0, dpi=300)

        oriented_img, _ = PageOrientationPreprocessor().process(original)
        processed = ImagePreprocessor().process(oriented_img)
        gray_img = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        TablePreprocessor(debug_image=observer).create_table_mask(gray_img)

        logger.success(f"Маска таблиц создана: {pdf_file.name}")
