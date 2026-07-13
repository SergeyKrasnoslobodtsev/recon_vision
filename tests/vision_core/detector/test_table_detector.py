from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from vision_core.debug_image_observer import DebugImageObserver
from vision_core.detector.table_detector import TableDetector
from vision_core.entities.bbox import BBox
from vision_core.entities.table import Table
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.preprocessor.image_orientation import PageOrientationPreprocessor
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor


class TestTableDetector:
    @pytest.fixture
    def table_detector(self) -> TableDetector:
        return TableDetector()

    def test_get_table_line_mask_uses_explicit_mask(self, table_detector: TableDetector):
        table = Table(id="table_0", bbox=BBox(x_min=1, y_min=1, x_max=4, y_max=4), num_rows=2, num_cols=2)
        table_mask = np.zeros((6, 6), dtype=np.uint8)
        table_mask[1:4, 1:4] = 255

        roi_mask = table_detector.get_table_line_mask(table, table_mask)

        assert roi_mask.shape == (3, 3)
        assert np.all(roi_mask == 255)

    def test_remove_table_lines_uses_passed_mask_without_hidden_state(self, table_detector: TableDetector):
        image = np.zeros((6, 6), dtype=np.uint8)
        image[1:4, 1:4] = 100
        table = Table(id="table_0", bbox=BBox(x_min=1, y_min=1, x_max=4, y_max=4), num_rows=2, num_cols=2)
        table_mask = np.zeros((6, 6), dtype=np.uint8)
        table_mask[2, 1:4] = 255

        result = table_detector.remove_table_lines(
            image=image, table=table, fill_value=255, padding=0, table_mask=table_mask
        )

        assert np.all(result[2, 1:4] == 255)
        assert result[1, 1] == 100

    def test_find_tables(self, pdf_file: Path, output_dir: Path):
        observer = DebugImageObserver(output_dir=output_dir)

        with PDFLoader(pdf_file.read_bytes()) as loader:
            original = loader.get_page_image(0, dpi=300)

        processed = ImagePreprocessor().process(original)
        oriented, _ = PageOrientationPreprocessor().process(processed)
        TableDetector(debug_image=observer).extract_raw_tables(
            TablePreprocessor(debug_image=observer).create_table_mask(oriented)
        )
        logger.success(f"Детекция таблиц пройдена: {pdf_file.name}")

    def test_extract_tables(self, pdf_file: Path, output_dir: Path):
        observer = DebugImageObserver(output_dir=output_dir)

        with PDFLoader(pdf_file.read_bytes()) as loader:
            original = loader.get_page_image(0, dpi=300)

        processed = ImagePreprocessor().process(original)
        oriented, _ = PageOrientationPreprocessor().process(processed)
        tables = TableDetector(debug_image=observer).detect_tables(oriented)

        if not tables:
            logger.warning(f"Таблицы не найдены: {pdf_file.name}")

        logger.success(f"Структура таблиц извлечена: {pdf_file.name}")
