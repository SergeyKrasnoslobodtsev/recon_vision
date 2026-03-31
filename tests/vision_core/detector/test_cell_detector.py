from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from vision_core.debug_image_observer import DebugImageObserver
from vision_core.detector.table_cell_detector import TableCellDetector
from vision_core.detector.table_detector import TableDetector
from vision_core.preprocessor.image_orientation import PageOrientationPreprocessor
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor

# command pytest tests/vision_core/detector/test_table_detector.py -v -s


@pytest.fixture
def orientation_preprocessor() -> PageOrientationPreprocessor:
    """Экземпляр PageOrientationPreprocessor"""
    return PageOrientationPreprocessor()


class TestCellDetector:
    """Тесты для TableCellDetector"""

    def test_extract_cells(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        preprocessor_table: TablePreprocessor,
        table_detector: TableDetector,
        cell_detector: TableCellDetector,
        orientation_preprocessor: PageOrientationPreprocessor,
    ):
        """Тестирует детекцию таблиц на изображении"""

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")

        observer = DebugImageObserver(output_dir=output_dir)

        for test_file in pdf_files:
            logger.info(f"Тестирование на файле: {test_file.name}")
            pdf_bytes = test_file.read_bytes()
            original = pdf_loader_single_page(pdf_bytes)

            processed = preprocessor_img.process(original)
            processed, _ = orientation_preprocessor.process(processed)

            table_mask = preprocessor_table.create_table_mask(processed)

            bboxes = table_detector.extract_raw_tables(table_mask)

            items: list[tuple[tuple[int, int, int, int], str]] = []
            for bbox in bboxes:
                roi_mask = bbox.roi(table_mask)
                cells_bboxes = cell_detector.extract_cells(
                    roi_mask,
                    bbox.to_tuple(),
                    merge_mode="cols",
                )

                for cell_bbox in cells_bboxes:
                    drawer.draw_structure(
                        cell_bbox.bbox.to_tuple(),
                        label=f"R{cell_bbox.row}C{cell_bbox.col}S{cell_bbox.colspan}",
                        color="darkgreen",
                        position=Position.TOP,
                    )

            drawer.save(output_dir / f"detected_cells_{test_file.stem}.png")
