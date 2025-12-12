import pytest
import cv2
from pathlib import Path
from loguru import logger
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.detector.table_detector import TableDetector
from vision_core.detector.table_cell_detector import TableCellDetector

import numpy as np


# command pytest tests/vision_core/detector/test_table_detector.py -v -s


class TestCellDetector:
    """Тесты для TableCellDetector"""

    def test_find_lines_from_mask_table(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        preprocessor_table: TablePreprocessor,
        table_detector: TableDetector,
        cell_detector: TableCellDetector,
        drawer_bbox_and_label,
    ):
        """Тестирует детекцию таблиц на изображении"""

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")

        for test_file in pdf_files[:1]:
            logger.info(f"Тестирование на файле: {test_file.name}")
            pdf_bytes = test_file.read_bytes()
            original = pdf_loader_single_page(pdf_bytes)

            processed = preprocessor_img.process(original)

            table_mask = preprocessor_table.create_table_mask(processed)

            bboxes = table_detector.extract_raw_tables(table_mask)

            debug_image = original.copy()

            for i, bbox in enumerate(bboxes):
                debug_image = drawer_bbox_and_label(
                    debug_image,
                    bbox,
                    label=f"Table {i + 1}",
                    color="blue",
                    position="top",
                )

                cells_bboxes = cell_detector.extract_cells(
                    table_mask,
                    bbox,
                    merge_mode="all",
                )

                for cell_bbox in cells_bboxes:
                    debug_image = drawer_bbox_and_label(
                        debug_image,
                        cell_bbox.bbox,
                        label=f"R{cell_bbox.row}C{cell_bbox.col}S{cell_bbox.colspan}",
                        color="darkgreen",
                        position="bottom",
                    )

            debug_image.save(output_dir / f"detected_cells_{test_file.stem}.png")
