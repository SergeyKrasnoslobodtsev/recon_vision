from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from vision_core.debug_image_observer import DebugImageObserver
from vision_core.detector.table_detector import TableDetector
from vision_core.entities.bbox import BBox
from vision_core.entities.table import Table
from vision_core.preprocessor.image_orientation import PageOrientationPreprocessor
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor

# command pytest tests/vision_core/detector/test_table_detector.py -v -s


@pytest.fixture
def orientation_preprocessor() -> PageOrientationPreprocessor:
    """Экземпляр PageOrientationPreprocessor"""
    return PageOrientationPreprocessor()


class TestTableDetector:
    """Тесты для TableDetector"""

    def test_get_table_line_mask_uses_explicit_mask(self, table_detector: TableDetector):
        table = Table(
            id="table_0",
            bbox=BBox(x_min=1, y_min=1, x_max=4, y_max=4),
            num_rows=2,
            num_cols=2,
        )
        table_mask = np.zeros((6, 6), dtype=np.uint8)
        table_mask[1:4, 1:4] = 255

        roi_mask = table_detector.get_table_line_mask(table, table_mask)

        assert roi_mask.shape == (3, 3)
        assert np.all(roi_mask == 255)

    def test_remove_table_lines_uses_passed_mask_without_hidden_state(self, table_detector: TableDetector):
        image = np.zeros((6, 6), dtype=np.uint8)
        image[1:4, 1:4] = 100
        table = Table(
            id="table_0",
            bbox=BBox(x_min=1, y_min=1, x_max=4, y_max=4),
            num_rows=2,
            num_cols=2,
        )
        table_mask = np.zeros((6, 6), dtype=np.uint8)
        table_mask[2, 1:4] = 255

        result = table_detector.remove_table_lines(
            image=image,
            table=table,
            fill_value=255,
            padding=0,
            table_mask=table_mask,
        )

        assert np.all(result[2, 1:4] == 255)
        assert result[1, 1] == 100

    def test_find_tables(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        preprocessor_table: TablePreprocessor,
        orientation_preprocessor: PageOrientationPreprocessor,
        table_detector: TableDetector,
    ):
        """Тестирует детекцию таблиц на изображении"""

        observer = DebugImageObserver(output_dir=output_dir)

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")

        for test_file in pdf_files:
            logger.info(f"Тестирование на файле: {test_file.name}")
            pdf_bytes = test_file.read_bytes()

            original = pdf_loader_single_page(pdf_bytes)

            processed_img = preprocessor_img.process(original)
            oriented_img, _ = orientation_preprocessor.process(processed_img)

            mask_table = preprocessor_table.create_table_mask(oriented_img)

            bboxes = table_detector.extract_raw_tables(mask_table)

            observer.on_labeled_boxes(
                oriented_img,
                items=[(bbox.to_tuple(), f"Table {index}") for index, bbox in enumerate(bboxes, start=1)],
                stage="detected_tables",
                prefix=test_file.stem,
                page_number=0,
                color="blue",
            )

    def test_extract_tables(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        orientation_preprocessor: PageOrientationPreprocessor,
        table_detector: TableDetector,
    ):
        """Тестирует детекцию таблиц на изображении"""

        observer = DebugImageObserver(output_dir=output_dir)

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")

        for test_file in pdf_files:
            logger.info(f"Тестирование на файле: {test_file.name}")
            pdf_bytes = test_file.read_bytes()

            original = pdf_loader_single_page(pdf_bytes)

            processed_img = preprocessor_img.process(original)
            oriented_img, _ = orientation_preprocessor.process(processed_img)

            tables = table_detector.detect_tables(oriented_img)

            if not tables:
                logger.warning(f"Таблицы не найдены в файле: {test_file.name}")
            else:
                items: list[tuple[tuple[int, int, int, int], str]] = []
                for index, table in enumerate(tables, start=1):
                    items.append((table.bbox.to_tuple(), f"Table {index}"))
                    for cell in table.cells:
                        items.append((cell.bbox.to_tuple(), f"R{cell.row}C{cell.col}S{cell.colspan}"))

                observer.on_labeled_boxes(
                    oriented_img,
                    items=items,
                    stage="detected_tables_struct",
                    prefix=test_file.stem,
                    page_number=0,
                    color="blue",
                )
