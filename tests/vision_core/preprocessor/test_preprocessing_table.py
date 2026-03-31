from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from vision_core.config import TablePreprocessorConfig
from vision_core.debug_image_observer import DebugImageObserver
from vision_core.preprocessor.image_orientation import PageOrientationPreprocessor
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor

# command pytest tests/vision_core/detector/test_table_detector.py -v -s


@pytest.fixture
def orientation_preprocessor() -> PageOrientationPreprocessor:
    """Экземпляр PageOrientationPreprocessor"""
    return PageOrientationPreprocessor()


class TestPreprocessingTable:
    """Тесты для PreprocessingTableDetector"""

    def test_detected_lines(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        preprocessor_table: TablePreprocessor,
        orientation_preprocessor: PageOrientationPreprocessor,
    ):
        """Тестирует детекцию линий на изображении"""
        observer = DebugImageObserver(output_dir=output_dir)
        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")

        cfg = TablePreprocessorConfig()

        for test_file in pdf_files[:1]:
            logger.info(f"Тестирование на файле: {test_file.name}")
            pdf_bytes = test_file.read_bytes()
            original = pdf_loader_single_page(pdf_bytes)

            processed = preprocessor_img.process(original)
            oriented_img, _ = orientation_preprocessor.process(processed)
            min_lenght_h = int(oriented_img.shape[0] * cfg.horizontal_length_ratio)
            min_lenght_v = int(oriented_img.shape[1] * cfg.vertical_length_ratio)
            img_table_preprocessing = preprocessor_table._processing(oriented_img)
            h_lines = preprocessor_table._detect_lines(
                img_table_preprocessing,
                kernel_size=min_lenght_h,
                orientation="horizontal",
            )
            v_lines = preprocessor_table._detect_lines(
                img_table_preprocessing,
                kernel_size=min_lenght_v,
                orientation="vertical",
            )
            mask = h_lines + v_lines

            observer.on_debug_image(
                oriented_img,
                mask,
                stage="table_lines_detection",
                prefix=f"{test_file.stem}",
                page_number=0,
            )
        logger.success("Тест детекции линий пройден успешно")

    def test_create_table_mask(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        preprocessor_table: TablePreprocessor,
        orientation_preprocessor: PageOrientationPreprocessor,
    ):
        """Тестирует создание маски таблиц на изображении"""

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

            processed = preprocessor_img.process(original)
            oriented_img, _ = orientation_preprocessor.process(processed)

            table_mask = preprocessor_table.create_table_mask(oriented_img)
            observer.on_debug_image(
                oriented_img,
                table_mask,
                stage="table_mask_creation",
                prefix=f"{test_file.stem}",
                page_number=0,
            )

        logger.success("Тест создания маски таблиц пройден успешно")
