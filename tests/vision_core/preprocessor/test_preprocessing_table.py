import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from loguru import logger
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.config import TablePreprocessorConfig

# command pytest tests/vision_core/detector/test_table_detector.py -v -s


class TestPreprocessingTable:
    """Тесты для PreprocessingTableDetector"""

    def test_detected_lines(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        preprocessor_table: TablePreprocessor,
    ):
        """Тестирует детекцию линий на изображении"""

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
            min_lenght_h = int(processed.shape[0] * cfg.horizontal_length_ratio)
            min_lenght_v = int(processed.shape[1] * cfg.vertical_length_ratio)
            img_table_preprocessing = preprocessor_table._processing(processed)
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
            Image.fromarray(mask).save(output_dir / f"lines_{test_file.stem}.png")

        logger.success("Тест детекции линий пройден успешно")

    def test_create_table_mask(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        preprocessor_table: TablePreprocessor,
    ):
        """Тестирует создание маски таблиц на изображении"""

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
            Image.fromarray(table_mask).save(
                output_dir / f"table_mask_{test_file.stem}.png"
            )

        logger.success("Тест создания маски таблиц пройден успешно")
