from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from vision_core.debug_observer import DebugObserver
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor


class TestPreprocessingImage:
    """Тесты для PreprocessingImage"""

    def test_preprocessing_image(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
    ):
        """Тестирует препроцессинг изображения"""

        observer = DebugObserver(output_dir=output_dir)

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files[:1]:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")
        for test_file in pdf_files[:1]:
            logger.info(f"Тестирование на файле: {test_file.name}")
            pdf_bytes = test_file.read_bytes()
            original = pdf_loader_single_page(pdf_bytes)

            processed = preprocessor_img.process(original)
            observer.on_side_by_side(
                original,
                processed,
                stage="preprocessing",
                page_number=0,
            )

        logger.success("Тест препроцессинга изображения пройден успешно")
