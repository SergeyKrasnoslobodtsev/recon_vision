import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from loguru import logger
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.paragraph_preprocessor import ParagraphPreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.detector.table_detector import TableDetector

# command pytest tests/vision_core/detector/test_table_detector.py -v -s


class TestPreprocessingParagraph:
    """Тесты для PreprocessingParagraph"""

    def test_detected_paragraphs(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        preprocessor_table: TablePreprocessor,
        table_detector: TableDetector,
        preprocessor_paragraph: ParagraphPreprocessor,
    ):
        """Тестирует детекцию линий на изображении"""

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

            mask_table = preprocessor_table.create_table_mask(processed)
            bboxes = table_detector.extract_raw_tables(mask_table)

            paragraph_mask = preprocessor_paragraph.create_paragraph_mask(
                processed, bboxes
            )
            paragraph_mask_image = Image.fromarray(paragraph_mask)
            paragraph_mask_image.save(
                output_dir / f"{test_file.stem}_raw_paragraph_mask.png"
            )

        logger.success("Тест детекции линий пройден успешно")
