import pytest
from pathlib import Path
from loguru import logger
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.preprocessor.paragraph_preprocessor import ParagraphPreprocessor
from vision_core.detector.paragraph_detector import ParagraphDetector
from vision_core.detector.table_detector import TableDetector

import numpy as np


# command pytest tests/vision_core/detector/test_paragraph_detector.py -v -s


class TestParagraphDetector:
    """Тесты для ParagraphDetector"""

    def test_find_paragraphs(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        preprocessor_table: TablePreprocessor,
        table_detector: TableDetector,
        preprocessor_paragraph: ParagraphPreprocessor,
        paragraph_detector: ParagraphDetector,
        drawer_bbox_and_label,
    ):
        """Тестирует детекцию абзацев на изображении"""

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
            mask_table = preprocessor_table.create_table_mask(processed)

            bboxes = table_detector.extract_raw_tables(mask_table)

            mask_paragraph = preprocessor_paragraph.create_paragraph_mask(
                processed, bboxes
            )

            bboxes = paragraph_detector.extract_paragraphs(mask_paragraph)
            debug_image = original.copy()

            for i, bbox in enumerate(bboxes):
                debug_image = drawer_bbox_and_label(
                    debug_image,
                    bbox,
                    label=f"Paragraph {i + 1}",
                    color="darkorange",
                    position="top",
                )

            debug_image.save(output_dir / f"detected_paragraphs_{test_file.stem}.png")
