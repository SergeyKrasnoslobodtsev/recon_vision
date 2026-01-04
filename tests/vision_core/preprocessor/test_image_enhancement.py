import pytest
import numpy as np
from pathlib import Path
from loguru import logger
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from PIL import Image


def test_enhance_real_documents(
    pdf_path: Path,
    output_dir: Path,
    pdf_loader_single_page: np.ndarray,
    preprocessor_img: ImagePreprocessor,
):
    """Тестирует улучшение реальных документов и сохраняет результаты"""

    if not pdf_path.exists():
        pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

    pdf_files = list(pdf_path.glob("*.pdf"))

    if not pdf_files:
        pytest.skip(f"PDF файлы не найдены в {pdf_path}")

    logger.info(f"Найдено PDF файлов: {len(pdf_files)}")

    for pdf_path in pdf_files:
        logger.info(f"Обработка: {pdf_path.name}")

        pdf_bytes = pdf_path.read_bytes()

        # Обрабатываем первую страницу
        original = pdf_loader_single_page(pdf_bytes)

        # Применяем препроцессинг
        enhanced = preprocessor_img.process(original)

        Image.fromarray(enhanced).save(output_dir / f"enhanced_{pdf_path.stem}.png")

    logger.info(f"Все файлы обработаны. Результаты в {output_dir}")
