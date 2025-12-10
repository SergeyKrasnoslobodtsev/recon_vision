import pytest
import cv2
from pathlib import Path
from loguru import logger
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.detector.table_detector import TableDetector


def test_enhance_real_documents():
    """Тестирует улучшение реальных документов и сохраняет результаты"""

    test_dir = Path("./examples/test")
    output_dir = Path("./examples/output/enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        pytest.skip(f"Папка с тестовыми файлами не найдена: {test_dir}")

    pdf_files = list(test_dir.glob("*.pdf"))

    if not pdf_files:
        pytest.skip(f"PDF файлы не найдены в {test_dir}")

    logger.info(f"Найдено PDF файлов: {len(pdf_files)}")

    preprocessor = ImagePreprocessor()

    for pdf_path in sorted(pdf_files):
        logger.info(f"Обработка: {pdf_path.name}")

        pdf_bytes = pdf_path.read_bytes()

        with PDFLoader(pdf_bytes) as loader:
            # Обрабатываем первую страницу
            original = loader.get_page_image(0, dpi=300)

            # Применяем препроцессинг
            enhanced = preprocessor.process(original)

            # Сохраняем результаты
            output_name = pdf_path.stem + "_enhanced.png"
            output_path = Path(output_dir / output_name)

            cv2.imwrite(str(output_path), enhanced)
            logger.success(f"Сохранено: {output_path}")

    logger.info(f"Все файлы обработаны. Результаты в {output_dir}")
