import pytest
import cv2
from pathlib import Path
from loguru import logger
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.paragraph_preprocessor import ParagraphPreprocessor

# command pytest tests/vision_core/detector/test_table_detector.py -v -s


class TestPreprocessingParagraph:
    """Тесты для PreprocessingParagraph"""

    @pytest.fixture
    def pdf_path(self) -> Path:
        """Путь к папке с тестовыми PDF"""
        return Path("./examples/test")

    @pytest.fixture
    def output_dir(self) -> Path:
        """Путь к папке для сохранения результатов"""
        output_dir = Path("./examples/output/paragraph_detection")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @pytest.fixture
    def preprocessor(self) -> ParagraphPreprocessor:
        """Экземпляр PreprocessingParagraph"""
        return ParagraphPreprocessor()

    def test_detected_paragraphs(
        self, pdf_path: Path, output_dir: Path, preprocessor: ParagraphPreprocessor
    ):
        """Тестирует детекцию линий на изображении"""

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")

        for test_file in pdf_files[:3]:
            logger.info(f"Тестирование на файле: {test_file.name}")
            pdf_bytes = test_file.read_bytes()
            with PDFLoader(pdf_bytes) as loader:
                original = loader.get_page_image(0, dpi=300)

            preprocessor_img = ImagePreprocessor()
            processed = preprocessor_img.process(original)

            img_paragraph_preprocessing = preprocessor._processing(processed)
            paragraph_mask = preprocessor._detect_paragraphs(
                img_paragraph_preprocessing
            )
            cv2.imwrite(
                str(output_dir / f"{test_file.stem}_raw_paragraph_mask.png"),
                paragraph_mask,
            )

        logger.success("Тест детекции линий пройден успешно")
