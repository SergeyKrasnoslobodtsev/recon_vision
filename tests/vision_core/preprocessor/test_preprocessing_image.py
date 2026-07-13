from pathlib import Path

from loguru import logger

from vision_core.debug_image_observer import DebugImageObserver
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor


class TestPreprocessingImage:
    def test_preprocessing_image(self, pdf_file: Path, output_dir: Path):
        observer = DebugImageObserver(output_dir=output_dir)

        with PDFLoader(pdf_file.read_bytes()) as loader:
            original = loader.get_page_image(0, dpi=300)

        ImagePreprocessor(debug_image=observer).process(original)
        logger.success(f"Препроцессинг пройден: {pdf_file.name}")
