import sys
import pytest
import shutil
import os
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from _pytest.logging import LogCaptureFixture
from loguru import logger


from app.services.cache_in_disk import CacheInDisk

from vision_core.entities.bbox import BBox
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.paragraph_preprocessor import ParagraphPreprocessor
from vision_core.detector.table_detector import TableDetector
from vision_core.detector.table_cell_detector import TableCellDetector
from vision_core.detector.paragraph_detector import ParagraphDetector

### Logger configuration

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


@pytest.fixture(autouse=True)
def configure_logger(caplog: LogCaptureFixture):
    """Фикстура для настройки логгера loguru для тестов"""
    logger.remove()
    logger.add(sys.stdout, level="DEBUG", colorize=True)


### Service fixtures


@pytest.fixture
def cache_service():
    """Фикстура для создания экземпляра CacheInDisk с уникальным временным кэшем"""
    # Создаем уникальную временную директорию для каждого теста
    temp_dir = tempfile.mkdtemp(prefix="test_cache_")
    cache_path = os.path.join(temp_dir, ".cache")

    cache = CacheInDisk(expire=60, cache_dir=cache_path)

    yield cache

    # Очистка после теста
    try:
        cache.cache.close()  # Закрываем соединение с БД
    except Exception:
        pass

    # Удаляем временную директорию
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # Если не удалось удалить сразу, пробуем позже
            import time

            time.sleep(0.1)
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


@pytest.fixture
def sample_pdf_bytes():
    """Фикстура для тестовых PDF-данных"""
    return b"%PDF-1.4 test content"


@pytest.fixture
def sample_structure():
    """Фикстура для тестовой структуры данных"""
    return {
        "seller": "Test Seller",
        "buyer": "Test Buyer",
        "period": {"start": "2025-01-01", "end": "2025-01-31"},
    }


### Common pdf file and output directory fixtures


@pytest.fixture
def pdf_path() -> Path:
    """Путь к папке с тестовыми PDF"""
    return Path("./examples/test")


@pytest.fixture
def output_dir() -> Path:
    """Путь к папке для сохранения результатов"""
    output_dir = Path("./examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def pdf_loader_single_page():
    """Фикстура-фабрика для получения изображения первой страницы из PDF-байтов"""
    from vision_core.loader.pdf_loader import PDFLoader

    def _loader(pdf_bytes):
        with PDFLoader(pdf_bytes) as loader:
            return loader.get_page_image(0, dpi=200)

    return _loader


### Preprocessors


@pytest.fixture
def preprocessor_img() -> ImagePreprocessor:
    """Экземпляр ImagePreprocessor"""
    return ImagePreprocessor()


@pytest.fixture
def preprocessor_table() -> TablePreprocessor:
    """Экземпляр TablePreprocessor"""
    return TablePreprocessor()


@pytest.fixture
def preprocessor_paragraph() -> ParagraphPreprocessor:
    """Экземпляр ParagraphPreprocessor"""
    return ParagraphPreprocessor()


### Detectors


@pytest.fixture
def table_detector() -> TableDetector:
    """Экземпляр TableDetector"""
    return TableDetector()


@pytest.fixture
def cell_detector() -> TableCellDetector:
    """Экземпляр TableCellDetector"""
    return TableCellDetector()


@pytest.fixture
def paragraph_detector() -> ParagraphDetector:
    """Экземпляр ParagraphDetector"""
    return ParagraphDetector()


### Utility fixtures


@pytest.fixture
def drawer_bbox_and_label():
    """Функция для отрисовки BBox и метки на изображении"""

    def _drawer_bbox_and_label(
        image: np.ndarray,
        bbox: BBox,
        label: str,
        color,
        position: str = "top",  # top or bottom
    ):
        """Отрисовывает BBox и метку на изображении"""
        # rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image)
        else:
            pil = image

        new_image = Image.new(pil.mode, pil.size, (255, 255, 255))

        draw = ImageDraw.Draw(new_image)
        fnt = ImageFont.load_default(size=24)
        pt1 = (int(bbox.x_min), int(bbox.y_min))
        pt2 = (int(bbox.x_max), int(bbox.y_max))
        draw.rectangle([pt1, pt2], outline=color, width=2)

        def _label(coords, label, font, color):
            bbox = draw.textbbox(coords, label, font=font)
            padded_bbox = (bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5)
            draw.rectangle(padded_bbox, fill=color)
            draw.text(coords, label, font=font, fill=(255, 255, 255, 255))

        if position == "top":
            coords = (pt1[0] + 5, pt1[1] - 30)
            _label(coords, label, fnt, color)

        else:
            coords = (pt1[0] + 5, pt2[1] - 30)
            _label(coords, label, fnt, color)

        combinesd_image = Image.new("RGB", (pil.width * 2, pil.height))
        combinesd_image.paste(pil, (0, 0))
        combinesd_image.paste(new_image, (pil.width, 0))

        return combinesd_image

    return _drawer_bbox_and_label
