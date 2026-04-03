import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

from app.domain.entities.reconciliation_data import ReconciliationData
from app.domain.value_objects.period import Period
from app.infrastructure.persistence.diskcache_process_repository import (
    DiskCacheProcessRepository,
)
from vision_core.detector.paragraph_detector import ParagraphDetector
from vision_core.detector.table_cell_detector import TableCellDetector
from vision_core.detector.table_detector import TableDetector
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    """Создаёт экземпляр репозитория процессов с уникальным временным кэшем."""
    # Создаем уникальную временную директорию для каждого теста
    temp_dir = tempfile.mkdtemp(prefix="test_cache_")
    cache_path = os.path.join(temp_dir, ".cache")

    cache = DiskCacheProcessRepository(expire=60, cache_dir=cache_path)

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


@pytest.fixture
def sample_reconciliation_data() -> ReconciliationData:
    """Возвращает тестовые доменные данные акта сверки."""
    return ReconciliationData(
        seller="Test Seller",
        buyer="Test Buyer",
        period=Period(start="2025-01-01", end="2025-01-31"),
        debit=[],
        credit=[],
        message="done",
    )


### Common pdf file and output directory fixtures


@pytest.fixture
def pdf_path() -> Path:
    """Путь к папке с тестовыми PDF"""
    return Path("./examples/trash")


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
            return loader.get_page_image(0, dpi=300)

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
