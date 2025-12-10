import sys
import pytest
import shutil
import os
import tempfile
from app.services.cache_in_disk import CacheInDisk
from _pytest.logging import LogCaptureFixture
from loguru import logger

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# coloring for loguru in pytest output
@pytest.fixture(autouse=True)
def configure_logger(caplog: LogCaptureFixture):
    """Фикстура для настройки логгера loguru для тестов"""
    logger.remove()
    logger.add(sys.stdout, level="DEBUG", colorize=True)


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
