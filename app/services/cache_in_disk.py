import uuid
from loguru import logger
from datetime import datetime, timezone
from diskcache import Cache
import asyncio


class ProcessData:
    """Структура для хранения данных процесса"""

    def __init__(self, pdf_bytes: bytes, structure: dict):
        self.pdf_bytes = pdf_bytes
        self.structure = structure
        self.created_at = datetime.now(timezone.utc)


class CacheInDisk:
    """Кэш для хранения данных процессов с автоматической очисткой по TTL"""

    def __init__(self, expire: int = 24 * 60 * 60, cache_dir: str = "./.cache"):
        self.cache = Cache(cache_dir)
        self.expire = expire

    async def cleanup_expired_cache(self):
        """Фоновая задача для очистки устаревших записей в кэше"""
        while True:
            await asyncio.sleep(3600)  # Проверка раз в час
            logger.info("Очистка устаревших записей кэша...")
            self.cache.expire()  # diskcache сам удаляет истёкшие записи
            logger.debug(f"Очистка завершена. Размер кэша: {len(self.cache)}")

    async def create_process(self, pdf_bytes: bytes, structure: dict) -> str:
        """Создает новый процесс и сохраняет его в кэш"""
        process_id = str(uuid.uuid4())
        data = ProcessData(pdf_bytes=pdf_bytes, structure=structure)
        self.cache.set(process_id, data, expire=self.expire)
        logger.info(f"Создан процесс {process_id}")
        return process_id

    async def get_structure(self, process_id: str) -> dict | None:
        """Получает структуру данных процесса по его идентификатору"""
        data = self.cache.get(process_id)
        return data.structure if isinstance(data, ProcessData) else None

    async def get_pdf(self, process_id: str) -> bytes | None:
        """Получает PDF-файл процесса по его идентификатору"""
        data = self.cache.get(process_id)
        return data.pdf_bytes if isinstance(data, ProcessData) else None

    async def delete_process(self, process_id: str):
        """Удаляет процесс из кэша"""
        deleted = self.cache.delete(process_id)
        if deleted:
            logger.info(f"Удален процесс {process_id}")
