import pytest
from app.services.cache_in_disk import CacheInDisk


class TestCacheIntegration:
    """Интеграционные тесты для CacheInDisk"""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, sample_pdf_bytes, sample_structure):
        """Тест полного жизненного цикла процесса"""
        cache = CacheInDisk(expire=60)

        # 1. Создание процесса
        process_id = await cache.create_process(sample_pdf_bytes, sample_structure)
        assert process_id is not None

        # 2. Получение структуры
        structure = await cache.get_structure(process_id)
        assert structure == sample_structure

        # 3. Получение PDF
        pdf_bytes = await cache.get_pdf(process_id)
        assert pdf_bytes == sample_pdf_bytes

        # 4. Удаление процесса
        await cache.delete_process(process_id)

        # 5. Проверка удаления
        structure = await cache.get_structure(process_id)
        assert structure is None

    @pytest.mark.asyncio
    async def test_multiple_processes(
        self, cache_service, sample_pdf_bytes, sample_structure
    ):
        """Тест работы с несколькими процессами одновременно"""
        process_ids = []

        # Создаем 10 процессов
        for i in range(10):
            process_id = await cache_service.create_process(
                sample_pdf_bytes, {**sample_structure, "index": i}
            )
            process_ids.append(process_id)

        # Проверяем, что все процессы доступны
        for i, process_id in enumerate(process_ids):
            structure = await cache_service.get_structure(process_id)
            assert structure["index"] == i

        # Удаляем все процессы
        for process_id in process_ids:
            await cache_service.delete_process(process_id)

    @pytest.mark.asyncio
    async def test_persistence(self, sample_pdf_bytes, sample_structure):
        """Тест персистентности данных при пересоздании кэша"""
        # Создаем процесс
        cache1 = CacheInDisk(expire=3600)
        process_id = await cache1.create_process(sample_pdf_bytes, sample_structure)

        # Пересоздаем кэш (имитация перезапуска)
        cache2 = CacheInDisk(expire=3600)

        # Проверяем, что данные сохранились
        structure = await cache2.get_structure(process_id)
        assert structure == sample_structure

        pdf_bytes = await cache2.get_pdf(process_id)
        assert pdf_bytes == sample_pdf_bytes
