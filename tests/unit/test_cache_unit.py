import pytest
import asyncio
from app.services.cache_in_disk import CacheInDisk, ProcessData


class TestProcessData:
    """Тесты для класса ProcessData"""

    def test_process_data_initialization(self, sample_pdf_bytes, sample_structure):
        """Тест создания ProcessData"""
        data = ProcessData(pdf_bytes=sample_pdf_bytes, structure=sample_structure)

        assert data.pdf_bytes == sample_pdf_bytes
        assert data.structure == sample_structure
        assert data.created_at is not None


class TestCacheInMemory:
    """Юнит-тесты для CacheInMemory"""

    @pytest.mark.asyncio
    async def test_create_process(
        self, cache_service, sample_pdf_bytes, sample_structure
    ):
        """Тест создания процесса"""
        process_id = await cache_service.create_process(
            sample_pdf_bytes, sample_structure
        )

        assert process_id is not None
        assert isinstance(process_id, str)
        assert len(process_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_get_structure(
        self, cache_service, sample_pdf_bytes, sample_structure
    ):
        """Тест получения структуры процесса"""
        process_id = await cache_service.create_process(
            sample_pdf_bytes, sample_structure
        )
        structure = await cache_service.get_structure(process_id)

        assert structure == sample_structure

    @pytest.mark.asyncio
    async def test_get_structure_not_found(self, cache_service):
        """Тест получения структуры несуществующего процесса"""
        structure = await cache_service.get_structure("non-existent-id")

        assert structure is None

    @pytest.mark.asyncio
    async def test_get_pdf(self, cache_service, sample_pdf_bytes, sample_structure):
        """Тест получения PDF процесса"""
        process_id = await cache_service.create_process(
            sample_pdf_bytes, sample_structure
        )
        pdf_bytes = await cache_service.get_pdf(process_id)

        assert pdf_bytes == sample_pdf_bytes

    @pytest.mark.asyncio
    async def test_get_pdf_not_found(self, cache_service):
        """Тест получения PDF несуществующего процесса"""
        pdf_bytes = await cache_service.get_pdf("non-existent-id")

        assert pdf_bytes is None

    @pytest.mark.asyncio
    async def test_delete_process(
        self, cache_service, sample_pdf_bytes, sample_structure
    ):
        """Тест удаления процесса"""
        process_id = await cache_service.create_process(
            sample_pdf_bytes, sample_structure
        )

        # Проверяем, что процесс существует
        structure = await cache_service.get_structure(process_id)
        assert structure is not None

        # Удаляем процесс
        await cache_service.delete_process(process_id)

        # Проверяем, что процесс удален
        structure = await cache_service.get_structure(process_id)
        assert structure is None

    @pytest.mark.asyncio
    async def test_delete_non_existent_process(self, cache_service):
        """Тест удаления несуществующего процесса"""
        # Не должно вызывать ошибку
        await cache_service.delete_process("non-existent-id")

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, sample_pdf_bytes, sample_structure):
        """Тест автоматического удаления по TTL"""
        import tempfile

        temp_dir = tempfile.mkdtemp(prefix="test_ttl_")

        cache = CacheInDisk(expire=1, cache_dir=temp_dir)
        process_id = await cache.create_process(sample_pdf_bytes, sample_structure)

        # Проверяем, что процесс существует
        structure = await cache.get_structure(process_id)
        assert structure is not None

        # Ждем истечения TTL
        await asyncio.sleep(2)

        # Вызываем expire для принудительной очистки
        cache.cache.expire()

        # Проверяем, что процесс удален
        structure = await cache.get_structure(process_id)
        assert structure is None

        # Очистка
        cache.cache.close()
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_cleanup_expired_cache(self, sample_pdf_bytes, sample_structure):
        """Тест фоновой задачи очистки кэша"""
        cache = CacheInDisk(expire=1)
        process_id = await cache.create_process(sample_pdf_bytes, sample_structure)

        # Запускаем задачу очистки на короткое время
        task = asyncio.create_task(cache.cleanup_expired_cache())

        await asyncio.sleep(2)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Проверяем, что устаревшие записи очищены
        structure = await cache.get_structure(process_id)
        assert structure is None
