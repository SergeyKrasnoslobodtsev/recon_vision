import asyncio
import shutil
import tempfile

import pytest

from app.domain.entities.process import ProcessState
from app.domain.enums.process_status import ProcessStatus
from app.infrastructure.persistence.models import ProcessData
from app.infrastructure.persistence.diskcache_process_repository import (
    DiskCacheProcessRepository,
)


class TestProcessData:
    """Проверяет модель хранения процесса."""

    def test_process_data_initialization(self, sample_pdf_bytes):
        """Создаёт ProcessData с доменным состоянием процесса."""
        process_state = ProcessState(source_pdf=sample_pdf_bytes)
        data = ProcessData(process_state=process_state, pdf_bytes=sample_pdf_bytes)

        assert data.process_state is process_state
        assert data.pdf_bytes == sample_pdf_bytes
        assert data.created_at is not None


class TestDiskCacheProcessRepository:
    """Проверяет репозиторий процессов на базе DiskCache."""

    @pytest.mark.asyncio
    async def test_add_assigns_process_id(self, cache_service, sample_pdf_bytes):
        """Сохраняет процесс и присваивает ему UUID."""
        process_state = ProcessState(source_pdf=sample_pdf_bytes)

        process_id = await cache_service.add(process_state)

        assert process_id
        assert isinstance(process_id, str)
        assert len(process_id) == 36
        assert process_state.process_id == process_id

    @pytest.mark.asyncio
    async def test_get_returns_saved_process_state(
        self,
        cache_service,
        sample_pdf_bytes,
        sample_reconciliation_data,
    ):
        """Возвращает ранее сохранённое состояние процесса."""
        process_state = ProcessState(source_pdf=sample_pdf_bytes)
        process_id = await cache_service.add(process_state)
        process_state.mark_completed(sample_reconciliation_data, message="done")
        await cache_service.update(process_state)

        restored = await cache_service.get(process_id)

        assert restored is not None
        assert restored.process_id == process_id
        assert restored.status == ProcessStatus.COMPLETED
        assert restored.source_pdf == sample_pdf_bytes
        assert restored.reconciliation_data is not None
        assert restored.reconciliation_data.seller == "Test Seller"

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unknown_process(self, cache_service):
        """Возвращает None для отсутствующего процесса."""
        restored = await cache_service.get("missing-process")

        assert restored is None

    @pytest.mark.asyncio
    async def test_update_requires_process_id(self, cache_service, sample_pdf_bytes):
        """Не обновляет процесс без идентификатора."""
        process_state = ProcessState(source_pdf=sample_pdf_bytes)

        with pytest.raises(ValueError, match="process_id"):
            await cache_service.update(process_state)

    @pytest.mark.asyncio
    async def test_delete_removes_saved_process(self, cache_service, sample_pdf_bytes):
        """Удаляет сохранённый процесс из репозитория."""
        process_state = ProcessState(source_pdf=sample_pdf_bytes)
        process_id = await cache_service.add(process_state)

        await cache_service.delete(process_id)

        restored = await cache_service.get(process_id)
        assert restored is None

    @pytest.mark.asyncio
    async def test_ttl_expiration_removes_process(self, sample_pdf_bytes):
        """Удаляет процесс после истечения TTL."""
        temp_dir = tempfile.mkdtemp(prefix="test_ttl_")
        repository = DiskCacheProcessRepository(expire=1, cache_dir=temp_dir)
        process_state = ProcessState(source_pdf=sample_pdf_bytes)
        process_id = await repository.add(process_state)

        await asyncio.sleep(2)
        repository.cache.expire()

        restored = await repository.get(process_id)
        assert restored is None

        repository.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
