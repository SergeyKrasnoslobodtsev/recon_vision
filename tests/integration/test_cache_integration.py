import pytest

from app.domain.entities.process import ProcessState
from app.domain.enums.process_status import ProcessStatus
from app.infrastructure.persistence.diskcache_process_repository import (
    DiskCacheProcessRepository,
)


class TestCacheIntegration:
    """Интеграционные тесты для репозитория процессов."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(
        self,
        sample_pdf_bytes,
        sample_reconciliation_data,
    ):
        """Проверяет полный жизненный цикл процесса в репозитории."""
        repository = DiskCacheProcessRepository(expire=60)
        process_state = ProcessState(source_pdf=sample_pdf_bytes)

        process_id = await repository.add(process_state)
        process_state.mark_completed(sample_reconciliation_data, message="done")
        await repository.update(process_state)

        restored = await repository.get(process_id)
        assert restored is not None
        assert restored.status == ProcessStatus.COMPLETED
        assert restored.reconciliation_data is not None
        assert restored.reconciliation_data.buyer == "Test Buyer"

        await repository.delete(process_id)

        restored = await repository.get(process_id)
        assert restored is None
        repository.close()

    @pytest.mark.asyncio
    async def test_multiple_processes(
        self, cache_service, sample_pdf_bytes
    ):
        """Проверяет работу с несколькими процессами одновременно."""
        process_ids = []

        for i in range(10):
            process_state = ProcessState(
                source_pdf=sample_pdf_bytes,
                metadata={"index": i},
            )
            process_id = await cache_service.add(process_state)
            process_ids.append(process_id)

        for i, process_id in enumerate(process_ids):
            restored = await cache_service.get(process_id)
            assert restored is not None
            assert restored.metadata["index"] == i

        for process_id in process_ids:
            await cache_service.delete(process_id)

    @pytest.mark.asyncio
    async def test_persistence(
        self,
        sample_pdf_bytes,
        sample_reconciliation_data,
        tmp_path,
    ):
        """Проверяет персистентность данных между экземплярами репозитория."""
        cache_dir = str(tmp_path / "cache")
        repository_1 = DiskCacheProcessRepository(expire=3600, cache_dir=cache_dir)
        process_state = ProcessState(source_pdf=sample_pdf_bytes)
        process_id = await repository_1.add(process_state)
        process_state.mark_completed(sample_reconciliation_data, message="done")
        await repository_1.update(process_state)
        repository_1.close()

        repository_2 = DiskCacheProcessRepository(expire=3600, cache_dir=cache_dir)
        restored = await repository_2.get(process_id)

        assert restored is not None
        assert restored.source_pdf == sample_pdf_bytes
        assert restored.reconciliation_data is not None
        assert restored.reconciliation_data.seller == "Test Seller"
        repository_2.close()
