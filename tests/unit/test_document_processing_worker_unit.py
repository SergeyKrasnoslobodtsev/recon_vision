import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from app.domain.entities.process import ProcessState
from app.domain.enums.process_status import ProcessStatus
from app.infrastructure.services import document_processing_worker as worker_module
from app.infrastructure.services.document_processing_worker import (
    InfrastructureDocumentProcessingWorker,
)


class TestInfrastructureDocumentProcessingWorker:
    """Проверяет инфраструктурный запуск фоновой обработки."""

    @pytest.mark.asyncio
    async def test_start_processes_document_and_marks_process_completed(
        self,
        sample_pdf_bytes,
        sample_reconciliation_data,
    ):
        process_state = ProcessState(
            process_id="process-123",
            source_pdf=sample_pdf_bytes,
        )
        process_repository = AsyncMock()
        process_repository.get.return_value = process_state
        document_payload = {"document": "payload"}
        document_builder = AsyncMock()
        document_builder.build.return_value = document_payload
        structured_data_extractor = AsyncMock()
        structured_data_extractor.extract.return_value = sample_reconciliation_data
        worker = InfrastructureDocumentProcessingWorker(
            process_repository=process_repository,
            document_builder=document_builder,
            structured_data_extractor=structured_data_extractor,
        )

        await worker.start("process-123")
        await asyncio.sleep(0)

        process_repository.get.assert_awaited_once_with("process-123")
        document_builder.build.assert_awaited_once_with(sample_pdf_bytes)
        structured_data_extractor.extract.assert_awaited_once_with(document_payload)
        process_repository.update.assert_awaited_once_with(process_state)
        assert process_state.status == ProcessStatus.COMPLETED
        assert process_state.document_payload == document_payload
        assert process_state.reconciliation_data == sample_reconciliation_data

    @pytest.mark.asyncio
    async def test_start_marks_process_failed_when_builder_crashes(
        self,
        sample_pdf_bytes,
        monkeypatch,
    ):
        process_state = ProcessState(
            process_id="process-123",
            source_pdf=sample_pdf_bytes,
        )
        process_repository = AsyncMock()
        process_repository.get.return_value = process_state
        document_builder = AsyncMock()
        document_builder.build.side_effect = RuntimeError("builder crashed")
        structured_data_extractor = AsyncMock()
        worker = InfrastructureDocumentProcessingWorker(
            process_repository=process_repository,
            document_builder=document_builder,
            structured_data_extractor=structured_data_extractor,
        )
        logger_exception = Mock()
        monkeypatch.setattr(worker_module.logger, "exception", logger_exception)

        await worker.start("process-123")
        await asyncio.sleep(0)

        process_repository.update.assert_awaited_once_with(process_state)
        assert process_state.status == ProcessStatus.FAILED
        assert process_state.message == "builder crashed"
        logger_exception.assert_called_once()
