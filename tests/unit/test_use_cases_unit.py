import base64
from unittest.mock import AsyncMock

import pytest

from app.application.dto.fill_reconciliation_act import FillReconciliationActCommand
from app.application.dto.get_process_status import GetProcessStatusCommand
from app.application.dto.submit_reconciliation_act import SubmitReconciliationActCommand
from app.application.errors import (
    ProcessFailedError,
    ProcessNotFoundError,
    ProcessNotReadyError,
)
from app.application.use_cases.fill_reconciliation_act import (
    FillReconciliationActUseCase,
)
from app.application.use_cases.get_process_status import GetProcessStatusUseCase
from app.application.use_cases.submit_reconciliation_act import (
    SubmitReconciliationActUseCase,
)
from app.application.dto.semantic_input import SemanticInput
from app.domain.entities.process import ProcessState
from app.domain.enums.process_status import ProcessStatus


class TestSubmitReconciliationActUseCase:
    @pytest.mark.asyncio
    async def test_execute_creates_process_and_completes_it(
        self,
        sample_pdf_bytes,
        sample_reconciliation_data,
    ):
        document_payload = {"document": "payload"}
        semantic_input = SemanticInput(linearized_text="payload")
        process_repository = AsyncMock()
        process_repository.add.return_value = "process-123"
        document_builder = AsyncMock()
        document_builder.build.return_value = document_payload
        semantic_input_projector = AsyncMock()
        semantic_input_projector.build.return_value = semantic_input
        structured_data_extractor = AsyncMock()
        structured_data_extractor.extract.return_value = sample_reconciliation_data
        use_case = SubmitReconciliationActUseCase(
            process_repository=process_repository,
            document_builder=document_builder,
            semantic_input_projector=semantic_input_projector,
            structured_data_extractor=structured_data_extractor,
        )
        command = SubmitReconciliationActCommand(
            document_base64=base64.b64encode(sample_pdf_bytes).decode("utf-8")
        )

        result = await use_case.execute(command)

        assert result.process_id == "process-123"
        process_repository.add.assert_awaited_once()
        stored_process = process_repository.add.await_args.args[0]
        assert isinstance(stored_process, ProcessState)
        assert stored_process.source_pdf == sample_pdf_bytes
        document_builder.build.assert_awaited_once_with(sample_pdf_bytes)
        semantic_input_projector.build.assert_awaited_once_with(document_payload)
        structured_data_extractor.extract.assert_awaited_once_with(semantic_input)
        assert process_repository.update.await_count == 2

        processing_state = process_repository.update.await_args_list[0].args[0]
        assert processing_state.process_id == "process-123"
        assert processing_state.status == ProcessStatus.COMPLETED
        assert processing_state.message == "Документ успешно обработан"
        assert processing_state.document_payload == document_payload
        assert processing_state.reconciliation_data == sample_reconciliation_data

        completed_state = process_repository.update.await_args_list[1].args[0]
        assert completed_state is processing_state

    @pytest.mark.asyncio
    async def test_execute_raises_on_invalid_base64(self):
        process_repository = AsyncMock()
        document_builder = AsyncMock()
        semantic_input_projector = AsyncMock()
        structured_data_extractor = AsyncMock()
        use_case = SubmitReconciliationActUseCase(
            process_repository=process_repository,
            document_builder=document_builder,
            semantic_input_projector=semantic_input_projector,
            structured_data_extractor=structured_data_extractor,
        )

        with pytest.raises(ValueError, match="Не удалось декодировать PDF из base64"):
            await use_case.execute(
                SubmitReconciliationActCommand(document_base64="!!!")
            )

        process_repository.add.assert_not_called()
        process_repository.update.assert_not_called()
        document_builder.build.assert_not_called()
        semantic_input_projector.project.assert_not_called()
        structured_data_extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_marks_process_failed_when_builder_crashes(
        self,
        sample_pdf_bytes,
    ):
        process_repository = AsyncMock()
        process_repository.add.return_value = "process-123"
        document_builder = AsyncMock()
        document_builder.build.side_effect = RuntimeError("builder crashed")
        semantic_input_projector = AsyncMock()
        structured_data_extractor = AsyncMock()
        use_case = SubmitReconciliationActUseCase(
            process_repository=process_repository,
            document_builder=document_builder,
            semantic_input_projector=semantic_input_projector,
            structured_data_extractor=structured_data_extractor,
        )
        command = SubmitReconciliationActCommand(
            document_base64=base64.b64encode(sample_pdf_bytes).decode("utf-8")
        )

        with pytest.raises(RuntimeError, match="builder crashed"):
            await use_case.execute(command)

        assert process_repository.update.await_count == 2
        failed_state = process_repository.update.await_args_list[-1].args[0]
        assert failed_state.status == ProcessStatus.FAILED
        assert failed_state.message == "builder crashed"
        semantic_input_projector.project.assert_not_called()
        structured_data_extractor.extract.assert_not_called()


class TestGetProcessStatusUseCase:
    @pytest.mark.asyncio
    async def test_execute_returns_process_state(self):
        process_state = ProcessState(process_id="process-123")
        process_repository = AsyncMock()
        process_repository.get.return_value = process_state
        use_case = GetProcessStatusUseCase(process_repository=process_repository)

        result = await use_case.execute(
            GetProcessStatusCommand(process_id="process-123")
        )

        assert result.process_state is process_state
        process_repository.get.assert_awaited_once_with("process-123")

    @pytest.mark.asyncio
    async def test_execute_raises_when_process_not_found(self):
        process_repository = AsyncMock()
        process_repository.get.return_value = None
        use_case = GetProcessStatusUseCase(process_repository=process_repository)

        with pytest.raises(ProcessNotFoundError, match="process-123"):
            await use_case.execute(GetProcessStatusCommand(process_id="process-123"))


class TestFillReconciliationActUseCase:
    @pytest.mark.asyncio
    async def test_execute_returns_filled_document_in_base64(self, sample_pdf_bytes):
        process_state = ProcessState(
            process_id="process-123",
            status=ProcessStatus.COMPLETED,
            source_pdf=sample_pdf_bytes,
        )
        filled_pdf = b"%PDF-1.4 filled content"
        process_repository = AsyncMock()
        process_repository.get.return_value = process_state
        pdf_filler = AsyncMock()
        pdf_filler.fill.return_value = filled_pdf
        use_case = FillReconciliationActUseCase(
            process_repository=process_repository,
            pdf_filler=pdf_filler,
        )
        command = FillReconciliationActCommand(process_id="process-123")

        result = await use_case.execute(command)

        pdf_filler.fill.assert_awaited_once_with(process_state, command)
        process_repository.update.assert_awaited_once_with(process_state)
        assert process_state.status == ProcessStatus.FILLED
        assert process_state.message == "Документ успешно заполнен"
        assert result.document_base64 == base64.b64encode(filled_pdf).decode("utf-8")

    @pytest.mark.asyncio
    async def test_execute_raises_when_process_not_found(self):
        process_repository = AsyncMock()
        process_repository.get.return_value = None
        pdf_filler = AsyncMock()
        use_case = FillReconciliationActUseCase(
            process_repository=process_repository,
            pdf_filler=pdf_filler,
        )

        with pytest.raises(ProcessNotFoundError, match="process-123"):
            await use_case.execute(
                FillReconciliationActCommand(process_id="process-123")
            )

        pdf_filler.fill.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [ProcessStatus.RECEIVED, ProcessStatus.PROCESSING],
    )
    async def test_execute_raises_when_process_is_not_ready(self, status):
        process_repository = AsyncMock()
        process_repository.get.return_value = ProcessState(
            process_id="process-123",
            status=status,
            source_pdf=b"pdf",
        )
        pdf_filler = AsyncMock()
        use_case = FillReconciliationActUseCase(
            process_repository=process_repository,
            pdf_filler=pdf_filler,
        )

        with pytest.raises(ProcessNotReadyError, match="ещё находится в обработке"):
            await use_case.execute(
                FillReconciliationActCommand(process_id="process-123")
            )

        pdf_filler.fill.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_raises_when_process_failed(self):
        process_repository = AsyncMock()
        process_repository.get.return_value = ProcessState(
            process_id="process-123",
            status=ProcessStatus.FAILED,
            source_pdf=b"pdf",
            message="semantic extraction failed",
        )
        pdf_filler = AsyncMock()
        use_case = FillReconciliationActUseCase(
            process_repository=process_repository,
            pdf_filler=pdf_filler,
        )

        with pytest.raises(ProcessFailedError, match="semantic extraction failed"):
            await use_case.execute(
                FillReconciliationActCommand(process_id="process-123")
            )

        pdf_filler.fill.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_raises_when_source_pdf_is_missing(self):
        process_repository = AsyncMock()
        process_repository.get.return_value = ProcessState(
            process_id="process-123",
            status=ProcessStatus.COMPLETED,
            source_pdf=b"",
        )
        pdf_filler = AsyncMock()
        use_case = FillReconciliationActUseCase(
            process_repository=process_repository,
            pdf_filler=pdf_filler,
        )

        with pytest.raises(ValueError, match="отсутствует исходный PDF"):
            await use_case.execute(
                FillReconciliationActCommand(process_id="process-123")
            )

        pdf_filler.fill.assert_not_called()
