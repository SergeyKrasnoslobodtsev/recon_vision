"""Реализует сценарий заполнения акта сверки."""

from __future__ import annotations

import base64

from app.application.dto.fill_reconciliation_act import (
    FillReconciliationActCommand,
    FillReconciliationActResult,
)
from app.application.errors import (
    ProcessFailedError,
    ProcessNotFoundError,
    ProcessNotReadyError,
)
from app.application.ports.pdf_filler import PdfFiller
from app.application.ports.process_repository import ProcessRepository
from app.domain.enums.process_status import ProcessStatus


class FillReconciliationActUseCase:
    """Оркестрирует заполнение PDF на основе сохранённого процесса."""

    def __init__(
        self,
        process_repository: ProcessRepository,
        pdf_filler: PdfFiller,
    ):
        self.process_repository = process_repository
        self.pdf_filler = pdf_filler

    async def execute(
        self,
        command: FillReconciliationActCommand,
    ) -> FillReconciliationActResult:
        """Заполняет PDF и возвращает его в base64.

        Args:
            command: Команда заполнения документа.

        Returns:
            FillReconciliationActResult: Заполненный документ в base64.

        Raises:
            ProcessNotFoundError: Если процесс не найден.
            ProcessNotReadyError: Если процесс ещё обрабатывается.
            ProcessFailedError: Если процесс завершился ошибкой.
            ValueError: Если в процессе отсутствует исходный PDF.
        """
        process_state = await self.process_repository.get(command.process_id)
        if process_state is None:
            raise ProcessNotFoundError(f"Процесс {command.process_id} не найден")

        if process_state.status in {ProcessStatus.RECEIVED, ProcessStatus.PROCESSING}:
            raise ProcessNotReadyError("Процесс ещё находится в обработке")

        if process_state.status == ProcessStatus.FAILED:
            raise ProcessFailedError(process_state.message or "Ошибка обработки")

        if not process_state.source_pdf:
            raise ValueError("В процессе отсутствует исходный PDF")

        filled_pdf = await self.pdf_filler.fill(process_state, command)
        process_state.mark_filled("Документ успешно заполнен")
        await self.process_repository.update(process_state)

        document_base64 = base64.b64encode(filled_pdf).decode("utf-8")
        return FillReconciliationActResult(document_base64=document_base64)
