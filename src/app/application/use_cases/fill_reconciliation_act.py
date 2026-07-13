"""Реализует сценарий заполнения акта сверки."""

from __future__ import annotations

import base64
from collections import defaultdict, deque
from dataclasses import replace

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
from app.domain.entities.ledger_entry import LedgerEntry, RowReference
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

        enriched_command = FillReconciliationActCommand(
            process_id=command.process_id,
            comments=command.comments,
            debit=self._restore_internal_references(
                command.debit,
                process_state.reconciliation_data.debit if process_state.reconciliation_data else [],
            ),
            credit=self._restore_internal_references(
                command.credit,
                process_state.reconciliation_data.credit if process_state.reconciliation_data else [],
            ),
        )

        filled_pdf = await self.pdf_filler.fill(process_state, enriched_command)
        process_state.mark_filled("Документ успешно заполнен")
        await self.process_repository.update(process_state)

        document_base64 = base64.b64encode(filled_pdf).decode("utf-8")
        return FillReconciliationActResult(document_base64=document_base64)

    def _restore_internal_references(
        self,
        entries: list[LedgerEntry],
        stored_entries: list[LedgerEntry],
    ) -> list[LedgerEntry]:
        """Восстанавливает служебную привязку к колонкам по сохранённым данным процесса.

        Args:
            entries: Записи, пришедшие из внешнего API.
            stored_entries: Исходные извлечённые записи, сохранённые внутри сервиса.

        Returns:
            list[LedgerEntry]: Записи с восстановленными internal row_reference.
        """
        exact_index: dict[tuple[str, str, str, str | None], deque[LedgerEntry]] = defaultdict(deque)
        row_index: dict[tuple[str, str], deque[LedgerEntry]] = defaultdict(deque)

        for stored_entry in stored_entries:
            row_reference = stored_entry.row_reference
            if row_reference is None:
                continue
            exact_index[(row_reference.id_table, row_reference.id_row, stored_entry.record, stored_entry.date)].append(
                stored_entry
            )
            row_index[(row_reference.id_table, row_reference.id_row)].append(stored_entry)

        restored_entries: list[LedgerEntry] = []
        for entry in entries:
            row_reference = entry.row_reference
            if row_reference is None:
                restored_entries.append(entry)
                continue

            exact_key = (row_reference.id_table, row_reference.id_row, entry.record, entry.date)
            row_key = (row_reference.id_table, row_reference.id_row)
            stored_entry = None

            if exact_index[exact_key]:
                stored_entry = exact_index[exact_key].popleft()
            elif row_index[row_key]:
                stored_entry = row_index[row_key].popleft()

            if stored_entry is None or stored_entry.row_reference is None:
                restored_entries.append(entry)
                continue

            restored_entries.append(
                replace(
                    entry,
                    row_reference=RowReference(
                        id_table=row_reference.id_table,
                        id_row=row_reference.id_row,
                        id_col=stored_entry.row_reference.id_col,
                        buyer_col=stored_entry.row_reference.buyer_col,
                    ),
                )
            )

        return restored_entries
