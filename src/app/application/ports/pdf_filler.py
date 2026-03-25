"""Определяет порт заполнения PDF документа."""

from __future__ import annotations

from typing import Protocol

from app.application.dto.fill_reconciliation_act import FillReconciliationActCommand
from app.domain.entities.process import ProcessState


class PdfFiller(Protocol):
    """Описывает контракт заполнения PDF на основе сохранённого процесса."""

    async def fill(
        self,
        process_state: ProcessState,
        command: FillReconciliationActCommand,
    ) -> bytes:
        """Возвращает заполненный PDF в байтах."""
