"""Определяет порт запуска фоновой обработки документа."""

from __future__ import annotations

from typing import Protocol


class DocumentProcessingWorker(Protocol):
    """Описывает контракт запуска фоновой обработки процесса."""

    async def start(self, process_id: str) -> None:
        """Запускает обработку сохранённого процесса по идентификатору."""
