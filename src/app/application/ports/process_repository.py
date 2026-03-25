"""Определяет порт хранилища процессов."""

from __future__ import annotations

from typing import Protocol

from app.domain.entities.process import ProcessState


class ProcessRepository(Protocol):
    """Описывает контракт хранилища состояния процессов."""

    async def add(self, process_state: ProcessState) -> str:
        """Сохраняет новый процесс и возвращает его идентификатор."""

    async def get(self, process_id: str) -> ProcessState | None:
        """Возвращает состояние процесса по идентификатору."""

    async def update(self, process_state: ProcessState) -> None:
        """Обновляет сохранённое состояние процесса."""

    async def delete(self, process_id: str) -> None:
        """Удаляет процесс по идентификатору."""
