"""Реализует сценарий чтения состояния процесса."""

from __future__ import annotations

from app.application.dto.get_process_status import (
    GetProcessStatusCommand,
    GetProcessStatusResult,
)
from app.application.errors import ProcessNotFoundError
from app.application.ports.process_repository import ProcessRepository


class GetProcessStatusUseCase:
    """Возвращает сохранённое состояние процесса обработки."""

    def __init__(self, process_repository: ProcessRepository):
        self.process_repository = process_repository

    async def execute(
        self,
        command: GetProcessStatusCommand,
    ) -> GetProcessStatusResult:
        """Читает текущее состояние процесса.

        Args:
            command: Команда получения состояния процесса.

        Returns:
            GetProcessStatusResult: Текущее состояние процесса.

        Raises:
            ProcessNotFoundError: Если процесс не найден.
        """
        process_state = await self.process_repository.get(command.process_id)
        if process_state is None:
            raise ProcessNotFoundError(f"Процесс {command.process_id} не найден")

        return GetProcessStatusResult(process_state=process_state)
