"""Запускает фоновую обработку документа через инфраструктурные сервисы."""

from __future__ import annotations

import asyncio

from loguru import logger

from app.application.ports.process_repository import ProcessRepository
from app.infrastructure.services.document_builder import VisionDocumentBuilder
from app.infrastructure.services.structured_data_extractor import (
    ReconciliationActExtractor,
)


class InfrastructureDocumentProcessingWorker:
    """Выполняет OCR и извлечение данных вне основного request path.

    Worker инкапсулирует вызовы `VisionDocumentBuilder` и
    `ReconciliationActExtractor`, обновляя состояние процесса в репозитории после
    завершения обработки или ошибки.

    Args:
        process_repository: Репозиторий состояний процессов.
        document_builder: Сервис построения канонического документа.
        structured_data_extractor: Сервис извлечения данных акта сверки.
    """

    def __init__(
        self,
        process_repository: ProcessRepository,
        document_builder: VisionDocumentBuilder | None = None,
        structured_data_extractor: ReconciliationActExtractor | None = None,
    ) -> None:
        self.process_repository = process_repository
        self.document_builder = document_builder or VisionDocumentBuilder()
        self.structured_data_extractor = structured_data_extractor or ReconciliationActExtractor()
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self, process_id: str) -> None:
        """Планирует фоновую обработку процесса и сразу возвращает управление.

        Args:
            process_id: Идентификатор сохранённого процесса.
        """
        task = asyncio.create_task(
            self._process(process_id),
            name=f"document-processing-{process_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _process(self, process_id: str) -> None:
        """Выполняет полную обработку сохранённого процесса.

        Args:
            process_id: Идентификатор процесса для обработки.
        """
        process_state = await self.process_repository.get(process_id)
        if process_state is None:
            logger.error(f"Процесс {process_id} не найден для фоновой обработки")
            return

        if not process_state.source_pdf:
            process_state.mark_failed("В процессе отсутствует исходный PDF")
            await self.process_repository.update(process_state)
            return

        try:
            document_payload = await self.document_builder.build(process_state.source_pdf)
            reconciliation_data = await self.structured_data_extractor.extract(document_payload)
            process_state.document_payload = document_payload
            process_state.mark_completed(
                reconciliation_data,
                message="Документ успешно обработан",
            )
            await self.process_repository.update(process_state)
        except Exception as exc:
            logger.exception(f"Фоновая обработка процесса {process_id} завершилась ошибкой")
            process_state.mark_failed(str(exc))
            await self.process_repository.update(process_state)
