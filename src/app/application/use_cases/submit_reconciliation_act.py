"""Реализует сценарий отправки акта сверки на обработку."""

from __future__ import annotations

import base64

from app.application.dto.submit_reconciliation_act import (
    SubmitReconciliationActCommand,
    SubmitReconciliationActResult,
)
from app.application.ports.document_builder import DocumentBuilder
from app.application.ports.process_repository import ProcessRepository
from app.application.ports.semantic_input_projector import SemanticInputProjector
from app.application.ports.structured_data_extractor import StructuredDataExtractor
from app.domain.entities.process import ProcessState


class SubmitReconciliationActUseCase:
    """Оркестрирует создание процесса обработки акта сверки."""

    def __init__(
        self,
        process_repository: ProcessRepository,
        document_builder: DocumentBuilder,
        semantic_input_projector: SemanticInputProjector,
        structured_data_extractor: StructuredDataExtractor,
    ):
        self.process_repository = process_repository
        self.document_builder = document_builder
        self.semantic_input_projector = semantic_input_projector
        self.structured_data_extractor = structured_data_extractor

    async def execute(
        self,
        command: SubmitReconciliationActCommand,
    ) -> SubmitReconciliationActResult:
        """Создаёт процесс, строит документ и извлекает данные.

        Args:
            command: Команда отправки PDF на обработку.

        Returns:
            SubmitReconciliationActResult: Идентификатор созданного процесса.

        Raises:
            ValueError: Если base64 документа некорректен.
        """
        try:
            pdf_bytes = base64.b64decode(command.document_base64, validate=True)
        except Exception as exc:
            raise ValueError("Не удалось декодировать PDF из base64") from exc

        process_state = ProcessState(source_pdf=pdf_bytes)
        process_id = await self.process_repository.add(process_state)
        process_state.process_id = process_id
        process_state.mark_processing("Документ принят в обработку")
        await self.process_repository.update(process_state)

        try:
            document_payload = await self.document_builder.build(pdf_bytes)
            semantic_input = await self.semantic_input_projector.build(document_payload)
            reconciliation_data = await self.structured_data_extractor.extract(
                semantic_input
            )

            process_state.document_payload = document_payload
            process_state.mark_completed(
                reconciliation_data,
                message="Документ успешно обработан",
            )
            await self.process_repository.update(process_state)
        except Exception as exc:
            process_state.mark_failed(str(exc))
            await self.process_repository.update(process_state)
            raise

        return SubmitReconciliationActResult(process_id=process_id)
