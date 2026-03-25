"""Собирает зависимости приложения в единый контейнер."""

from __future__ import annotations

from dataclasses import dataclass

from app.application.ports.process_repository import ProcessRepository
from app.application.use_cases.fill_reconciliation_act import (
    FillReconciliationActUseCase,
)
from app.application.use_cases.get_process_status import GetProcessStatusUseCase
from app.application.use_cases.submit_reconciliation_act import (
    SubmitReconciliationActUseCase,
)
from app.infrastructure.services.document_builder import VisionDocumentBuilder
from app.infrastructure.services.pdf_filler import DocumentPdfFiller
from app.infrastructure.services.structured_data_extractor import (
    StubStructuredDataExtractor,
)


@dataclass(slots=True)
class ApplicationContainer:
    """Хранит зависимости прикладного слоя приложения."""

    process_repository: ProcessRepository
    submit_reconciliation_act: SubmitReconciliationActUseCase
    get_process_status: GetProcessStatusUseCase
    fill_reconciliation_act: FillReconciliationActUseCase


def create_container(process_repository: ProcessRepository) -> ApplicationContainer:
    """Создаёт контейнер зависимостей приложения.

    Args:
        process_repository: Репозиторий хранения состояний процессов.

    Returns:
        ApplicationContainer: Собранный контейнер зависимостей.
    """
    document_builder = VisionDocumentBuilder()
    structured_data_extractor = StubStructuredDataExtractor()
    pdf_filler = DocumentPdfFiller()

    return ApplicationContainer(
        process_repository=process_repository,
        submit_reconciliation_act=SubmitReconciliationActUseCase(
            process_repository=process_repository,
            document_builder=document_builder,
            structured_data_extractor=structured_data_extractor,
        ),
        get_process_status=GetProcessStatusUseCase(
            process_repository=process_repository,
        ),
        fill_reconciliation_act=FillReconciliationActUseCase(
            process_repository=process_repository,
            pdf_filler=pdf_filler,
        ),
    )
