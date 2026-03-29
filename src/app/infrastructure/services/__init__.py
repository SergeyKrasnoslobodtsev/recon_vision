"""Содержит активные инфраструктурные сервисы приложения."""

from app.infrastructure.services.document_builder import VisionDocumentBuilder
from app.infrastructure.services.pdf_filler import DocumentPdfFiller
from app.infrastructure.services.semantic_input_projector import (
    DocumentSemanticInputProjector,
)
from app.infrastructure.services.structured_data_extractor import (
    StubStructuredDataExtractor,
)

__all__ = [
    "DocumentSemanticInputProjector",
    "DocumentPdfFiller",
    "StubStructuredDataExtractor",
    "VisionDocumentBuilder",
]
