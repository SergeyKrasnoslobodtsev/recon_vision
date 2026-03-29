"""Адаптирует vision_core pipeline к порту построения документа."""

from __future__ import annotations

from typing import Optional

from vision_core.pipelines.build_document import DocumentBuildPipeline


class VisionDocumentBuilder:
    """Строит канонический Document через vision_core.

    Инициализация pipeline выполняется лениво, чтобы не загружать OCR-движок
    на старте приложения без необходимости.
    """

    def __init__(self):
        self._pipeline: Optional[DocumentBuildPipeline] = None

    async def build(self, pdf_bytes: bytes):
        """Строит канонический документ из PDF.

        Args:
            pdf_bytes: Исходный PDF-файл в байтах.

        Returns:
            Document: Каноническое представление документа.
        """
        pipeline = self._get_pipeline()
        return pipeline.build(pdf_bytes)

    def _get_pipeline(self) -> DocumentBuildPipeline:
        """Возвращает и при необходимости создаёт pipeline построения документа."""
        if self._pipeline is None:
            self._pipeline = DocumentBuildPipeline()
        return self._pipeline
