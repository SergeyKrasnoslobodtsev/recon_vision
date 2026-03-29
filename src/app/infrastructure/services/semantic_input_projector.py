"""Проецирует канонический документ в входной контракт semantic analysis."""

from __future__ import annotations

from app.application.dto.semantic_input import (
    SemanticCell,
    SemanticInput,
    SemanticParagraph,
    SemanticTable,
)
from vision_core.entities.document import Document


class DocumentSemanticInputProjector:
    """Строит SemanticInput из канонического Document.

    Проектор отделяет semantic layer от внутренней структуры OCR/layout-модели,
    сохраняя при этом два смысловых канала: paragraphs и tables.
    """

    async def build(self, document_payload: Document) -> SemanticInput:
        """Преобразует Document в SemanticInput.

        Args:
            document_payload: Канонический документ, построенный в vision_core.

        Returns:
            SemanticInput: Нормализованный вход для semantic analysis.

        Raises:
            TypeError: Если вместо Document передан объект другого типа.
        """
        if not isinstance(document_payload, Document):
            raise TypeError("SemanticInputProjector ожидает канонический Document")

        paragraphs: list[SemanticParagraph] = []
        tables: list[SemanticTable] = []

        for page in document_payload.pages:
            for paragraph in page.paragraphs:
                if not paragraph.text.strip():
                    continue
                paragraphs.append(
                    SemanticParagraph(
                        id=paragraph.id,
                        page_number=page.page_number,
                        text=paragraph.text,
                        block_type=paragraph.type.name.lower(),
                        bbox=paragraph.bbox.to_tuple(),
                    )
                )

            for table in page.tables:
                cells: list[SemanticCell] = []
                for cell in table.cells:
                    cells.append(
                        SemanticCell(
                            row=cell.row,
                            col=cell.col,
                            text=(cell.value or "").strip(),
                            rowspan=cell.rowspan,
                            colspan=cell.colspan,
                        )
                    )
                tables.append(
                    SemanticTable(
                        id=table.id,
                        page_number=page.page_number,
                        cells=cells,
                    )
                )

        return SemanticInput(
            paragraphs=paragraphs,
            tables=tables,
            linearized_text=self._build_linearized_text(paragraphs, tables),
            document_metadata=dict(document_payload.metadata),
        )

    def _build_linearized_text(
        self,
        paragraphs: list[SemanticParagraph],
        tables: list[SemanticTable],
    ) -> str:
        """Собирает производное линейное представление документа.

        Args:
            paragraphs: Нормализованные параграфы.
            tables: Нормализованные таблицы.

        Returns:
            str: Линейное текстовое представление документа.
        """
        parts: list[str] = []

        for paragraph in paragraphs:
            parts.append(paragraph.text.strip())

        for table in tables:
            table_lines = [
                f"table:{table.id} row:{cell.row} col:{cell.col} text:{cell.text}"
                for cell in table.cells
                if cell.text
            ]
            if table_lines:
                parts.append("\n".join(table_lines))

        return "\n\n".join(part for part in parts if part)
