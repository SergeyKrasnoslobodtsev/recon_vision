from app.infrastructure.services.semantic_input_projector import (
    DocumentSemanticInputProjector,
)
from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.entities.paragraph import Paragraph, ParagraphType
from vision_core.entities.table import Table


def test_project_builds_semantic_input_from_document():
    projector = DocumentSemanticInputProjector()
    document = Document(
        pages=[
            Page(
                page_number=0,
                paragraphs=[
                    Paragraph(
                        id="p-1",
                        text="Текст абзаца",
                        type=ParagraphType.TEXT,
                        bbox=BBox(x_min=1, y_min=2, x_max=10, y_max=20),
                    )
                ],
                tables=[
                    Table(
                        id="table-1",
                        num_rows=1,
                        num_cols=1,
                        cells=[
                            Cell(
                                row=0,
                                col=0,
                                value="Ячейка",
                                bbox=BBox(x_min=5, y_min=5, x_max=15, y_max=15),
                            )
                        ],
                        bbox=BBox(x_min=0, y_min=0, x_max=20, y_max=20),
                    )
                ],
            )
        ],
        metadata={"message": "done"},
    )

    semantic_input = __import__("asyncio").run(projector.build(document))

    assert len(semantic_input.paragraphs) == 1
    assert semantic_input.paragraphs[0].text == "Текст абзаца"
    assert semantic_input.paragraphs[0].block_type == "text"
    assert len(semantic_input.tables) == 1
    assert semantic_input.tables[0].cells[0].text == "Ячейка"
    assert semantic_input.document_metadata == {"message": "done"}
    assert "Текст абзаца" in semantic_input.linearized_text
    assert "table:table-1" in semantic_input.linearized_text