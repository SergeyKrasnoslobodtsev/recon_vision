"""Реализует заполнение PDF по координатам канонического документа."""

from __future__ import annotations

from loguru import logger

from app.application.dto.fill_reconciliation_act import FillReconciliationActCommand
from app.domain.entities.process import ProcessState
from app.infrastructure.services.pdf_fill.context import build_fill_contexts, resolve_entry_cell
from app.infrastructure.services.pdf_fill.render import (
    PageRenderer,
    format_amount,
    load_aligned_page_images,
    resolve_font_file,
    save_page_images_as_pdf,
)
from vision_core.entities.document import Document


class DocumentPdfFiller:
    def __init__(self):
        self._font_file = resolve_font_file()

    async def fill(
        self,
        process_state: ProcessState,
        command: FillReconciliationActCommand,
    ) -> bytes:
        document = process_state.document_payload
        if not isinstance(document, Document):
            raise ValueError("В процессе отсутствует канонический Document для заполнения")

        aligned_pages = load_aligned_page_images(process_state.source_pdf, document)
        page_images = {p.page_number: img.copy() for p, img in zip(document.pages, aligned_pages, strict=False)}
        contexts = build_fill_contexts(document)

        # Создаем рендереры для каждой страницы
        renderers = {page_num: PageRenderer(img, self._font_file) for page_num, img in page_images.items()}

        for entry in [*command.debit, *command.credit]:
            # if entry.value == 0.0:
            #     continue
            resolved = resolve_entry_cell(contexts, entry)
            if resolved is None:
                continue
            cell, page_number, font_size = resolved
            ref = entry.row_reference
            logger.debug(f"заполняем таблица={ref.id_table} R{ref.id_row}:C{ref.buyer_col} значение={entry.value}")
            renderers[page_number].draw_centered_text(cell, format_amount(entry.value), font_size)

        # Добавляем комментарии на последнюю страницу с таблицей
        if command.comments:
            for page in reversed(document.pages):
                if page.tables:
                    renderers[page.page_number].draw_comments(command.comments)
                    break

        return save_page_images_as_pdf([page_images[p.page_number] for p in document.pages])
