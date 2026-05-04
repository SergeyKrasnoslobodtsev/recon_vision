"""Реализует заполнение PDF по координатам канонического документа."""

from __future__ import annotations

import PIL.Image as Image
from loguru import logger

from app.application.dto.fill_reconciliation_act import FillReconciliationActCommand
from app.domain.entities.ledger_entry import LedgerEntry
from app.domain.entities.process import ProcessState
from app.infrastructure.services.pdf_fill_context import TableFillContext, build_fill_contexts
from app.infrastructure.services.pdf_fill_render import (
    draw_centered_text,
    draw_comments,
    load_aligned_page_images,
    resolve_font_file,
    save_page_images_as_pdf,
)
from vision_core.entities.document import Document


class DocumentPdfFiller:
    """Заполняет PDF по координатам строк из канонического документа.

    Реализация опирается на сохранённый `Document` в `process_state.document_payload`.
    Для каждой строки ищется таблица и целевая ячейка суммы, после чего
    значения покупателя из дебета и кредита записываются в свои колонки.
    """

    def __init__(self):
        self.font_file = resolve_font_file()

    async def fill(
        self,
        process_state: ProcessState,
        command: FillReconciliationActCommand,
    ) -> bytes:
        """Заполняет исходный PDF данными акта сверки.

        Args:
            process_state: Состояние процесса с исходным PDF и каноническим документом.
            command: Команда заполнения документа.

        Returns:
            bytes: Новый PDF с нанесёнными данными.

        Raises:
            ValueError: Если отсутствует канонический документ или координаты строки.
        """
        document_payload = process_state.document_payload
        if not isinstance(document_payload, Document):
            raise ValueError("В процессе отсутствует канонический Document для заполнения")

        aligned_pages = load_aligned_page_images(process_state.source_pdf, document_payload)
        page_images = {
            page.page_number: image.copy() for page, image in zip(document_payload.pages, aligned_pages, strict=False)
        }
        contexts = build_fill_contexts(document_payload)

        for entry in command.debit:
            self._draw_entry(page_images, contexts, entry, entry_kind="debit")
        for entry in command.credit:
            self._draw_entry(page_images, contexts, entry, entry_kind="credit")

        draw_comments(page_images, document_payload, command.comments, self.font_file)
        return save_page_images_as_pdf([page_images[page.page_number] for page in document_payload.pages])

    def _draw_entry(
        self,
        page_images: dict[int, Image.Image],
        contexts: dict[str, TableFillContext],
        entry: LedgerEntry,
        entry_kind: str,
    ) -> None:
        """Рисует одну сумму акта по координатам строки таблицы.

        Args:
            page_images: Изображения страниц, подготовленные для рисования.
            contexts: Контексты заполнения таблиц.
            entry: Запись акта сверки со значением покупателя.
            entry_kind: Тип суммы, `debit` или `credit`.
        """
        row_reference = entry.row_reference
        if row_reference is None:
            logger.debug("pdf_filler: entry skipped, row_reference is missing")
            return

        context = contexts.get(row_reference.id_table)
        if context is None:
            logger.debug(f"pdf_filler: table={row_reference.id_table} not found in fill contexts")
            return

        row_index = int(row_reference.id_row)
        value_cell, buyer_col = self._resolve_value_cell(context, row_index, row_reference.id_col)
        if value_cell is None:
            logger.debug(
                f"pdf_filler: cell not found table={context.table.id} row={row_index} "
                f"kind={entry_kind} seller_col={row_reference.id_col}"
            )
            return

        logger.debug(
            f"pdf_filler: draw table={context.table.id} row={row_index} kind={entry_kind} "
            f"seller_col={row_reference.id_col} buyer_col={buyer_col} value={entry.value}"
        )
        draw_centered_text(
            page_images[context.page_number],
            value_cell,
            self._format_amount(entry.value),
            context.font_size,
            self.font_file,
        )

    def _format_amount(self, value: float) -> str:
        """Форматирует сумму в виде 0 000 000,00.

        Args:
            value: Денежное значение.

        Returns:
            str: Строка суммы с пробелами по тысячам и запятой в дробной части.
        """
        return f"{value:,.2f}".replace(",", " ").replace(".", ",")

    def _resolve_value_cell(
        self,
        context: TableFillContext,
        row_index: int,
        seller_col: int | None,
    ) -> tuple[Image.Image | None, int | None]:
        """Определяет целевую ячейку для суммы.

        Args:
            context: Контекст заполнения таблицы.
            row_index: Индекс строки.
            seller_col: Колонка продавца из сохранённой структуры.

        Returns:
            tuple[Image.Image | None, int | None]: Ячейка покупателя и её колонка.
        """
        if seller_col is None:
            return None, None

        buyer_col = context.seller_to_buyer_cols.get(seller_col, seller_col)
        return context.table.get_cell(row_index, buyer_col), buyer_col
