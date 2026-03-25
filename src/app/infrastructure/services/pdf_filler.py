"""Реализует заполнение PDF по координатам канонического документа."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pymupdf

from app.application.dto.fill_reconciliation_act import FillReconciliationActCommand
from app.domain.entities.ledger_entry import LedgerEntry
from app.domain.entities.process import ProcessState
from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.entities.table import Table


class DocumentPdfFiller:
    """Заполняет PDF по координатам строк из канонического документа.

    Реализация опирается на сохранённый `Document` в `process_state.document_payload`.
    Для каждой записи ищется таблица и строка, после чего текст распределяется по
    ячейкам строки слева направо и записывается в PDF через PyMuPDF.
    """

    def __init__(
        self,
        default_font_size: float = 8.0,
        min_font_size: float = 6.0,
        cell_padding: float = 2.0,
    ):
        self.default_font_size = default_font_size
        self.min_font_size = min_font_size
        self.cell_padding = cell_padding
        self.font_file = self._resolve_font_file()

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

        pdf_document = pymupdf.open(stream=process_state.source_pdf, filetype="pdf")
        try:
            self._write_entries(pdf_document, document_payload, command.debit)
            self._write_entries(pdf_document, document_payload, command.credit)
            self._write_comments(pdf_document, command.comments)
            return pdf_document.tobytes()
        finally:
            pdf_document.close()

    def _write_entries(
        self,
        pdf_document: pymupdf.Document,
        source_document: Document,
        entries: list[LedgerEntry],
    ) -> None:
        """Записывает записи акта в соответствующие строки документа."""
        grouped_entries: dict[tuple[str, str], list[LedgerEntry]] = defaultdict(list)

        for entry in entries:
            if entry.row_reference is None:
                raise ValueError("Для заполнения записи требуется row_reference")
            grouped_entries[
                (entry.row_reference.id_table, entry.row_reference.id_row)
            ].append(entry)

        for (table_id, row_id), row_entries in grouped_entries.items():
            page_index, page_model, table_model = self._resolve_table(
                source_document,
                table_id,
            )
            target_cells = self._resolve_row_cells(table_model, row_id)
            text_chunks = self._build_cell_text_chunks(row_entries)
            self._write_row_to_pdf(
                pdf_page=pdf_document[page_index],
                page_model=page_model,
                cells=target_cells,
                text_chunks=text_chunks,
            )

    def _resolve_table(
        self,
        source_document: Document,
        table_id: str,
    ) -> tuple[int, Page, Table]:
        """Находит таблицу по идентификатору во всём документе."""
        normalized_target = self._normalize_identifier(table_id)

        for page_index, page in enumerate(source_document.pages):
            for table in page.tables:
                normalized_table_id = self._normalize_identifier(table.id)
                if normalized_table_id == normalized_target:
                    return page_index, page, table

        raise ValueError(f"Таблица {table_id} не найдена в каноническом документе")

    def _resolve_row_cells(self, table: Table, row_id: str):
        """Находит ячейки строки таблицы по идентификатору строки."""
        row_index = self._parse_row_index(row_id)
        candidate_indices = [row_index]
        if row_index > 0:
            candidate_indices.append(row_index - 1)

        for candidate_index in candidate_indices:
            row_cells = [cell for cell in table.cells if cell.row == candidate_index]
            if row_cells:
                return sorted(row_cells, key=lambda cell: (cell.col, cell.bbox.x_min))

        raise ValueError(f"Строка {row_id} не найдена в таблице {table.id}")

    def _build_cell_text_chunks(self, entries: list[LedgerEntry]) -> list[str]:
        """Готовит текстовые блоки для записи в ячейки строки."""
        dates = "\n".join(entry.date for entry in entries if entry.date)
        records = "\n".join(entry.record for entry in entries if entry.record)
        values = "\n".join(self._format_value(entry.value) for entry in entries)

        return [chunk for chunk in (dates, records, values) if chunk]

    def _write_row_to_pdf(
        self,
        pdf_page: pymupdf.Page,
        page_model: Page,
        cells,
        text_chunks: list[str],
    ) -> None:
        """Записывает подготовленные текстовые блоки в ячейки строки."""
        if not text_chunks:
            return

        if len(cells) == 1:
            self._write_text_to_cell(
                pdf_page,
                page_model,
                cells[0],
                "\n".join(text_chunks),
            )
            return

        if len(cells) == 2 and len(text_chunks) >= 2:
            merged_first_cell = "\n".join(text_chunks[:-1])
            self._write_text_to_cell(pdf_page, page_model, cells[0], merged_first_cell)
            self._write_text_to_cell(pdf_page, page_model, cells[1], text_chunks[-1])
            return

        for cell, text in zip(cells, text_chunks):
            self._write_text_to_cell(pdf_page, page_model, cell, text)

    def _write_text_to_cell(
        self,
        pdf_page: pymupdf.Page,
        page_model: Page,
        cell,
        text: str,
    ) -> None:
        """Пишет текст в область ячейки с подбором размера шрифта."""
        target_bbox = cell.get_largest_free_space(padding=self.cell_padding) or cell.bbox
        rect = self._to_pdf_rect(page_model, target_bbox)
        rect = pymupdf.Rect(
            rect.x0 + self.cell_padding,
            rect.y0 + self.cell_padding,
            rect.x1 - self.cell_padding,
            rect.y1 - self.cell_padding,
        )

        font_size = self.default_font_size
        while font_size >= self.min_font_size:
            remaining = pdf_page.insert_textbox(
                rect,
                text,
                fontsize=font_size,
                **self._get_font_kwargs(),
                align=0,
                overlay=True,
            )
            if remaining >= 0:
                return
            font_size -= 1.0

        pdf_page.insert_textbox(
            rect,
            text,
            fontsize=self.min_font_size,
            **self._get_font_kwargs(),
            align=0,
            overlay=True,
        )

    def _write_comments(
        self,
        pdf_document: pymupdf.Document,
        comments: str | None,
    ) -> None:
        """Добавляет комментарий на первую страницу документа."""
        if not comments or not pdf_document:
            return

        first_page = pdf_document[0]
        page_rect = first_page.rect
        comment_rect = pymupdf.Rect(
            36,
            page_rect.height - 72,
            page_rect.width - 36,
            page_rect.height - 24,
        )
        first_page.insert_textbox(
            comment_rect,
            f"Комментарии:\n{comments}",
            fontsize=self.default_font_size,
            **self._get_font_kwargs(),
            align=0,
            overlay=True,
        )

    def _to_pdf_rect(self, page: Page, bbox) -> pymupdf.Rect:
        """Преобразует координаты bbox из image-space в PDF-space."""
        image_shape = page.metadata.get("image_shape") or []
        page_size = page.metadata.get("page_size") or []

        if len(image_shape) != 2 or len(page_size) != 2:
            return pymupdf.Rect(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)

        image_height, image_width = float(image_shape[0]), float(image_shape[1])
        page_width, page_height = float(page_size[0]), float(page_size[1])
        scale_x = page_width / image_width if image_width else 1.0
        scale_y = page_height / image_height if image_height else 1.0

        return pymupdf.Rect(
            bbox.x_min * scale_x,
            bbox.y_min * scale_y,
            bbox.x_max * scale_x,
            bbox.y_max * scale_y,
        )

    def _parse_row_index(self, row_id: str) -> int:
        """Извлекает числовой индекс строки из строкового идентификатора."""
        if row_id.isdigit():
            return int(row_id)

        match = re.search(r"(\d+)$", row_id)
        if match is None:
            raise ValueError(f"Идентификатор строки {row_id} не содержит числового индекса")
        return int(match.group(1))

    def _normalize_identifier(self, value: str) -> str:
        """Нормализует строковый идентификатор для мягкого сопоставления."""
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _format_value(self, value: float) -> str:
        """Преобразует числовое значение в компактную строку."""
        return f"{value:g}"

    def _get_font_kwargs(self) -> dict[str, str]:
        """Возвращает параметры шрифта для PyMuPDF."""
        if self.font_file is None:
            return {"fontname": "helv"}
        return {
            "fontname": "dejavu",
            "fontfile": self.font_file,
        }

    def _resolve_font_file(self) -> str | None:
        """Находит системный шрифт с поддержкой кириллицы."""
        candidates = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None