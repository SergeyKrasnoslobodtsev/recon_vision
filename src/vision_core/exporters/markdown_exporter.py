"""Экспорт Document в Markdown для отладки и семантического анализа."""

from __future__ import annotations

from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.entities.paragraph import Paragraph, ParagraphType
from vision_core.entities.table import Table


class DocumentMarkdownExporter:
    """Конвертирует Document в Markdown.

    Контент на каждой странице упорядочивается по вертикальной позиции (Y),
    что соответствует порядку чтения. Объединённые ячейки (colspan) раскрываются
    повторением значения — такой формат корректно интерпретируется LLM.
    """

    def export(self, document: Document) -> str:
        """Конвертирует весь документ в Markdown.

        Args:
            document: Канонический документ.

        Returns:
            Markdown-строка документа.
        """
        pages_md = [self._render_page(page) for page in document.pages]
        return "\n\n---\n\n".join(pages_md).strip()

    def _render_page(self, page: Page) -> str:
        """Рендерит одну страницу в Markdown.

        Параграфы и таблицы сортируются по Y-позиции (reading order).
        """
        header = f"## Страница {page.page_number + 1}"

        blocks: list[tuple[float, str]] = []

        for paragraph in page.paragraphs:
            blocks.append((paragraph.bbox.y_min, self._render_paragraph(paragraph)))

        for table in page.tables:
            blocks.append((table.bbox.y_min, self._render_table(table)))

        blocks.sort(key=lambda b: b[0])
        content = "\n\n".join(text for _, text in blocks)

        return f"{header}\n\n{content}" if content else header

    def _render_paragraph(self, paragraph: Paragraph) -> str:
        """Рендерит параграф в Markdown.

        HEADER -> h3, FOOTER -> курсив, остальное -> обычный текст.
        """
        text = paragraph.text.strip()
        if not text:
            return ""

        if paragraph.type == ParagraphType.HEADER:
            return f"### {text}"
        if paragraph.type == ParagraphType.FOOTER:
            return f"*{text}*"
        return text

    def _render_table(self, table: Table) -> str:
        """Рендерит таблицу в Markdown.

        Строки упорядочены по row. Объединённые ячейки (colspan > 1)
        раскрываются: значение повторяется во всех занятых столбцах.
        """
        if not table.cells or table.num_cols == 0:
            return ""

        grid = self._build_grid(table)
        lines: list[str] = []

        for row_idx, row_cells in enumerate(grid):
            line = "| " + " | ".join(self._cell_text(c) for c in row_cells) + " |"
            lines.append(line)
            if row_idx == 0:
                separator = "| " + " | ".join("---" for _ in row_cells) + " |"
                lines.append(separator)

        table_info = f"<!-- table_id={table.id} rows={table.num_rows} cols={table.num_cols} -->"
        return table_info + "\n" + "\n".join(lines)

    def _build_grid(self, table: Table) -> list[list[str]]:
        """Строит двумерную сетку значений.

        Ячейки с colspan/rowspan > 1 помечаются маркером ``<`` (продолжение
        по горизонтали) или ``^`` (продолжение по вертикали), чтобы в markdown
        было видно, где объединённая ячейка, а где отдельная.
        """
        grid: list[list[str]] = [[""] * table.num_cols for _ in range(table.num_rows)]

        for cell in table.cells:
            text = self._cell_text(cell.value)
            r_end = min(cell.row + cell.rowspan, table.num_rows)
            c_end = min(cell.col + cell.colspan, table.num_cols)
            for r in range(cell.row, r_end):
                for c in range(cell.col, c_end):
                    if r == cell.row and c == cell.col:
                        grid[r][c] = text
                    elif c > cell.col:
                        grid[r][c] = "<"
                    else:
                        grid[r][c] = "^"

        return grid

    def _cell_text(self, value: str | None) -> str:
        """Нормализует текст ячейки: убирает переносы строк, экранирует пайп."""
        if not value:
            return ""
        return value.strip().replace("\n", " ").replace("|", "\\|")
