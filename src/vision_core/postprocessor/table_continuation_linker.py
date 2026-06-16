"""Постпроцессор для связывания таблиц, продолжающихся на следующей странице."""

from __future__ import annotations

from loguru import logger

from vision_core.entities.page import Page
from vision_core.entities.paragraph import ParagraphType
from vision_core.entities.table import Table


class TableContinuationLinker:
    """Проставляет ссылку ``continuation_of`` между таблицами на соседних страницах.

    Таблица на странице N+1 считается продолжением последней таблицы страницы N,
    если выполнены оба условия:
      1. Таблица находится в верхней части страницы (y_min < top_margin_ratio * h).
      2. Над таблицей нет текстовых параграфов типа BODY_TEXT (допускаются только
         HEADER/FOOTER — колонтитулы и номера страниц).

    Метод ``link`` мутирует поле ``continuation_of`` у таблиц-продолжений.
    """

    def __init__(self, top_margin_ratio: float = 0.15) -> None:
        self.top_margin_ratio = top_margin_ratio

    def link(self, pages: list[Page]) -> None:
        """Проставляет ссылки continuation_of для всех страниц документа.

        Args:
            pages: Страницы документа в порядке следования.
        """
        for i in range(len(pages) - 1):
            self._link_tables_within_page(pages[i])

            prev_table = self._bottom_table(pages[i])
            next_table = self._top_table(pages[i + 1])

            logger.debug(
                f"Страницы {i}->{i + 1}: "
                f"нижняя таблица={prev_table.id if prev_table else None}, "
                f"верхняя таблица={next_table.id if next_table else None}"
            )

            if prev_table and next_table and self._is_continuation(next_table, pages[i + 1]):
                next_table.continuation_of = prev_table.id
                logger.debug(f"Связано: {next_table.id} -> continuation_of={prev_table.id}")
            else:
                logger.warning(f" Таблица {next_table.id if next_table else None} не связана с предыдущей страницей")

    def _link_tables_within_page(self, page: Page) -> None:
        """Связывает таблицы, разбитые внутри одной страницы."""
        if len(page.tables) < 2:
            return

        # Отсортировать таблицы по y_min (вертикальный порядок)
        sorted_tables = sorted(page.tables, key=lambda t: t.bbox.y_min)

        # Проверить соседние пары
        for i in range(len(sorted_tables) - 1):
            table1 = sorted_tables[i]
            table2 = sorted_tables[i + 1]

            if self._is_same_page_continuation(table1, table2):
                table2.continuation_of = table1.id
                logger.debug(f"Связано (внутристраничное): {table2.id} -> {table1.id}")

    def _is_same_page_continuation(self, table1: Table, table2: Table) -> bool:
        """Проверяет, является ли table2 продолжением table1 на одной странице."""
        # Условие 1: количество столбцов совпадает
        if table1.num_cols != table2.num_cols:
            logger.debug(f"  Не продолжение: столбцы {table1.num_cols} != {table2.num_cols}")
            return False

        # Условие 2: расстояние < медиана высоты строк table1
        gap = table2.bbox.y_min - table1.bbox.y_max
        median_height = table1.median_height_blobs_per_cells()

        if gap < 0 or gap > median_height:
            logger.debug(f"  Не продолжение: gap={gap:.1f} > median_height={median_height:.1f}")
            return False

        logger.debug(f"  Продолжение: gap={gap:.1f} <= median_height={median_height:.1f}")
        return True

    def _bottom_table(self, page: Page) -> Table | None:
        """Возвращает таблицу с наибольшим y_max на странице."""
        if not page.tables:
            return None
        return max(page.tables, key=lambda t: t.bbox.y_max)

    def _top_table(self, page: Page) -> Table | None:
        """Возвращает таблицу с наименьшим y_min на странице."""
        if not page.tables:
            return None
        return min(page.tables, key=lambda t: t.bbox.y_min)

    def _is_continuation(self, table: Table, page: Page) -> bool:
        """Проверяет, является ли таблица продолжением с предыдущей страницы."""
        page_height = self._page_height(page)
        if page_height == 0:
            logger.debug(f"  _is_continuation {table.id}: page_height=0, отклонено")
            return False

        y_ratio = table.bbox.y_min / page_height
        if y_ratio > self.top_margin_ratio:
            logger.debug(
                f"  _is_continuation {table.id}: y_min/h={y_ratio:.3f} > "
                f"{self.top_margin_ratio} (таблица не в верхней части), отклонено"
            )
            return False

        blocking = [p for p in page.paragraphs if p.type == ParagraphType.BODY_TEXT and p.bbox.y_min < table.bbox.y_min]
        if blocking:
            logger.debug(f"  _is_continuation {table.id}: {len(blocking)} BODY_TEXT параграфов выше таблицы, отклонено")
            return False

        logger.debug(f"  _is_continuation {table.id}: y_min/h={y_ratio:.3f}, параграфов выше нет — продолжение")
        return True

    def _page_height(self, page: Page) -> int:
        shape = page.metadata.get("image_shape")
        if not shape or len(shape) < 1:
            return 0
        return int(shape[0])
