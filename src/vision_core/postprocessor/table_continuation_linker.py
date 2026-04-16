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
            prev_table = self._bottom_table(pages[i])
            next_table = self._top_table(pages[i + 1])

            logger.debug(
                f"Страницы {i}->{i + 1}: "
                f"нижняя таблица={prev_table.id if prev_table else None}, "
                f"верхняя таблица={next_table.id if next_table else None}"
            )

            if prev_table and next_table and self._is_continuation(next_table, pages[i + 1]):
                next_table.continuation_of = prev_table.id
                logger.debug(f"  Связано: {next_table.id} -> continuation_of={prev_table.id}")
            else:
                logger.debug("  Не связано")

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

        blocking = [
            p for p in page.paragraphs
            if p.type == ParagraphType.BODY_TEXT and p.bbox.y_min < table.bbox.y_min
        ]
        if blocking:
            logger.debug(
                f"  _is_continuation {table.id}: {len(blocking)} BODY_TEXT параграфов выше таблицы, отклонено"
            )
            return False

        logger.debug(
            f"  _is_continuation {table.id}: y_min/h={y_ratio:.3f}, параграфов выше нет — продолжение"
        )
        return True

    def _page_height(self, page: Page) -> int:
        shape = page.metadata.get("image_shape")
        if not shape or len(shape) < 1:
            return 0
        return int(shape[0])
