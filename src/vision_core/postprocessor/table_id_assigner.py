"""Постпроцессор для перенумерации идентификаторов таблиц в документе."""

from __future__ import annotations

from vision_core.entities.page import Page


class TableIdAssigner:
    """Пересчитывает идентификаторы таблиц в порядке следования по документу.

    Построение выполняется до связывания таблиц-продолжений, чтобы у каждой
    таблицы был устойчивый натуральный порядковый номер в пределах документа:
    ``0``, ``1``, ``2`` и т.д. Если таблица уже содержит ``continuation_of`` с
    уникальным исходным идентификатором, ссылка также будет обновлена.
    """

    def assign(self, pages: list[Page]) -> None:
        """Перенумеровывает таблицы in-place.

        Args:
            pages: Страницы документа в порядке следования.
        """
        old_to_new: dict[str, str] = {}
        ordered_tables = [table for page in pages for table in page.tables]

        for index, table in enumerate(ordered_tables):
            old_to_new[table.id] = str(index)
            table.id = str(index)

        for table in ordered_tables:
            if table.continuation_of is None:
                continue
            table.continuation_of = old_to_new.get(table.continuation_of, table.continuation_of)
