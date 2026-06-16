"""Сущность представления логической таблицы. Собирается из сырых таблиц."""

from enum import Enum

from pydantic import BaseModel, Field

from vision_core.entities.cell import Cell


class ColumnType(Enum):
    """Тип колонки логической таблицы."""

    DEBIT = "debit"
    CREDIT = "credit"
    OTHER = "other"


class Column(BaseModel):
    """Сущность представления колонки логической таблицы.

    Attributes:
        index: Индекс колонки в логической таблице.
        type: Тип колонки в логической таблице.
    """

    index: int
    type: ColumnType = ColumnType.OTHER


class Row(BaseModel):
    """Сущность представления строки логической таблицы.

    Attributes:
        index: Индекс строки в логической таблице.
        cells: Список ячеек в строке.
    """

    index: int
    cells: list[Cell] = Field(default_factory=list)


class LogicalTable(BaseModel):
    """Логическая таблица, собранная из сырых таблиц.

    Attributes:
        id: Уникальный идентификатор логической таблицы.
        start_page: Номер страницы, на которой начинается логическая таблица.
        end_page: Номер страницы, на которой заканчивается логическая таблица.
    """

    id: str
    rows: list[Row] = Field(default_factory=list)
    columns: list[Column] = Field(default_factory=list)
    start_page: int = 0
    end_page: int = 0

    @property
    def dc_cols(self) -> set[int]:
        return {c.index for c in self.columns if c.type in (ColumnType.DEBIT, ColumnType.CREDIT)}

    def column_cells(self, index: int) -> list[Cell]:
        return [c for r in self.rows for c in r.cells if c.col == index]
