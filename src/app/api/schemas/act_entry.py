"""Определяет транспортные схемы записей акта сверки."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RowId(BaseModel):
    """Описывает идентификатор строки в таблице документа."""

    id_row: str = Field(..., description="Идентификатор строки")
    id_table: str = Field(..., description="Идентификатор таблицы")


class ActEntryRequest(BaseModel):
    """Описывает запись для сценария заполнения акта сверки."""

    row_id: RowId = Field(..., description="Идентификатор строки")
    record: str = Field(..., description="Описание операции")
    value: float = Field(..., ge=0, description="Значение записи")
    date: str | None = Field(None, description="Дата операции")


class ActEntryResponse(BaseModel):
    """Описывает запись, извлечённую из акта сверки."""

    row_id: RowId = Field(..., description="Идентификатор строки")
    record: str = Field(..., description="Описание операции")
    value: float = Field(..., description="Значение записи")
    date: str | None = Field(None, description="Дата операции")
