from pydantic import BaseModel, Field
from typing import Optional


class RowId(BaseModel):
    id_table: str = Field(..., description="Идентификатор таблицы")
    id_row: str = Field(..., description="Идентификатор строки")


class ActEntryRequest(BaseModel):
    row_id: RowId = Field(..., description="Идентификатор строки")
    record: str = Field(..., description="Описание операции")
    value: float = Field(..., ge=0, description="Значение записи")
    date: Optional[str] = Field(None, description="Дата операции")


class ActEntryResponse(BaseModel):
    row_id: RowId = Field(..., description="Идентификатор строки")
    record: str = Field(..., description="Описание операции")
    value: float = Field(..., description="Значение записи")
    date: Optional[str] = Field(None, description="Дата операции")
