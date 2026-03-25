"""Определяет транспортные схемы API акта сверки."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.api.schemas.act_entry import ActEntryRequest, ActEntryResponse


class ReconciliationAct(BaseModel):
    """Описывает запрос или ответ с PDF-документом в base64."""

    document: str = Field(..., description="PDF документ в формате base64")


class FillReconciliationActRequest(BaseModel):
    """Описывает запрос на заполнение акта сверки."""

    process_id: str = Field(..., description="Идентификатор процесса")
    comments: str | None = Field(None, description="Комментарии к акту сверки")
    debit: list[ActEntryRequest] = Field(..., description="Записи дебета")
    credit: list[ActEntryRequest] = Field(..., description="Записи кредита")


class PeriodResponse(BaseModel):
    """Описывает период сверки во внешнем API."""

    model_config = {"extra": "allow"}


class ReconciliationDataResponse(BaseModel):
    """Описывает успешный ответ сервиса по результатам обработки документа."""

    process_id: str = Field(..., description="Идентификатор процесса")
    status: int = Field(..., description="Статус обработки акта сверки")
    message: str = Field(..., description="Сообщение")
    seller: str = Field(..., description="Продавец")
    buyer: str = Field(..., description="Покупатель")
    period: PeriodResponse = Field(..., description="Период сверки")
    debit: list[ActEntryResponse] = Field(default_factory=list, description="Записи дебета")
    credit: list[ActEntryResponse] = Field(default_factory=list, description="Записи кредита")
