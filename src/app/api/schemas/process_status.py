"""Определяет транспортные схемы статусов обработки."""

from pydantic import BaseModel, Field


class GetProcessStatusRequest(BaseModel):
    """Описывает запрос получения состояния процесса."""

    process_id: str = Field(..., description="Идентификатор процесса")


class ProcessIdResponse(BaseModel):
    """Описывает ответ с идентификатором процесса."""

    process_id: str = Field(..., description="Уникальный идентификатор процесса")


class StatusResponse(BaseModel):
    """Описывает краткий ответ о состоянии обработки."""

    status: int = Field(..., description="Статус обработки акта сверки")
    message: str = Field(..., description="Сообщение")
