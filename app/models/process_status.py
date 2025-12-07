from pydantic import BaseModel, Field


class GetProcessStatusRequest(BaseModel):
    process_id: str = Field(..., description="Идентификатор процесса")


class ProcessIdResponse(BaseModel):
    process_id: str = Field(..., description="Уникальный идентификатор процесса")


class StatusResponse(BaseModel):
    status: int = Field(..., description="Статус обработки акта сверки")
    message: str = Field(..., description="Сообщение")
