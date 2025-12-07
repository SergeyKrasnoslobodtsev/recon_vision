from pydantic import BaseModel, Field
from typing import Optional, List
from .act_entry import ActEntryRequest, ActEntryResponse


class ReconciliationAct(BaseModel):
    document: str = Field(..., description="PDF документ в формате base64")


class FillReconciliationActRequest(BaseModel):
    process_id: str = Field(..., description="Идентификатор процесса")
    comments: Optional[str] = Field(None, description="Комментарии к акту сверки")
    debit: List[ActEntryRequest] = Field(..., description="Записи дебета")
    credit: List[ActEntryRequest] = Field(..., description="Записи кредита")


class ReconciliationDataResponse(BaseModel):
    process_id: str = Field(..., description="Идентификатор процесса")
    status: int = Field(description="Статус обработки акта сверки")
    message: str = Field(..., description="Сообщение")
    seller: str = Field(..., description="Продавец")
    buyer: str = Field(..., description="Покупатель")
    period: dict = Field(..., description="Период сверки")
    debit: List[ActEntryResponse] = Field(
        default_factory=list, description="Записи дебета"
    )
    credit: List[ActEntryResponse] = Field(
        default_factory=list, description="Записи кредита"
    )
