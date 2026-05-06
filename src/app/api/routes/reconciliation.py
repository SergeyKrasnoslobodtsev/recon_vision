"""Определяет HTTP-маршруты работы с актами сверки."""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.dependencies import (
    get_fill_reconciliation_act_use_case,
    get_process_status_use_case,
    get_submit_reconciliation_act_use_case,
)
from app.api.schemas.act_entry import ActEntryRequest, ActEntryResponse, RowId
from app.api.schemas.process_status import (
    GetProcessStatusRequest,
    ProcessIdResponse,
    StatusResponse,
)
from app.api.schemas.reconciliation import (
    FillReconciliationActRequest,
    PeriodResponse,
    ReconciliationAct,
    ReconciliationDataResponse,
)
from app.application.dto.fill_reconciliation_act import FillReconciliationActCommand
from app.application.dto.get_process_status import GetProcessStatusCommand
from app.application.dto.submit_reconciliation_act import SubmitReconciliationActCommand
from app.application.errors import (
    ProcessFailedError,
    ProcessNotFoundError,
    ProcessNotReadyError,
)
from app.application.use_cases.fill_reconciliation_act import (
    FillReconciliationActUseCase,
)
from app.application.use_cases.get_process_status import GetProcessStatusUseCase
from app.application.use_cases.submit_reconciliation_act import (
    SubmitReconciliationActUseCase,
)
from app.domain.entities.ledger_entry import LedgerEntry, RowReference
from app.domain.entities.process import ProcessState
from app.domain.entities.reconciliation_data import ReconciliationData
from app.domain.enums.process_status import ProcessStatus

router = APIRouter(tags=["reconciliation"])


def _to_domain_entry(entry: ActEntryRequest) -> LedgerEntry:
    """Преобразует transport-модель записи в доменную сущность."""
    return LedgerEntry(
        record=entry.record,
        value=entry.value,
        date=entry.date,
        row_reference=RowReference(
            id_table=entry.row_id.id_table,
            id_row=entry.row_id.id_row,
            id_col=entry.row_id.id_col,
            buyer_col=entry.row_id.buyer_col,
        ),
    )


def _to_transport_entry(entry: LedgerEntry) -> ActEntryResponse:
    """Преобразует доменную запись в transport-модель API."""
    row_reference = entry.row_reference or RowReference(id_table="", id_row="")
    return ActEntryResponse(
        row_id=RowId(
            id_row=row_reference.id_row,
            id_table=row_reference.id_table,
            id_col=row_reference.id_col,
            buyer_col=row_reference.buyer_col,
        ),
        record=entry.record,
        value=entry.value,
        date=entry.date,
    )


def _to_reconciliation_response(
    process_state: ProcessState,
    reconciliation_data: ReconciliationData,
) -> ReconciliationDataResponse:
    """Собирает успешный ответ API из доменного состояния."""
    return ReconciliationDataResponse(
        process_id=process_state.process_id or "",
        status=0,
        message=process_state.message or reconciliation_data.message or "done",
        seller=reconciliation_data.seller,
        buyer=reconciliation_data.buyer,
        period=PeriodResponse(
            start=reconciliation_data.period.start,
            end=reconciliation_data.period.end,
        ),
        debit=[_to_transport_entry(item) for item in reconciliation_data.debit],
        credit=[_to_transport_entry(item) for item in reconciliation_data.credit],
    )


def _processing_response() -> JSONResponse:
    """Возвращает API-ответ для статуса обработки."""
    payload = StatusResponse(status=0, message="wait")
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=payload.model_dump())


def _not_found_response(process_id: str) -> JSONResponse:
    """Возвращает API-ответ для отсутствующего процесса."""
    payload = StatusResponse(status=-1, message=f"Процесс {process_id} не существует")
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=payload.model_dump())


def _failed_response(message: str) -> JSONResponse:
    """Возвращает API-ответ для статуса ошибки."""
    payload = StatusResponse(status=-2, message=message)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(),
    )


@router.post(
    "/send_reconciliation_act",
    status_code=status.HTTP_201_CREATED,
    response_model=ProcessIdResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": StatusResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": StatusResponse},
    },
)
async def send_reconciliation_act(
    request: ReconciliationAct,
    use_case: SubmitReconciliationActUseCase = Depends(get_submit_reconciliation_act_use_case),
):
    """Принимает акт сверки на обработку.

    Args:
        request: Тело запроса с исходным PDF.
    """
    logger.info("Начинаем обработку акта сверки...")
    try:
        result = await use_case.execute(SubmitReconciliationActCommand(document_base64=request.document))
        logger.info(f"Акт сверки принят, process_id={result.process_id}")
        return ProcessIdResponse(process_id=result.process_id)
    except ValueError as exc:
        payload = StatusResponse(status=-2, message=str(exc))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=payload.model_dump(),
        )
    except Exception as exc:
        logger.exception("Ошибка отправки акта сверки")
        return _failed_response(str(exc))


@router.post(
    "/process_status",
    responses={
        status.HTTP_200_OK: {"model": ReconciliationDataResponse},
        status.HTTP_201_CREATED: {"model": StatusResponse},
        status.HTTP_404_NOT_FOUND: {"model": StatusResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": StatusResponse},
    },
)
async def get_process_status(
    request: GetProcessStatusRequest,
    use_case: GetProcessStatusUseCase = Depends(get_process_status_use_case),
):
    """Возвращает статус процесса обработки акта сверки."""
    logger.info(f"Получение статуса процесса для процесса {request.process_id}...")
    try:
        result = await use_case.execute(GetProcessStatusCommand(process_id=request.process_id))
    except ProcessNotFoundError:
        return _not_found_response(request.process_id)
    except Exception as exc:
        logger.exception(f"Ошибка чтения статуса процесса для процесса {request.process_id}")
        return _failed_response(str(exc))

    process_state = result.process_state
    if process_state.status in {ProcessStatus.RECEIVED, ProcessStatus.PROCESSING}:
        return _processing_response()

    if process_state.status == ProcessStatus.FAILED:
        return _failed_response(process_state.message or "processing failed")

    if process_state.reconciliation_data is None:
        return _processing_response()

    return _to_reconciliation_response(
        process_state=process_state,
        reconciliation_data=process_state.reconciliation_data,
    )


@router.post(
    "/fill_reconciliation_act",
    response_model=ReconciliationAct,
    responses={
        status.HTTP_201_CREATED: {"model": StatusResponse},
        status.HTTP_200_OK: {"model": ReconciliationAct},
        status.HTTP_404_NOT_FOUND: {"model": StatusResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": StatusResponse},
    },
)
async def fill_reconciliation_act(
    request: FillReconciliationActRequest,
    use_case: FillReconciliationActUseCase = Depends(get_fill_reconciliation_act_use_case),
):
    """Принимает команду на заполнение акта сверки.

    Args:
        request: Данные заполнения акта сверки.
    """
    logger.info(f"Начинаем заполнение акта сверки для процесса {request.process_id}...")
    try:
        result = await use_case.execute(
            FillReconciliationActCommand(
                process_id=request.process_id,
                comments=request.comments,
                debit=[_to_domain_entry(item) for item in request.debit],
                credit=[_to_domain_entry(item) for item in request.credit],
            )
        )
        return ReconciliationAct(document=result.document_base64)
    except ProcessNotFoundError:
        logger.warning(f"Процесс {request.process_id} не найден при попытке заполнить акт сверки")
        return _not_found_response(request.process_id)
    except ProcessNotReadyError:
        logger.warning(f"Процесс {request.process_id} ещё не готов для заполнения акта сверки")
        return _processing_response()
    except ProcessFailedError as exc:
        logger.warning(f"Процесс {request.process_id} завершился с ошибкой: {exc}")
        return _failed_response(str(exc))
    except ValueError as exc:
        logger.warning(f"Ошибка в данных для заполнения акта сверки: {exc}")
        return _failed_response(str(exc))
    except Exception as exc:
        logger.exception(f"Ошибка заполнения акта сверки для процесса {request.process_id}")
        return _failed_response(str(exc))
