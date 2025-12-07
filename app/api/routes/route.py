from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from app.models.recon_act import (
    ReconciliationAct,
    ReconciliationDataResponse,
    FillReconciliationActRequest,
)
from app.models.process_status import (
    ProcessIdResponse,
    StatusResponse,
)

router = APIRouter()


@router.post(
    "/send_reconciliation_act",
    status_code=status.HTTP_201_CREATED,
    response_model=ProcessIdResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": StatusResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": StatusResponse},
    },
)
async def send_reconciliation_act(request: ReconciliationAct):
    await run_in_threadpool(logger.info, "Sending reconciliation act...")
    pass


@router.post(
    "/process_status",
    responses={
        status.HTTP_200_OK: {"model": ReconciliationDataResponse},
        status.HTTP_201_CREATED: {"model": StatusResponse},
        status.HTTP_404_NOT_FOUND: {"model": StatusResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": StatusResponse},
    },
)
async def get_process_status():
    await run_in_threadpool(logger.info, "Getting process status...")
    pass


@router.post(
    "/fill_reconciliation_act",
    response_model=ReconciliationAct,
    responses={
        status.HTTP_200_OK: {"model": ReconciliationAct},
        status.HTTP_404_NOT_FOUND: {"model": StatusResponse},
        # status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": StatusResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": StatusResponse},
    },
)
async def fill_reconciliation_act(request: FillReconciliationActRequest):
    await run_in_threadpool(logger.info, "Filling reconciliation act...")
    pass
