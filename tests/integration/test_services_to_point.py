from fastapi import APIRouter, HTTPException, status, Request
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
from app.services.pdf_parser import PDFParserService
from app.services.pdf_filler import PDFFillerService




router = APIRouter()

# Инициализация сервисов
pdf_parser = PDFParserService()
pdf_filler = PDFFillerService()


@router.post(
    "/send_reconciliation_act",
    status_code=status.HTTP_201_CREATED,
    response_model=ProcessIdResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": StatusResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": StatusResponse},
    },
)
async def send_reconciliation_act(request: ReconciliationAct, req: Request):
    """Получает PDF акта сверки, парсит его и сохраняет в кэш"""
    try:
        logger.info("Получен запрос на обработку акта сверки")

        # Валидация PDF
        if not pdf_parser.validate_pdf(request.document):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Невалидный PDF документ",
            )

        # Парсинг PDF
        structure = await pdf_parser.parse_pdf(request.document)

        # Сохранение в кэш
        cache = req.app.state.cache
        import base64

        pdf_bytes = base64.b64decode(request.document)
        process_id = await cache.create_process(pdf_bytes, structure)

        logger.info(f"Создан процесс {process_id}")
        return ProcessIdResponse(process_id=process_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки акта сверки: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера",
        )


@router.post(
    "/process_status",
    responses={
        status.HTTP_200_OK: {"model": ReconciliationDataResponse},
        status.HTTP_201_CREATED: {"model": StatusResponse},
        status.HTTP_404_NOT_FOUND: {"model": StatusResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": StatusResponse},
    },
)
async def get_process_status(request: ProcessIdResponse, req: Request):
    """Получает статус обработки процесса"""
    try:
        logger.info(f"Запрос статуса процесса {request.process_id}")

        cache = req.app.state.cache
        structure = await cache.get_structure(request.process_id)

        if structure is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Процесс не найден или истёк",
            )

        # Возвращаем распарсенные данные
        logger.info(f"Процесс {request.process_id} найден")
        return ReconciliationDataResponse(**structure)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения статуса: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера",
        )


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
async def fill_reconciliation_act(request: FillReconciliationActRequest, req: Request):
    """Заполняет PDF акт сверки данными"""
    try:
        logger.info(f"Запрос на заполнение акта для процесса {request.process_id}")

        cache = req.app.state.cache

        # Получаем оригинальный PDF из кэша
        pdf_bytes = await cache.get_pdf(request.process_id)
        if pdf_bytes is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Процесс не найден или истёк",
            )

        # Заполняем PDF данными из request
        filled_pdf_bytes = await pdf_filler.fill_pdf(
            pdf_bytes, request.data.model_dump()
        )

        # Кодируем в base64
        import base64

        filled_pdf_base64 = base64.b64encode(filled_pdf_bytes).decode("utf-8")

        logger.info(f"Акт успешно заполнен для процесса {request.process_id}")
        return ReconciliationAct(document=filled_pdf_base64)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка заполнения акта: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера",
        )
