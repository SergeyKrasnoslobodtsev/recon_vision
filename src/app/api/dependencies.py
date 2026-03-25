"""Определяет dependency-функции FastAPI для прикладного слоя."""

from __future__ import annotations

from fastapi import Request

from app.application.use_cases.fill_reconciliation_act import (
    FillReconciliationActUseCase,
)
from app.application.use_cases.get_process_status import GetProcessStatusUseCase
from app.application.use_cases.submit_reconciliation_act import (
    SubmitReconciliationActUseCase,
)
from app.bootstrap.container import ApplicationContainer


def get_container(request: Request) -> ApplicationContainer:
    """Возвращает контейнер зависимостей приложения.

    Args:
        request: Текущий HTTP-запрос.

    Returns:
        ApplicationContainer: Контейнер зависимостей.
    """
    return request.app.state.container


def get_submit_reconciliation_act_use_case(
    request: Request,
) -> SubmitReconciliationActUseCase:
    """Возвращает use case отправки акта сверки."""
    return get_container(request).submit_reconciliation_act


def get_process_status_use_case(request: Request) -> GetProcessStatusUseCase:
    """Возвращает use case чтения статуса процесса."""
    return get_container(request).get_process_status


def get_fill_reconciliation_act_use_case(
    request: Request,
) -> FillReconciliationActUseCase:
    """Возвращает use case заполнения акта сверки."""
    return get_container(request).fill_reconciliation_act
