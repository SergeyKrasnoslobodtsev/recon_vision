"""Управляет жизненным циклом FastAPI приложения."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.bootstrap.container import create_container
from app.infrastructure.config.settings import AppSettings
from app.infrastructure.persistence.diskcache_process_repository import (
    DiskCacheProcessRepository,
)


def create_lifespan(settings: AppSettings) -> Callable[[FastAPI], AsyncIterator[None]]:
    """Создаёт lifespan-обработчик для приложения.

    Args:
        settings: Конфигурация приложения.

    Returns:
        Callable[[FastAPI], AsyncIterator[None]]: Функция lifespan для FastAPI.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        repository = DiskCacheProcessRepository(
            expire=settings.ttl_cache_in_hours * 3600, cache_dir=settings.database_dir
        )
        cleanup_task = asyncio.create_task(repository.cleanup_expired_cache())
        container = create_container(repository, settings)
        app.state.process_repository = repository
        app.state.container = container

        logger.info("Инициализирован process repository и контейнер приложения")

        try:
            yield
        finally:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                logger.debug("Фоновая очистка кэша остановлена")
            repository.close()

    return lifespan
