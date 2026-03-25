"""Определяет конфигурацию приложения."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from starlette.config import Config
from starlette.datastructures import Secret


@dataclass(frozen=True)
class AppSettings:
    """Хранит конфигурацию приложения.

    Attributes:
        api_prefix: Префикс API маршрутов.
        version: Версия сервиса.
        debug: Флаг режима отладки.
        project_name: Имя проекта.
        docs_dir: Каталог со статическими ресурсами документации.
        ttl_cache_in_hours: TTL кэша в часах.
        secret_key: Секрет приложения.
        database_url: Строка подключения к базе данных.
        memoization_flag: Флаг мемоизации.
        max_connections_count: Максимум подключений.
        min_connections_count: Минимум подключений.
        logging_level: Уровень логирования.
    """

    api_prefix: str
    version: str
    debug: bool
    project_name: str
    docs_dir: str
    ttl_cache_in_hours: int
    secret_key: Secret
    database_url: str
    memoization_flag: bool
    max_connections_count: int
    min_connections_count: int
    logging_level: int


def get_settings() -> AppSettings:
    """Возвращает конфигурацию приложения.

    Returns:
        AppSettings: Собранная конфигурация из окружения.
    """

    config = Config(".env")
    debug = config("DEBUG", cast=bool, default=False)

    return AppSettings(
        api_prefix="/api",
        version="0.1.0",
        debug=debug,
        project_name=config("PROJECT_NAME", default="ReconVision"),
        docs_dir="./static",
        ttl_cache_in_hours=24,
        secret_key=config("SECRET_KEY", cast=Secret, default=""),
        database_url=config("DATABASE_URL", default="sqlite:///./app.db"),
        memoization_flag=config("MEMOIZATION_FLAG", cast=bool, default=True),
        max_connections_count=config(
            "MAX_CONNECTIONS_COUNT",
            cast=int,
            default=10,
        ),
        min_connections_count=config(
            "MIN_CONNECTIONS_COUNT",
            cast=int,
            default=10,
        ),
        logging_level=logging.DEBUG if debug else logging.INFO,
    )
