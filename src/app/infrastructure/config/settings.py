"""Определяет конфигурацию приложения."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from starlette.config import Config


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
        database_dir: Каталог для хранения кэша.
        logging_level: Уровень логирования.
        vision_core_config_path: Путь к конфигу vision_core.
    """

    api_prefix: str
    version: str
    debug: bool
    project_name: str
    docs_dir: str
    ttl_cache_in_hours: int
    database_dir: str
    logging_level: int
    vision_core_config_path: str


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
        ttl_cache_in_hours=config("TTL_CACHE_IN_HOURS", cast=int, default=24),
        database_dir=config("DATABASE_DIR", default="./.cache"),
        logging_level=logging.DEBUG if debug else logging.INFO,
        vision_core_config_path=config("VISION_CORE_CONFIG_PATH", default="./config/vision_core.yaml"),
    )
