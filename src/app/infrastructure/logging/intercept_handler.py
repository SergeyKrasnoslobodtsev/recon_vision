"""Адаптирует стандартный logging к Loguru."""

import logging

from loguru import logger


class InterceptHandler(logging.Handler):
    """Проксирует записи стандартного логгера в Loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        """Перенаправляет запись в Loguru.

        Args:
            record: Запись стандартного логгера.
        """
        logger_opt = logger.opt(depth=7, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())
