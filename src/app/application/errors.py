"""Определяет ошибки прикладного слоя."""


class ApplicationError(Exception):
    """Базовая ошибка прикладного слоя."""


class ProcessNotFoundError(ApplicationError):
    """Сигнализирует, что процесс не найден."""


class ProcessNotReadyError(ApplicationError):
    """Сигнализирует, что процесс ещё не завершил обработку."""


class ProcessFailedError(ApplicationError):
    """Сигнализирует, что процесс завершился ошибкой."""
