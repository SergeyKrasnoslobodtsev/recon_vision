"""
Модуль с исключениями, специфичными для extractor.



"""

from __future__ import annotations


class ExtractorError(Exception):
    """Базовый класс исключений extractor."""

    message: str = "Ошибка extractor"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(self.message.format(**kwargs) if kwargs else self.message)


class CompanyExtractionError(ExtractorError):
    """Ошибка извлечения организаций из документа."""

    message = "Ошибка извлечения организаций из документа: {details}"


class CompanyNotFoundError(CompanyExtractionError):
    """Не удалось извлечь ни одной организации из документа."""

    message = "Не удалось извлечь ни одной организации из документа"


class RoleAssignmentError(CompanyExtractionError):
    """Ошибка определения ролей организаций в документе."""

    message = "Ошибка определения роли для организации '{org}'"


class RoleAssignmentNotFoundError(CompanyExtractionError):
    """Не удалось определить роли организаций в документе."""

    message = "Не удалось определить роли организаций в документе"


class DcExtractionError(ExtractorError):
    """Ошибка извлечения бухгалтерских записей дебет/кредит из таблиц документа."""

    message = "Ошибка извлечения дебет/кредит из таблицы '{table_id}': {details}"
