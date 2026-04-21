"""Иерархия исключений vision_core.

Структура:
    VisionCoreError                     # базовый абстрактный класс
      DocumentParseError                # ошибки разбора структуры документа
        DcColsNotFoundError             # не найдены колонки дебет/кредит
        DcColsInvalidPositionError      # нарушено расположение дебет/кредит
      PdfLoadError                      # ошибки загрузки PDF
      OcrError                          # ошибки OCR-движка
"""

from __future__ import annotations


class VisionCoreError(Exception):
    """Базовый класс исключений vision_core.

    Сообщение задаётся в подклассе через атрибут `message`.
    Дополнительная информация передаётся через kwargs и подставляется в сообщение.
    """

    message: str = "Ошибка vision_core"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(self.message.format(**kwargs) if kwargs else self.message)


# ---------------------------------------------------------------------------
# Разбор документа
# ---------------------------------------------------------------------------

class DocumentParseError(VisionCoreError):
    """Ошибка разбора структуры документа."""

    message = "Ошибка разбора документа"


class DcColsNotFoundError(DocumentParseError):
    """Не найдены колонки дебет/кредит в таблице."""

    message = "Не найдены колонки дебет/кредит: таблица '{table_id}'"


class DcColsInvalidPositionError(DocumentParseError):
    """Нарушено расположение колонок дебет/кредит."""

    message = (
        "Нарушено расположение дебет/кредит: таблица '{table_id}', "
        "col {col} выходит за пределы таблицы (num_cols={num_cols})"
    )


# ---------------------------------------------------------------------------
# Загрузка PDF
# ---------------------------------------------------------------------------

class PdfLoadError(VisionCoreError):
    """Не удалось загрузить PDF."""

    message = "Не удалось загрузить PDF"


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

class OcrError(VisionCoreError):
    """Ошибка OCR-движка."""

    message = "Ошибка OCR-движка"


class OcrEmptyResultError(OcrError):
    """OCR не вернул результатов для страницы."""

    message = "OCR не вернул результатов: страница {page_number}"
