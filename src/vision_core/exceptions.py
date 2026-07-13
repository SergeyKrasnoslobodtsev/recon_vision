"""Иерархия исключений vision_core.

Структура:
    VisionCoreError                       # базовый абстрактный класс
      ModelLoadError                      # ошибки загрузки моделей
      DocumentParseError                  # ошибки разбора структуры документа
        DcColsNotFoundError               # не найдены колонки дебет/кредит
        DcColsInvalidPositionError        # нарушено расположение дебет/кредит
        TableNotFoundError                # не найдены таблицы
        TableParseError                   # ошибка разбора таблицы
        ParagraphNotFoundError            # не найдены параграфы
      PdfLoadError                        # ошибки загрузки PDF
      OcrError                            # ошибки OCR-движка
        RecognitionQualityError           # качество распознавания ниже порога
        OcrEmptyResultError               # OCR не вернул результатов
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


class ModelLoadError(VisionCoreError):
    """Ошибка загрузки модели."""

    message = "Ошибка загрузки модели: {details}"


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


class DcColsResolutionError(DocumentParseError):
    """Ошибка при разрешении колонок дебет/кредит для таблицы-продолжения."""

    message = "Ошибка разрешения колонок дебет/кредит для таблицы '{table_id}': {details}"


class TableNotFoundError(DocumentParseError):
    """Ошибка разбора структуры таблицы."""

    message = "В документе не найдены таблицы"


class TableParseError(DocumentParseError):
    """Ошибка разбора структуры таблицы."""

    message = "Ошибка разбора структуры таблицы: {details}"


class ParagraphNotFoundError(DocumentParseError):
    """Ошибка разбора структуры документа: не найдено ни одного абзаца."""

    message = "В документе не найдено ни одного абзаца"


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


class RecognitionQualityError(OcrError):
    """Качество распознавания OCR ниже допустимого порога."""

    message = "Качество распознавания OCR слишком низкое: средняя уверенность {mean_confidence:.2f}"


class OcrEmptyResultError(OcrError):
    """OCR не вернул результатов для страницы."""

    message = "OCR не вернул результатов: страница {page_number}"
