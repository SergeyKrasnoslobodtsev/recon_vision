"""Заполняет ячейки таблиц распознанным текстом."""

from __future__ import annotations

from loguru import logger

from vision_core.entities.bbox import BBox
from vision_core.entities.table import Table
from vision_core.ocr.base import OcrResult


class CellTextFiller:
    """Заполняет ячейки таблиц текстом из OCR-результатов.

    Использует геометрическое вхождение центра OCR-bbox в bbox ячейки.
    """

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        self.confidence_threshold = confidence_threshold

    def fill_cells(self, tables: list[Table], ocr_results: list[OcrResult]) -> None:
        """Заполняет ячейки таблиц текстом in-place.

        Args:
            tables: Список таблиц для заполнения.
            ocr_results: Результаты OCR со всего изображения.
        """
        if not tables:
            logger.warning("Нет таблиц для заполнения, пропускаем этап cell text filling.")
            return

        for table in tables:
            logger.debug(f"Заполнение ячеек для таблицы {table.id} с {len(table.cells)} ячейками...")
            logger.debug("=" * 40)
            for cell in table.cells:
                texts: list[str] = []
                for ocr_item in ocr_results:
                    if ocr_item.confidence < self.confidence_threshold:
                        continue
                    if not cell.bbox.contains_center(BBox.from_tuple(ocr_item.bbox)):
                        continue
                    texts.append(ocr_item.text)
                    cell.blobs.append(BBox.from_tuple(ocr_item.bbox))
                cell.value = "\n".join(texts)
                logger.debug(f"R:{cell.row} C:{cell.col} - {cell.value}")
            logger.debug("=" * 40)
            logger.debug(f"Завершено заполнение таблицы {table.id}.")

    def exclude_table_text(
        self,
        ocr_results: list[OcrResult],
        tables: list[Table],
    ) -> list[OcrResult]:
        """Возвращает OCR-результаты, не попадающие ни в одну таблицу.

        Args:
            ocr_results: Все OCR-результаты страницы.
            tables: Список таблиц для исключения.

        Returns:
            Отфильтрованный список OCR-результатов.
        """
        if not tables:
            logger.warning("Нет таблиц для исключения, пропускаем этап exclude table text.")
            return ocr_results

        return [
            item
            for item in ocr_results
            if not any(table.bbox.contains_center(BBox.from_tuple(item.bbox)) for table in tables)
        ]
