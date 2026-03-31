"""Заполняет ячейки таблиц распознанным текстом."""

from __future__ import annotations

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
        for table in tables:
            for cell in table.cells:
                texts: list[str] = []
                for ocr_item in ocr_results:
                    if ocr_item.confidence < self.confidence_threshold:
                        continue
                    if not cell.bbox.contains_center(BBox.from_tuple(ocr_item.bbox)):
                        continue
                    texts.append(ocr_item.text)
                    cell.blobs.append(ocr_item.bbox)
                cell.value = "\n".join(texts)

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
            return ocr_results

        return [
            item for item in ocr_results
            if not any(
                table.bbox.contains_center(BBox.from_tuple(item.bbox))
                for table in tables
            )
        ]
