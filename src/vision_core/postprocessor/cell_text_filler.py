"""Заполняет ячейки таблиц распознанным текстом."""

from __future__ import annotations

from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.table import Table
from vision_core.ocr.base import OcrResult


class CellTextFiller:
    """Заполняет ячейки таблиц текстом из OCR-результатов.

    Сначала пытается привязать OCR-блок к ячейке по геометрическому overlap.
    Если overlap недостаточен, использует резервную эвристику с центром bbox.
    Неоднозначные блоки, которые сопоставляются сразу с несколькими ячейками,
    намеренно не записываются автоматически, чтобы не вносить неверный текст.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        cell_overlap_threshold: float = 0.5,
        table_overlap_threshold: float = 0.4,  # TODO: Вынести в конфиг
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.cell_overlap_threshold = cell_overlap_threshold
        self.table_overlap_threshold = table_overlap_threshold

    def fill_cells(self, tables: list[Table], ocr_results: list[OcrResult]) -> None:
        """Заполняет ячейки таблиц текстом in-place.

        Args:
            tables: Список таблиц для заполнения.
            ocr_results: Результаты OCR со всего изображения.
        """
        for table in tables:
            for cell in table.cells:
                cell.value = None
                cell.blobs.clear()

            texts_by_cell: dict[int, list[str]] = {id(cell): [] for cell in table.cells}
            blobs_by_cell: dict[int, list[BBox]] = {id(cell): [] for cell in table.cells}

            for ocr_item in ocr_results:
                if ocr_item.confidence < self.confidence_threshold:
                    continue

                ocr_bbox = BBox.from_tuple(ocr_item.bbox)
                target_cell = self._select_cell(table.cells, ocr_bbox)
                if target_cell is None:
                    continue

                texts_by_cell[id(target_cell)].append(ocr_item.text)
                blobs_by_cell[id(target_cell)].append(ocr_bbox)

            for cell in table.cells:
                cell.value = "\n".join(texts_by_cell[id(cell)])
                cell.blobs.extend(blobs_by_cell[id(cell)])

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
            item
            for item in ocr_results
            if not any(self._belongs_to_table(BBox.from_tuple(item.bbox), table) for table in tables)
        ]

    def _select_cell(self, cells: list[Cell], ocr_bbox: BBox) -> Cell | None:
        """Выбирает ячейку для OCR-блока или возвращает None для спорного случая.

        Args:
            cells: Кандидатные ячейки таблицы.
            ocr_bbox: BBox OCR-блока.

        Returns:
            Cell | None: Однозначно выбранная ячейка или None.
        """
        overlap_candidates = [(cell.bbox.intersection_over_min(ocr_bbox), cell) for cell in cells]
        strong_candidates = [
            (score, cell) for score, cell in overlap_candidates if score >= self.cell_overlap_threshold
        ]

        if len(strong_candidates) == 1:
            return strong_candidates[0][1]
        if len(strong_candidates) > 1:
            return None

        center_matches = [cell for cell in cells if cell.bbox.contains_center(ocr_bbox)]
        if len(center_matches) == 1:
            return center_matches[0]

        return None

    def _belongs_to_table(self, ocr_bbox: BBox, table: Table) -> bool:
        """Проверяет, относится ли OCR-блок к области таблицы.

        Args:
            ocr_bbox: BBox OCR-блока.
            table: Таблица-кандидат.

        Returns:
            bool: True, если блок относится к таблице.
        """
        return (
            table.bbox.contains_center(ocr_bbox)
            or table.bbox.intersection_over_min(ocr_bbox) >= self.table_overlap_threshold
        )
