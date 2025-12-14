from pydantic import BaseModel
from functools import cached_property
from .cell import Cell
from .bbox import BBox


class Table(BaseModel):
    id: str
    bbox: BBox
    num_rows: int
    num_cols: int
    cells: list["Cell"] = []
    start_page: int
    end_page: int

    @property
    def area(self) -> float:
        """Площадь таблицы"""
        return self.bbox.area

    @property
    def width(self) -> float:
        """Ширина таблицы"""
        return self.bbox.width

    @property
    def height(self) -> float:
        """Высота таблицы"""
        return self.bbox.height

    @property
    def center(self) -> tuple[float, float]:
        """Центр таблицы (x, y)"""
        center_x = (self.bbox.x_min + self.bbox.x_max) / 2
        center_y = (self.bbox.y_min + self.bbox.y_max) / 2
        return (center_x, center_y)

    def padding(self, pixel: float) -> "Table":
        """Возвращает Table с добавленным отступом в пикселях"""
        return Table(
            id=self.id,
            bbox=self.bbox.padding(pixel),
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            cells=[cell.padding(pixel) for cell in self.cells],
            start_page=self.start_page,
            end_page=self.end_page,
        )

    def intersect(self, other: "Table") -> float:
        """Площадь пересечения двух таблиц"""
        return self.bbox.intersect(other.bbox)

    @cached_property
    def _rows_index(self) -> list[list[Cell]]:
        """Индекс строк для быстрого доступа"""
        rows = [[] for _ in range(self.num_rows)]
        for cell in self.cells:
            rows[cell.row].append(cell)
        return rows

    @cached_property
    def _cols_index(self) -> list[list[Cell]]:
        """Индекс столбцов для быстрого доступа"""
        cols = [[] for _ in range(self.num_cols)]
        for cell in self.cells:
            cols[cell.col].append(cell)
        return cols

    @cached_property
    def _cell_index(self) -> dict[tuple[int, int], Cell]:
        return {(c.row, c.col): c for c in self.cells}

    def get_cell(self, row: int, col: int) -> Cell | None:
        """Получить ячейку по номеру строки и столбца"""
        return self._cell_index.get((row, col))

    def get_rows(self, include_merged: bool = False) -> list[list[Cell]]:
        if not include_merged:
            return self._rows_index.copy()

        # Для включения объединенных ячеек
        rows = [[] for _ in range(self.num_rows)]
        for cell in self.cells:
            for r in range(cell.row, min(cell.row + cell.rowspan, self.num_rows)):
                rows[r].append(cell)
        return rows

    def get_cols(self, include_merged: bool = False) -> list[list[Cell]]:
        if not include_merged:
            return self._cols_index.copy()

        # Для включения объединенных ячеек
        cols = [[] for _ in range(self.num_cols)]
        for cell in self.cells:
            for c in range(cell.col, min(cell.col + cell.colspan, self.num_cols)):
                cols[c].append(cell)
        return cols

    def median_height_blobs_per_cells(self) -> float:
        """Медианная высота blob'ов на ячейку"""
        heights = []
        for cell in self.cells:
            if cell.blobs:
                total_height = sum(blob.height for blob in cell.blobs)
                heights.append(total_height / len(cell.blobs))
        if not heights:
            return 0.0
        heights.sort()
        mid = len(heights) // 2
        if len(heights) % 2 == 0:
            return (heights[mid - 1] + heights[mid]) / 2
        else:
            return heights[mid]

    def is_valid(self) -> bool:
        """Проверяет, что таблица имеет больше одной ячейки и положительные размеры"""
        return self.num_rows > 1 and self.num_cols > 1 and len(self.cells) > 1

    def validate_structure(self) -> bool:
        """Детальная проверка: соответствуют ли ячейки размерам таблицы"""
        for cell in self.cells:
            if cell.row >= self.num_rows or cell.col >= self.num_cols:
                return False
            if cell.row + cell.rowspan > self.num_rows:
                return False
            if cell.col + cell.colspan > self.num_cols:
                return False
        return True

    def __str__(self) -> str:
        return f"Table(id={self.id}, rows={self.num_rows}, cols={self.num_cols}, pages={self.start_page}-{self.end_page})"
