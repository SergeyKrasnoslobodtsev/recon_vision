from pydantic import BaseModel
from .bbox import BBox
from typing import Union, Optional
from datetime import datetime


class Cell(BaseModel):
    row: int
    col: int
    colspan: int = 1
    rowspan: int = 1
    value: Optional[Union[str, int, float, datetime]] = None
    bbox: BBox
    blobs: list[BBox] = []

    @property
    def area(self) -> float:
        """Площадь ячейки"""
        return self.bbox.area * self.colspan * self.rowspan

    @property
    def width(self) -> float:
        """Ширина ячейки"""
        return self.bbox.width * self.colspan

    @property
    def height(self) -> float:
        """Высота ячейки"""
        return self.bbox.height * self.rowspan

    @property
    def center(self) -> tuple[float, float]:
        """Центр ячейки (x, y)"""
        center_x = (self.bbox.x_min + self.bbox.x_max) / 2
        center_y = (self.bbox.y_min + self.bbox.y_max) / 2
        return (center_x, center_y)

    @property
    def is_merged(self) -> bool:
        """Является ли ячейка объединенной (с colspan или rowspan > 1)"""
        return self.colspan > 1 or self.rowspan > 1

    @property
    def is_empty(self) -> bool:
        """Пустая ли ячейка (без значения и без blob'ов)"""
        return self.value is None or self.value.strip() == ""

    def padding(self, pixel: float) -> "Cell":
        """Возвращает Cell с добавленным отступом в пикселях"""
        return Cell(
            row=self.row,
            col=self.col,
            colspan=self.colspan,
            rowspan=self.rowspan,
            value=self.value,
            bbox=self.bbox.padding(pixel),
            blobs=[blob.padding(pixel) for blob in self.blobs],
        )

    def intersect(self, other: "Cell") -> float:
        """Площадь пересечения двух ячеек"""
        return self.bbox.intersect(other.bbox)

    def get_largest_free_space(self, padding: float = 1.0) -> BBox | None:
        """Возвращает наибольшее свободное пространство в ячейке"""
        if not self.blobs:
            return self.bbox

        free_spaces = []

        for blob in self.blobs:
            # Справа
            if blob.x_max + padding < self.bbox.x_max:
                free_spaces.append(
                    BBox(
                        x_min=blob.x_max + padding,
                        y_min=self.bbox.y_min,
                        x_max=self.bbox.x_max,
                        y_max=self.bbox.y_max,
                    )
                )
            # Слева
            if blob.x_min - padding > self.bbox.x_min:
                free_spaces.append(
                    BBox(
                        x_min=self.bbox.x_min,
                        y_min=self.bbox.y_min,
                        x_max=blob.x_min - padding,
                        y_max=self.bbox.y_max,
                    )
                )
            # Снизу
            if blob.y_max + padding < self.bbox.y_max:
                free_spaces.append(
                    BBox(
                        x_min=self.bbox.x_min,
                        y_min=blob.y_max + padding,
                        x_max=self.bbox.x_max,
                        y_max=self.bbox.y_max,
                    )
                )
            # Сверху
            if blob.y_min - padding > self.bbox.y_min:
                free_spaces.append(
                    BBox(
                        x_min=self.bbox.x_min,
                        y_min=self.bbox.y_min,
                        x_max=self.bbox.x_max,
                        y_max=blob.y_min - padding,
                    )
                )

        return max(free_spaces, key=lambda b: b.area) if free_spaces else None

    def __str__(self) -> str:
        return f"Cell(row={self.row}, col={self.col}, colspan={self.colspan}, rowspan={self.rowspan}, value={self.value}, bbox={self.bbox}, blobs={self.blobs})"
