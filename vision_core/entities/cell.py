from pydantic import BaseModel
from .bbox import BBox


class Cell(BaseModel):
    row: int
    col: int
    colspan: int = 1
    rowspan: int = 1
    value: str | None = None
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

    def get_largest_free_space(self, padding: float = 2.0) -> BBox | None:
        """Возвращает наибольшее свободное пространство в ячейке"""
        if not self.blobs:
            return self.bbox

        # Ищем максимальное свободное пространство между blobs
        max_free_area = 0
        best_bbox = None

        # Справа от последнего blob
        rightmost = max(self.blobs, key=lambda b: b.x_max)
        right_space = BBox(
            x_min=rightmost.x_max + padding,
            y_min=self.bbox.y_min,
            x_max=self.bbox.x_max,
            y_max=self.bbox.y_max,
        )
        if right_space.width > 0 and right_space.area > max_free_area:
            max_free_area = right_space.area
            best_bbox = right_space

        # Снизу от последнего blob
        bottommost = max(self.blobs, key=lambda b: b.y_max)
        bottom_space = BBox(
            x_min=self.bbox.x_min,
            y_min=bottommost.y_max + padding,
            x_max=self.bbox.x_max,
            y_max=self.bbox.y_max,
        )
        if bottom_space.height > 0 and bottom_space.area > max_free_area:
            best_bbox = bottom_space

        return best_bbox

    def __str__(self) -> str:
        return f"Cell(row={self.row}, col={self.col}, colspan={self.colspan}, rowspan={self.rowspan}, value={self.value}, bbox={self.bbox}, blobs={self.blobs})"
