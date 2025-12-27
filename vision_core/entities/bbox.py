from pydantic import BaseModel
import numpy as np


class BBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        """Ширина BBox"""
        # гарантируем неотрицательную ширину (защитa от некорректных координат)
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        """Высота BBox"""
        # гарантируем неотрицательную высоту (защитa от некорректных координат)
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        """Площадь BBox"""
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Возвращает координаты BBox в виде кортежа целых чисел (x_min, y_min, x_max, y_max)"""
        return (int(self.x_min), int(self.y_min), int(self.x_max), int(self.y_max))

    def padding(self, pixel: float) -> "BBox":
        """Возвращает BBox с добавленным отступом в пикселях"""
        return BBox(
            x_min=self.x_min - pixel,
            y_min=self.y_min - pixel,
            x_max=self.x_max + pixel,
            y_max=self.y_max + pixel,
        )

    def margin(self, pixel: float) -> "BBox":
        """Возвращает BBox с уменьшенным отступом в пикселях"""
        return BBox(
            x_min=self.x_min + pixel,
            y_min=self.y_min + pixel,
            x_max=self.x_max - pixel,
            y_max=self.y_max - pixel,
        )

    def intersect(self, other: "BBox") -> float:
        """Площадь пересечения двух BBox"""
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)
        # учитываем вырожденные случаи и возвращаем неотрицательное перекрытие
        x_overlap = max(0.0, x_max - x_min)
        y_overlap = max(0.0, y_max - y_min)
        return x_overlap * y_overlap

    def roi(self, image: np.ndarray) -> np.ndarray:
        """Возвращает вырезанную область изображения по BBox"""
        h, w = image.shape[:2]
        y_min = max(0, int(self.y_min))
        y_max = min(h, int(self.y_max))
        x_min = max(0, int(self.x_min))
        x_max = min(w, int(self.x_max))

        if y_min >= y_max or x_min >= x_max:
            return np.array([], dtype=image.dtype)

        return image[y_min:y_max, x_min:x_max]

    def iou(self, other: "BBox") -> float:
        """Метрика IoU для двух BBox"""
        intersection = self.intersect(other)
        union = self.area + other.area - intersection

        # стандартный случай
        if union > 0.0:
            return intersection / union

        # если union == 0: оба бокса вырожденные (площадь == 0)
        # считаем IoU == 1.0, если боксы совпадают по координатам, иначе 0.0
        eps = 1e-9
        same = (
            abs(self.x_min - other.x_min) <= eps
            and abs(self.y_min - other.y_min) <= eps
            and abs(self.x_max - other.x_max) <= eps
            and abs(self.y_max - other.y_max) <= eps
        )
        return 1.0 if same else 0.0

    def intersection_over_min(self, other: "BBox") -> float:
        """Площадь пересечения / min(площадь self, площадь other).
        Удобно для случаев, когда маленький bbox полностью внутри большого."""
        inter = self.intersect(other)
        min_area = min(self.area, other.area)
        eps = 1e-9
        if min_area > 0.0:
            return inter / min_area
        # если обе площади == 0, считаем совпадение по координатам
        same = (
            abs(self.x_min - other.x_min) <= eps
            and abs(self.y_min - other.y_min) <= eps
            and abs(self.x_max - other.x_max) <= eps
            and abs(self.y_max - other.y_max) <= eps
        )
        return 1.0 if same else 0.0

    def contains(self, other: "BBox", eps: float = 1e-9) -> bool:
        """Полностью ли other внутри self (включая границы)"""
        return (
            other.x_min + eps >= self.x_min
            and other.y_min + eps >= self.y_min
            and other.x_max - eps <= self.x_max
            and other.y_max - eps <= self.y_max
        )

    def contains_center(self, other: "BBox") -> bool:
        """Проверяет, лежит ли центр other внутри self"""
        cx = (other.x_min + other.x_max) / 2.0
        cy = (other.y_min + other.y_max) / 2.0
        return (self.x_min <= cx <= self.x_max) and (self.y_min <= cy <= self.y_max)
