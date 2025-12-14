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
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Высота BBox"""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Площадь BBox"""
        return self.width * self.height

    def padding(self, pixel: float) -> "BBox":
        """Возвращает BBox с добавленным отступом в пикселях"""
        return BBox(
            x_min=self.x_min - pixel,
            y_min=self.y_min - pixel,
            x_max=self.x_max + pixel,
            y_max=self.y_max + pixel,
        )

    def intersect(self, other: "BBox") -> float:
        """Площадь пересечения двух BBox"""
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)

        if x_min < x_max and y_min < y_max:
            return (x_max - x_min) * (y_max - y_min)
        return 0.0

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
        return intersection / union if union > 0 else 0.0
