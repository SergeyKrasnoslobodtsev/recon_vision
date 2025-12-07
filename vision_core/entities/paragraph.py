from pydantic import BaseModel
import enum
from .bbox import BBox


class ParagraphType(enum.Enum):
    HEADER = 0
    FOOTER = 1
    NONE = 2


class Paragraph(BaseModel):
    id: str
    text: str
    type: ParagraphType = ParagraphType.NONE
    bbox: BBox
    blobs: list[BBox] = []

    @property
    def area(self) -> float:
        """Площадь параграфа"""
        return self.bbox.area

    @property
    def width(self) -> float:
        """Ширина параграфа"""
        return self.bbox.width

    @property
    def height(self) -> float:
        """Высота параграфа"""
        return self.bbox.height

    @property
    def center(self) -> tuple[float, float]:
        """Центр параграфа (x, y)"""
        center_x = (self.bbox.x_min + self.bbox.x_max) / 2
        center_y = (self.bbox.y_min + self.bbox.y_max) / 2
        return (center_x, center_y)

    @property
    def is_empty(self) -> bool:
        """Пустой ли параграф (без текста)"""
        return self.text is None or self.text.strip() == ""

    def padding(self, pixel: float) -> "Paragraph":
        """Возвращает Paragraph с добавленным отступом в пикселях"""
        return Paragraph(
            id=self.id,
            text=self.text,
            bbox=self.bbox.padding(pixel),
            blobs=[blob.padding(pixel) for blob in self.blobs],
        )

    def intersect(self, other: "Paragraph") -> float:
        """Площадь пересечения двух параграфов"""
        return self.bbox.intersect(other.bbox)

    def is_at_top(self, page_height: float, threshold: float = 0.15) -> bool:
        """Проверяет, находится ли параграф в верхней части страницы (Header)"""
        return self.bbox.y_min < (page_height * threshold)

    def is_at_bottom(self, page_height: float, threshold: float = 0.15) -> bool:
        """Проверяет, находится ли параграф в нижней части страницы (Footer)"""
        return self.bbox.y_max > (page_height * (1 - threshold))

    def classify_by_position(self, page_height: float) -> "Paragraph":
        """Автоматически классифицирует параграф как HEADER/FOOTER/NONE"""
        if self.is_at_top(page_height):
            return Paragraph(**{**self.model_dump(), "type": ParagraphType.HEADER})
        elif self.is_at_bottom(page_height):
            return Paragraph(**{**self.model_dump(), "type": ParagraphType.FOOTER})
        return self

    def merge_with(self, other: "Paragraph") -> "Paragraph":
        """Объединяет два параграфа в один"""
        # Объединяем bbox
        merged_bbox = BBox(
            x_min=min(self.bbox.x_min, other.bbox.x_min),
            y_min=min(self.bbox.y_min, other.bbox.y_min),
            x_max=max(self.bbox.x_max, other.bbox.x_max),
            y_max=max(self.bbox.y_max, other.bbox.y_max),
        )

        # Объединяем текст
        merged_text = f"{self.text} {other.text}".strip()

        # Объединяем blobs
        merged_blobs = self.blobs + other.blobs

        return Paragraph(
            id=f"{self.id}+{other.id}",
            text=merged_text,
            type=self.type,
            bbox=merged_bbox,
            blobs=merged_blobs,
        )
