from enum import Enum

from pydantic import BaseModel

from .bbox import BBox


class ParagraphType(Enum):
    """
    Перечисление типов структурных элементов, обнаруженных на странице.
    """

    UNKNOWN = "Неизвестный тип"
    PAGE_HEADER = "Верхний колонтитул"  # Элементы в верхней части страницы (Например, название документа, дата)
    PAGE_FOOTER = "Нижний колонтитул"  # Элементы в нижней части страницы (Например, номера страниц)
    SECTION_TITLE = "Заголовок раздела"  # Главный заголовок конкретного раздела
    TABLE_CAPTION = "Подпись таблицы"
    BODY_TEXT = "Основной текст"

    # Можно добавить более гранулярные типы, если потребуется:
    # SUBHEADER = "Подзаголовок"
    # PAGE_NUMBER = "Номер страницы"


class Paragraph(BaseModel):
    id: str
    text: str
    type: ParagraphType = ParagraphType.UNKNOWN
    bbox: BBox
    blobs: list[BBox] = []  # bbox каждой OCR-строки внутри абзаца

    @property
    def area(self) -> float:
        return self.bbox.area

    @property
    def width(self) -> float:
        return self.bbox.width

    @property
    def height(self) -> float:
        return self.bbox.height

    @property
    def center(self) -> tuple[float, float]:
        cx = (self.bbox.x_min + self.bbox.x_max) / 2
        cy = (self.bbox.y_min + self.bbox.y_max) / 2
        return (cx, cy)

    @property
    def is_empty(self) -> bool:
        return self.text is None or self.text.strip() == ""

    def is_same_line(self, other: "Paragraph", row_height: float) -> bool:
        """True если абзацы на одной вертикали и не дальше row_height друг от друга"""
        self_left_align = abs(self.bbox.x_min - other.bbox.x_min) < (row_height * 0.5)
        vertical_proximity = abs(self.bbox.y_min - other.bbox.y_min) <= row_height
        return self_left_align and vertical_proximity

    def padding(self, pixel: float) -> "Paragraph":
        return Paragraph(
            id=self.id,
            text=self.text,
            bbox=self.bbox.padding(pixel),
            blobs=[b.padding(pixel) for b in self.blobs],
            type=self.type,
        )

    def intersect(self, other: "Paragraph") -> float:
        return self.bbox.intersect(other.bbox)

    def is_at_top(self, page_height: float, threshold: float = 0.15) -> bool:
        return self.bbox.y_min < (page_height * threshold)

    def is_at_bottom(self, page_height: float, threshold: float = 0.15) -> bool:
        return self.bbox.y_max > (page_height * (1 - threshold))

    def is_centered(self, page_width: float, margin: float = 0.2) -> bool:
        cx = (self.bbox.x_min + self.bbox.x_max) / 2
        return abs(cx - page_width / 2) < page_width * margin

    def classify_by_position(
        self,
        page_shape: tuple,
        tables: list[BBox] | None,
        median_height_row: float,
    ) -> "Paragraph":
        h = page_shape[0]
        w = page_shape[1]
        # 1. Проверка колонтитулов (Приоритет 1)
        if self.is_at_top(h, 0.1):
            return Paragraph(**{**self.model_dump(), "type": ParagraphType.PAGE_HEADER})
        elif self.is_at_bottom(h, 0.1):
            return Paragraph(**{**self.model_dump(), "type": ParagraphType.PAGE_FOOTER})

        # 2. Проверка подписи таблицы (Приоритет 2)
        # Проверяем, находится ли блок прямо над областью, помеченной как подпись
        if tables:
            for caption_bbox in tables:
                if abs(self.bbox.y_max - caption_bbox.y_min) <= median_height_row:
                    return Paragraph(**{**self.model_dump(), "type": ParagraphType.TABLE_CAPTION})

        # 3. Проверка заголовка раздела (Приоритет 3)
        if self.is_centered(w, 0.1):
            # Здесь может понадобиться проверка на уникальность текста или жирность, но пока оставим по позиции
            return Paragraph(**{**self.model_dump(), "type": ParagraphType.SECTION_TITLE})

        # 4. Основной текст (По умолчанию)
        return Paragraph(**{**self.model_dump(), "type": ParagraphType.BODY_TEXT})

    def to_markdown(self) -> str:
        text = self.text.strip()
        if not text:
            return ""
        if self.type in (ParagraphType.PAGE_HEADER, ParagraphType.PAGE_FOOTER):
            return f"<small>{text}</small>"
        if self.type == ParagraphType.SECTION_TITLE:
            return f"**{text}**"
        if self.type == ParagraphType.TABLE_CAPTION:
            return f"*{text}*"
        return text

    def merge_with(self, other: "Paragraph") -> "Paragraph":
        merged_bbox = BBox(
            x_min=min(self.bbox.x_min, other.bbox.x_min),
            y_min=min(self.bbox.y_min, other.bbox.y_min),
            x_max=max(self.bbox.x_max, other.bbox.x_max),
            y_max=max(self.bbox.y_max, other.bbox.y_max),
        )
        return Paragraph(
            id=f"{self.id}+{other.id}",
            text=f"{self.text} {other.text}".strip(),
            type=self.type,
            bbox=merged_bbox,
            blobs=self.blobs + other.blobs,
        )
