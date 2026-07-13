"""Рендерит заполненный PDF из канонических страниц документа."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pymupdf
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from vision_core.entities.cell import Cell
from vision_core.entities.document import Document
from vision_core.entities.page import Page
from vision_core.utils.image_utils import rotate_image

# Путь от корня проекта, независимо от текущей рабочей директории
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
_FONT = _PROJECT_ROOT / "assets" / "fonts" / "LiberationSerif-Regular.ttf"
_COMMENTS_FONT_SIZE = 24


class PdfLoader:
    """Загружает PDF и предоставляет страницы как PIL-изображения."""

    def __init__(self, pdf_bytes: bytes):
        self.doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    def __enter__(self) -> PdfLoader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.doc.close()

    def get_page_image(self, page_number: int, dpi: int) -> Image.Image:
        return self.doc[page_number].get_pixmap(dpi=dpi, alpha=False).pil_image()


def load_aligned_page_images(pdf_bytes: bytes, document: Document) -> list[Image.Image]:
    dpi = int(document.metadata.get("dpi") or 300)
    aligned_pages: list[Image.Image] = []

    with PdfLoader(pdf_bytes) as pdf_loader:
        for page in document.pages:
            source_image = pdf_loader.get_page_image(page.page_number, dpi)
            aligned_image = align_page_image(source_image, page)
            canvas_image = resize_page_image_to_document(aligned_image, page)
            logger.debug(
                f"page={page.page_number} dpi={dpi} "
                f"source={source_image.width}x{source_image.height} "
                f"aligned={aligned_image.width}x{aligned_image.height} "
                f"canvas={canvas_image.width}x{canvas_image.height}"
            )
            aligned_pages.append(canvas_image)

    return aligned_pages


def align_page_image(page_image: Image.Image, page: Page) -> Image.Image:
    orientation_deg = int(round(float(page.metadata.get("orientation_deg") or 0.0))) % 360
    deskew_angle_deg = float(page.metadata.get("deskew_angle_deg") or 0.0)
    image_array = np.array(page_image)
    image_array = rotate_image(image_array, orientation_deg)
    image_array = rotate_image(image_array, deskew_angle_deg)
    return Image.fromarray(image_array)


def resize_page_image_to_document(page_image: Image.Image, page: Page) -> Image.Image:
    width, height = resolve_page_canvas_size(page)
    if (page_image.width, page_image.height) == (width, height):
        return page_image
    return page_image.resize((width, height), Image.Resampling.LANCZOS)


def resolve_page_canvas_size(page: Page) -> tuple[int, int]:
    """Возвращает размер страницы в координатах документа.

    Проверяет три источника: image_shape, page_size, source_image_shape.
    Поднимает ValueError если ни один не найден в metadata.
    """
    image_shape = page.metadata.get("image_shape")
    if isinstance(image_shape, (list, tuple)) and len(image_shape) == 2:
        return int(image_shape[1]), int(image_shape[0])

    page_size = page.metadata.get("page_size")
    if isinstance(page_size, (list, tuple)) and len(page_size) == 2:
        return int(page_size[0]), int(page_size[1])

    source_shape = page.metadata.get("source_image_shape")
    if isinstance(source_shape, (list, tuple)) and len(source_shape) == 2:
        return int(source_shape[1]), int(source_shape[0])

    raise ValueError(f"Не найден размер страницы для page={page.page_number}")


class PageRenderer:
    """Управляет рисованием на одной странице с кешированием шрифтов."""

    def __init__(self, page_image: Image.Image, font_path: str | None):
        self.page_image = page_image
        self.font_path = font_path
        self.draw = ImageDraw.Draw(page_image)
        self._font_cache: dict[int, ImageFont.ImageFont | ImageFont.FreeTypeFont] = {}

    def _get_font(self, font_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        if font_size not in self._font_cache:
            self._font_cache[font_size] = load_font(font_size, self.font_path)
        return self._font_cache[font_size]

    def _draw_strikethrough(self, cell: Cell) -> None:
        """Зачеркивает существующий текст (горизонтальная линия через каждый blob)."""
        for blob in cell.blobs:
            y_middle = (blob.y_min + blob.y_max) / 2
            self.draw.line(
                [(blob.x_min, y_middle), (blob.x_max, y_middle)],
                fill=(0, 0, 0),
                width=1,
            )

    def _get_free_space_with_priority(self, cell: Cell) -> tuple[float, float]:
        """Находит свободное место для размещения текста и возвращает (x, y, direction)."""
        if not cell.blobs:
            return (cell.center[0], cell.center[1], "center")

        padding = 3.0

        # Вычисляем свободное место в каждом направлении
        max_blob_x = max(blob.x_max for blob in cell.blobs)
        min_blob_x = min(blob.x_min for blob in cell.blobs)
        min_blob_y = min(blob.y_min for blob in cell.blobs)

        space_right = cell.bbox.x_max - max_blob_x
        space_left = min_blob_x - cell.bbox.x_min
        space_top = min_blob_y - cell.bbox.y_min

        # Находим направление с наибольшим свободным местом
        spaces = [
            ("top", space_top),
            ("right", space_right),
            ("left", space_left),
        ]
        direction, _ = max(spaces, key=lambda x: x[1])

        # Вычисляем координаты для размещения текста
        cell_center_x = (cell.bbox.x_min + cell.bbox.x_max) / 2
        cell_center_y = (cell.bbox.y_min + cell.bbox.y_max) / 2

        if direction == "top":
            x = cell_center_x  # По середине по X
            y = (cell.bbox.y_min + min_blob_y) / 2  # По середине между верхней границей и текстом
        elif direction == "right":
            x = max_blob_x + padding  # С отступом от текста
            y = cell_center_y
        else:  # left
            x = cell.bbox.x_min + padding  # С отступом от левой границы ячейки
            y = cell_center_y

        return (x, y, direction)

    def draw_centered_text(self, cell: Cell, text: str, font_size: int) -> None:
        """Рисует текст в ячейке. Если есть существующий текст, зачеркивает его и размещает новый в свободное место."""
        font = self._get_font(font_size)

        # Если есть существующий текст (blobs), зачеркиваем его
        if cell.blobs:
            self._draw_strikethrough(cell)
            x, y, direction = self._get_free_space_with_priority(cell)
        else:
            x, y, direction = cell.center[0], cell.center[1], "center"

        text_left, text_top, text_right, text_bottom = self.draw.textbbox((0, 0), text, font=font)
        text_width = text_right - text_left
        text_height = text_bottom - text_top

        if direction == "center":
            x_pos = x - text_width / 2 - text_left
            y_pos = y - text_height / 2 - text_top
        elif direction == "top":
            # Центрировано по X, вертикально в верхнем пространстве
            x_pos = x - text_width / 2 - text_left
            y_pos = y - text_height / 2 - text_top
        elif direction == "left":
            # Слева с отступом, центрировано вертикально
            x_pos = x - text_left
            y_pos = y - text_height / 2 - text_top
        else:  # right
            # Справа с отступом, центрировано вертикально
            x_pos = x - text_left
            y_pos = y - text_height / 2 - text_top

        self.draw.text((x_pos, y_pos), text, fill=(0, 0, 0), font=font)

    def draw_comments(self, comments: str) -> None:
        font = self._get_font(_COMMENTS_FONT_SIZE)
        x0, y0, x1, y1 = self.draw.multiline_textbbox((0, 0), comments, font=font)
        text_width, text_height = x1 - x0, y1 - y0
        self.draw.multiline_text(
            (self.page_image.width - text_width - 12, self.page_image.height - text_height - 12),
            comments,
            fill="blue",
            font=font,
        )


def save_page_images_as_pdf(page_images: list[Image.Image]) -> bytes:
    if not page_images:
        return b""
    buffer = BytesIO()
    rgb_pages = [page.convert("RGB") for page in page_images]
    rgb_pages[0].save(buffer, format="PDF", save_all=True, append_images=rgb_pages[1:], resolution=72.0)
    return buffer.getvalue()


def format_amount(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def resolve_font_file() -> str | None:
    if _FONT.exists():
        logger.info(f"найден шрифт: {_FONT}")
        return str(_FONT)
    logger.warning(f"шрифт не найден по пути: {_FONT}")
    return None


def load_font(font_size: int, font_path: str | None) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    if font_path is not None:
        logger.debug(f"загружаем шрифт из {font_path} для размера {font_size}")
        return ImageFont.truetype(font_path, size=font_size)
    logger.debug(f"используется шрифт по умолчанию для размера {font_size}")
    return ImageFont.load_default()
