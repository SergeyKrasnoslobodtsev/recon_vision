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

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


class PdfLoader:
    """Загружает PDF и предоставляет страницы как PIL-изображения."""

    def __init__(self, pdf_bytes: bytes):
        self.doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    def __enter__(self) -> PdfLoader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.doc.close()

    def get_page_image(self, page_number: int, dpi: int) -> Image.Image:
        """Рендерит страницу PDF в PIL-изображение.

        Args:
            page_number: Номер страницы.
            dpi: Разрешение рендера.

        Returns:
            Image.Image: Изображение страницы.
        """
        return self.doc[page_number].get_pixmap(dpi=dpi, alpha=False).pil_image()


def load_aligned_page_images(pdf_bytes: bytes, document: Document) -> list[Image.Image]:
    """Загружает и выравнивает страницы PDF в координатах документа.

    Args:
        pdf_bytes: Исходный PDF.
        document: Канонический документ.

    Returns:
        list[Image.Image]: Выровненные страницы.
    """
    dpi = int(document.metadata.get("dpi") or 300)
    aligned_pages: list[Image.Image] = []

    with PdfLoader(pdf_bytes) as pdf_loader:
        for page in document.pages:
            source_image = pdf_loader.get_page_image(page.page_number, dpi)
            aligned_image = align_page_image(source_image, page)
            canvas_image = resize_page_image_to_document(aligned_image, page)
            logger.debug(
                f"pdf_filler: page={page.page_number} dpi={dpi} "
                f"source={source_image.width}x{source_image.height} "
                f"aligned={aligned_image.width}x{aligned_image.height} "
                f"canvas={canvas_image.width}x{canvas_image.height}"
            )
            aligned_pages.append(canvas_image)

    return aligned_pages


def align_page_image(page_image: Image.Image, page: Page) -> Image.Image:
    """Корректирует ориентацию и наклон страницы.

    Args:
        page_image: Исходное изображение страницы.
        page: Каноническая страница.

    Returns:
        Image.Image: Выровненное изображение.
    """
    orientation_deg = int(round(float(page.metadata.get("orientation_deg") or 0.0))) % 360
    deskew_angle_deg = float(page.metadata.get("deskew_angle_deg") or 0.0)
    image_array = np.array(page_image)
    image_array = rotate_image(image_array, orientation_deg)
    image_array = rotate_image(image_array, deskew_angle_deg)
    return Image.fromarray(image_array)


def resize_page_image_to_document(page_image: Image.Image, page: Page) -> Image.Image:
    """Приводит изображение к каноническому размеру страницы.

    Args:
        page_image: Выровненное изображение.
        page: Каноническая страница.

    Returns:
        Image.Image: Изображение в координатах документа.
    """
    width, height = resolve_page_canvas_size(page)
    if (page_image.width, page_image.height) == (width, height):
        return page_image
    return page_image.resize((width, height), Image.Resampling.LANCZOS)


def resolve_page_canvas_size(page: Page) -> tuple[int, int]:
    """Возвращает размер страницы в координатах документа.

    Args:
        page: Каноническая страница.

    Returns:
        tuple[int, int]: Ширина и высота страницы.

    Raises:
        ValueError: Если размер страницы не найден в metadata.
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


def draw_centered_text(
    page_image: Image.Image,
    cell: Cell,
    text: str,
    font_size: int,
    font_path: str | None,
) -> None:
    """Рисует текст по центру ячейки.

    Args:
        page_image: Изображение страницы.
        cell: Целевая ячейка.
        text: Рисуемый текст.
        font_size: Размер шрифта.
        font_path: Путь к системному шрифту.
    """
    draw = ImageDraw.Draw(page_image)
    font = load_font(font_size, font_path)
    text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), text, font=font)
    text_width = text_right - text_left
    text_height = text_bottom - text_top
    x_pos = cell.bbox.x_min + (cell.width - text_width) / 2 - text_left
    y_pos = cell.bbox.y_min + (cell.height - text_height) / 2 - text_top
    draw.text((x_pos, y_pos), text, fill=(0, 0, 0), font=font)


def draw_comments(
    page_images: dict[int, Image.Image], document: Document, comments: str | None, font_path: str | None
) -> None:
    """Рисует комментарии в правом нижнем углу последней страницы.

    Args:
        page_images: Изображения страниц.
        document: Канонический документ.
        comments: Текст комментария.
        font_path: Путь к системному шрифту.
    """
    if not comments or not document.pages:
        return

    page_number = _resolve_comments_page_number(document)
    page_image = page_images[page_number]
    draw = ImageDraw.Draw(page_image)
    font = load_font(24, font_path)
    text = comments
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.multiline_text(
        (page_image.width - text_width - 12, page_image.height - text_height - 12),
        text,
        fill="blue",
        font=font,
    )


def _resolve_comments_page_number(document: Document) -> int:
    """Возвращает страницу, на которой нужно рисовать комментарий.

    Args:
        document: Канонический документ.

    Returns:
        int: Номер страницы с последней таблицей или последней страницы документа.
    """
    last_table_page_number: int | None = None
    for page in document.pages:
        if page.tables:
            last_table_page_number = page.page_number

    if last_table_page_number is not None:
        return last_table_page_number

    return document.pages[-1].page_number


def save_page_images_as_pdf(page_images: list[Image.Image]) -> bytes:
    """Сохраняет PIL-страницы в PDF.

    Args:
        page_images: Страницы документа.

    Returns:
        bytes: PDF в байтах.
    """
    if not page_images:
        return b""

    buffer = BytesIO()
    rgb_pages = [page.convert("RGB") for page in page_images]
    rgb_pages[0].save(buffer, format="PDF", save_all=True, append_images=rgb_pages[1:], resolution=72.0)
    return buffer.getvalue()


def resolve_font_file() -> str | None:
    """Возвращает доступный системный шрифт.

    Returns:
        str | None: Путь к шрифту или None.
    """
    for font_path in _FONT_CANDIDATES:
        if Path(font_path).exists():
            return font_path
    return None


def load_font(font_size: int, font_path: str | None) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Загружает шрифт заданного размера.

    Args:
        font_size: Размер шрифта.
        font_path: Путь к системному шрифту.

    Returns:
        ImageFont.ImageFont | ImageFont.FreeTypeFont: Шрифт для рисования.
    """
    if font_path is not None:
        return ImageFont.truetype(font_path, size=font_size)
    return ImageFont.load_default()
