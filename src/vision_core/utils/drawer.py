from enum import Enum
from pathlib import Path
from textwrap import wrap
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Position(Enum):
    TOP = 0
    BOTTOM = 1
    NONE = 2


class Drawer:
    """Рисует debug-структуры поверх изображения или рядом с ним.

    Side-by-side режим создаёт холст вида [source | debug].
    Все отладочные аннотации по умолчанию рисуются на правой части.
    При text_panel_width > 0 справа добавляется отдельная текстовая панель.
    """

    _FONT_CANDIDATES = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    )

    def __init__(
        self,
        image: np.ndarray | Image.Image,
        *,
        side_by_side: bool = False,
        text_panel_width: int = 0,
        blank_color: tuple[int, int, int] = (255, 255, 255),
        draw_on: str = "auto",  # "auto" | "left" | "right" | "both"
    ):
        if isinstance(image, np.ndarray):
            base = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            base = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        if base.mode != "RGB":
            base = base.convert("RGB")

        self._left = base
        self._side_by_side = side_by_side
        self._blank_color = blank_color
        self._text_panel_width = max(0, int(text_panel_width))

        w, h = self._left.size

        if side_by_side:
            right = Image.new("RGB", (w, h), color=blank_color)
            canvas_width = (w * 2) + self._text_panel_width
            canvas = Image.new("RGB", (canvas_width, h), color=blank_color)
            canvas.paste(self._left, (0, 0))
            canvas.paste(right, (w, 0))
            self._canvas = canvas
            self._x_left_offset = 0
            self._x_right_offset = w
            self._text_panel_offset = w * 2
            self._draw_on = "right" if draw_on == "auto" else draw_on
        else:
            self._canvas = self._left
            self._x_left_offset = 0
            self._x_right_offset = 0
            self._text_panel_offset = self._canvas.size[0]
            self._draw_on = "left" if draw_on == "auto" else draw_on

        self._draw = ImageDraw.Draw(self._canvas)
        self._font = None

    def draw_side_by_side_image(self, image: np.ndarray | Image.Image):
        """Вставляет изображение в правую часть side-by-side холста.

        Args:
            image: Изображение для вставки в debug-область справа.

        Raises:
            ValueError: Если режим side_by_side не включён.

        Returns:
            Drawer: Текущий экземпляр для цепочки вызовов.
        """
        if not self._side_by_side:
            raise ValueError("draw_side_by_side_image доступен только в режиме side_by_side=True")

        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize(self._left.size, Image.LANCZOS)
        self._canvas.paste(img, (self._x_right_offset, 0))
        self._draw = ImageDraw.Draw(self._canvas)
        return self

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._canvas.save(path)

    def to_pil(self) -> Image.Image:
        return self._canvas

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._canvas)

    def _get_font(self):
        if self._font is not None:
            return self._font

        for font_path in self._FONT_CANDIDATES:
            if Path(font_path).exists():
                self._font = ImageFont.truetype(font_path, size=14)
                return self._font

        self._font = ImageFont.load_default()
        return self._font

    def _resolve_targets(self, draw_on: str | None = None) -> list[int]:
        mode = self._draw_on
        if draw_on is not None:
            mode = draw_on
        if mode not in {"left", "right", "both"}:
            raise ValueError(f"draw_on must be one of: left/right/both (got {mode!r})")

        if not self._side_by_side:
            return [self._x_left_offset]

        if mode == "both":
            return [self._x_left_offset, self._x_right_offset]
        if mode == "left":
            return [self._x_left_offset]
        return [self._x_right_offset]

    def draw_structure(
        self,
        bbox_xyxy: tuple[int, int, int, int],
        *,
        label: str | None = None,
        color: str | tuple[int, int, int] = "blue",
        width: int = 2,
        position: Position = Position.TOP,
        fill: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
        draw_on: str | None = None,
    ):
        """
        bbox_xyxy: (x1, y1, x2, y2) в координатах ОРИГИНАЛА (левой части).
        В режиме side-by-side рисуем со смещением на правую часть автоматически.
        """
        x1, y1, x2, y2 = map(int, bbox_xyxy)

        for xoff in self._resolve_targets(draw_on):
            xx1, xx2 = x1 + xoff, x2 + xoff
            if fill is not None:
                if len(fill) == 4:
                    self._draw_translucent_rectangle(
                        bbox=(xx1, y1, xx2, y2),
                        outline=color,
                        width=width,
                        fill=fill,
                    )
                else:
                    self._draw.rectangle([xx1, y1, xx2, y2], outline=color, width=width, fill=fill)
            else:
                self._draw.rectangle([xx1, y1, xx2, y2], outline=color, width=width)

            if label:
                font = self._get_font()
                text_w, text_h = self._text_size(label, font)
                pad = 2

                # TOP по умолчанию
                place_top = position is not Position.BOTTOM

                if place_top:
                    tx, ty = xx1, max(0, y1 - text_h - 2 * pad)
                else:
                    tx, ty = xx1, min(self._canvas.size[1] - text_h - 2 * pad, y2 + 2)

                # фон под текст
                self._draw.rectangle(
                    [tx, ty, tx + text_w + 2 * pad, ty + text_h + 2 * pad],
                    fill=(255, 255, 255),
                    outline=color,
                    width=1,
                )
                self._draw.text((tx + pad, ty + pad), label, fill=color, font=font)

        return self

    def draw_boxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        *,
        color: str | tuple[int, int, int] = "blue",
        width: int = 2,
        fill: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
        draw_on: str | None = None,
    ):
        """Рисует набор рамок без подписей.

        Args:
            boxes: Список bbox в формате (x1, y1, x2, y2).
            color: Цвет рамки.
            width: Толщина рамки.
            fill: Цвет заливки. Для полупрозрачной заливки передавайте RGBA.
            draw_on: Сторона холста: left, right или both.

        Returns:
            Drawer: Текущий экземпляр для цепочки вызовов.
        """
        for bbox_xyxy in boxes:
            self.draw_structure(
                bbox_xyxy,
                color=color,
                width=width,
                fill=fill,
                draw_on=draw_on,
            )

        return self

    _CYCLIC_PALETTE: tuple[tuple[int, int, int, int], ...] = (
        (220, 50, 50, 90),    # red
        (50, 180, 50, 90),    # green
        (50, 100, 220, 90),   # blue
        (220, 140, 0, 90),    # orange
        (140, 50, 200, 90),   # violet
        (0, 180, 180, 90),    # teal
        (200, 50, 140, 90),   # pink
        (120, 80, 30, 90),    # brown
    )

    def draw_boxes_cyclic(
        self,
        boxes: list[tuple[int, int, int, int]],
        *,
        palette: tuple[tuple[int, int, int, int], ...] | None = None,
        width: int = 1,
        draw_on: str | None = None,
    ):
        """Рисует набор рамок, циклически меняя цвет из палитры.

        Удобно для отладки: каждый bbox получает свой цвет, что позволяет
        визуально различать соседние блоки и оценивать качество детекции.

        Args:
            boxes: Список bbox в формате (x1, y1, x2, y2).
            palette: RGBA-палитра цветов. По умолчанию используется встроенная.
            width: Толщина рамки.
            draw_on: Сторона холста: left, right или both.

        Returns:
            Drawer: Текущий экземпляр для цепочки вызовов.
        """
        colors = palette if palette is not None else self._CYCLIC_PALETTE
        for i, bbox_xyxy in enumerate(boxes):
            fill = colors[i % len(colors)]
            self.draw_structure(
                bbox_xyxy,
                color=fill[:3],
                width=width,
                fill=fill,
                draw_on=draw_on,
            )
        return self

    def draw_labeled_boxes(
        self,
        items: list[tuple[tuple[int, int, int, int], str]],
        *,
        color: str | tuple[int, int, int] = "blue",
        width: int = 2,
        position: Position = Position.TOP,
        fill: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
        draw_on: str | None = None,
    ):
        """Рисует набор рамок с подписями.

        Args:
            items: Список пар (bbox, label).
            color: Цвет рамки и подписи.
            width: Толщина рамки.
            position: Позиция подписи относительно рамки.
            fill: Цвет заливки. Для полупрозрачной заливки передавайте RGBA.
            draw_on: Сторона холста: left, right или both.

        Returns:
            Drawer: Текущий экземпляр для цепочки вызовов.
        """
        for bbox_xyxy, label in items:
            self.draw_structure(
                bbox_xyxy,
                label=label,
                color=color,
                width=width,
                position=position,
                fill=fill,
                draw_on=draw_on,
            )

        return self

    def draw_table_structure(
        self,
        table_bbox_xyxy: tuple[int, int, int, int],
        *,
        cells: list[Any],
        label: str | None = None,
        color: str | tuple[int, int, int] = "blue",
        cell_color: str | tuple[int, int, int] = "cornflowerblue",
        text_color: str | tuple[int, int, int] = "black",
        draw_on: str | None = None,
    ):
        """Рисует структуру таблицы и текст по ячейкам.

        Args:
            table_bbox_xyxy: Габаритный bbox таблицы.
            cells: Список объектов ячеек с полями bbox и value.
            label: Подпись таблицы.
            color: Цвет внешней рамки таблицы.
            cell_color: Цвет внутренних рамок ячеек.
            text_color: Цвет текста в ячейках.
            draw_on: Сторона холста: left, right или both.

        Returns:
            Drawer: Текущий экземпляр для цепочки вызовов.
        """
        self.draw_structure(
            table_bbox_xyxy,
            label=label,
            color=color,
            position=Position.TOP,
            draw_on=draw_on,
        )

        for cell in cells:
            self.draw_structure(
                cell.bbox.to_tuple(),
                color=cell_color,
                width=1,
                draw_on=draw_on,
            )
            if cell.value and cell.value.strip():
                self.draw_text_in_bbox(
                    cell.bbox.to_tuple(),
                    text=cell.value,
                    fill=text_color,
                    draw_on=draw_on,
                )

        return self

    def draw_text_in_bbox(
        self,
        bbox_xyxy: tuple[int, int, int, int],
        *,
        text: str,
        fill: str | tuple[int, int, int] = "black",
        padding: int = 2,
        line_spacing: int = 2,
        draw_on: str | None = None,
    ):
        """Рисует текст внутри bbox с переносами и обрезкой по высоте.

        Args:
            bbox_xyxy: Область для текста в координатах исходного изображения.
            text: Текст для вывода.
            fill: Цвет текста.
            padding: Внутренний отступ.
            line_spacing: Интервал между строками.
            draw_on: Сторона холста: left, right или both.

        Returns:
            Drawer: Текущий экземпляр для цепочки вызовов.
        """
        font = self._get_font()
        line_height = self._line_height(font)
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        max_width = max(1, (x2 - x1) - 2 * padding)
        max_height = max(1, (y2 - y1) - 2 * padding)
        wrapped_lines = self._wrap_text(text=text, max_width=max_width, font=font)
        max_lines = max(1, max_height // max(1, line_height + line_spacing))

        for xoff in self._resolve_targets(draw_on):
            cursor_y = y1 + padding
            for line in wrapped_lines[:max_lines]:
                self._draw.text((x1 + xoff + padding, cursor_y), line, fill=fill, font=font)
                cursor_y += line_height + line_spacing

        return self

    def _draw_translucent_rectangle(
        self,
        *,
        bbox: tuple[int, int, int, int],
        outline: str | tuple[int, int, int],
        width: int,
        fill: tuple[int, int, int, int],
    ) -> None:
        overlay = Image.new("RGBA", self._canvas.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(list(bbox), outline=outline, width=width, fill=fill)
        self._canvas = Image.alpha_composite(self._canvas.convert("RGBA"), overlay).convert("RGB")
        self._draw = ImageDraw.Draw(self._canvas)

    def draw_text_panel(
        self,
        lines: list[str],
        *,
        title: str | None = None,
        color: str | tuple[int, int, int] = "black",
        line_spacing: int = 6,
        padding: int = 12,
    ):
        """Рисует текстовую панель справа от debug-области.

        Args:
            lines: Строки для вывода.
            title: Заголовок панели.
            color: Цвет текста.
            line_spacing: Дополнительный вертикальный интервал между строками.
            padding: Внутренний отступ панели.

        Returns:
            Drawer: Текущий экземпляр для цепочки вызовов.

        Raises:
            ValueError: Если текстовая панель не включена.
        """
        if self._text_panel_width <= 0:
            raise ValueError("Text panel is disabled. Pass text_panel_width > 0.")

        font = self._get_font()
        panel_left = self._text_panel_offset
        panel_right = self._canvas.size[0]
        max_width = max(1, panel_right - panel_left - 2 * padding)
        line_height = self._line_height(font)
        cursor_y = padding

        if title:
            cursor_y = self._draw_wrapped_text(
                text=title,
                x=panel_left + padding,
                y=cursor_y,
                max_width=max_width,
                fill=color,
                font=font,
                line_height=line_height,
                line_spacing=line_spacing,
            )
            cursor_y += line_spacing

        for line in lines:
            cursor_y = self._draw_wrapped_text(
                text=line,
                x=panel_left + padding,
                y=cursor_y,
                max_width=max_width,
                fill=color,
                font=font,
                line_height=line_height,
                line_spacing=line_spacing,
            )
            cursor_y += line_spacing
            if cursor_y >= self._canvas.size[1] - padding:
                break

        return self

    def _draw_wrapped_text(
        self,
        *,
        text: str,
        x: int,
        y: int,
        max_width: int,
        fill: str | tuple[int, int, int],
        font,
        line_height: int,
        line_spacing: int,
    ) -> int:
        wrapped_lines = self._wrap_text(text=text, max_width=max_width, font=font)
        cursor_y = y

        for wrapped_line in wrapped_lines:
            self._draw.text((x, cursor_y), wrapped_line, fill=fill, font=font)
            cursor_y += line_height + line_spacing

        return cursor_y

    def _wrap_text(self, *, text: str, max_width: int, font) -> list[str]:
        normalized = " ".join(text.split()) or " "
        approx_char_width = max(1, self._text_size("M", font)[0])
        chunk_size = max(1, max_width // approx_char_width)
        preliminary_lines = wrap(
            normalized,
            width=chunk_size,
            break_long_words=True,
            break_on_hyphens=False,
        )

        lines: list[str] = []
        for raw_line in preliminary_lines or [normalized]:
            current = raw_line
            while current:
                next_line = current
                while self._text_size(next_line, font)[0] > max_width:
                    if len(next_line) == 1:
                        break
                    split_at = next_line.rfind(" ")
                    if split_at <= 0:
                        next_line = next_line[:-1]
                    else:
                        next_line = next_line[:split_at]
                lines.append(next_line)
                current = current[len(next_line) :].lstrip()

        return lines

    def _line_height(self, font) -> int:
        return max(1, self._text_size("Ag", font)[1])

    def _text_size(self, text: str, font) -> tuple[int, int]:
        left, top, right, bottom = self._draw.textbbox((0, 0), text, font=font)
        return max(1, right - left), max(1, bottom - top)
