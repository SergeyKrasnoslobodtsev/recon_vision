import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Union, Optional
from enum import Enum
from pathlib import Path


class Position(Enum):
    TOP = 0
    BOTTOM = 1
    NONE = 2


class Drawer:
    """
    Drawer, упрощающий отрисовку боксов/подписей в тестах.

    Новый режим:
      side_by_side=True -> итоговое изображение: [original | blank], рисование по умолчанию на blank (справа).
    """

    def __init__(
        self,
        image: Union[np.ndarray, Image.Image],
        *,
        side_by_side: bool = False,
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

        w, h = self._left.size

        if side_by_side:
            right = Image.new("RGB", (w, h), color=blank_color)
            canvas = Image.new("RGB", (w * 2, h), color=blank_color)
            canvas.paste(self._left, (0, 0))
            canvas.paste(right, (w, 0))
            self._canvas = canvas
            self._x_left_offset = 0
            self._x_right_offset = w
            self._draw_on = "right" if draw_on == "auto" else draw_on
        else:
            self._canvas = self._left
            self._x_left_offset = 0
            self._x_right_offset = 0
            self._draw_on = "left" if draw_on == "auto" else draw_on

        self._draw = ImageDraw.Draw(self._canvas)
        self._font = None

    def _get_font(self):
        if self._font is not None:
            return self._font
        try:
            self._font = ImageFont.load_default()
        except Exception:
            self._font = None
        return self._font

    def _targets(self):
        mode = self._draw_on
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
        label: Optional[str] = None,
        color: Union[str, tuple[int, int, int]] = "blue",
        width: int = 2,
        position=None,
        fill: Optional[tuple[int, int, int]] = None,
    ):
        """
        bbox_xyxy: (x1, y1, x2, y2) в координатах ОРИГИНАЛА (левой части).
        В режиме side-by-side рисуем со смещением на правую часть автоматически.
        """
        x1, y1, x2, y2 = map(int, bbox_xyxy)

        for xoff in self._targets():
            xx1, xx2 = x1 + xoff, x2 + xoff
            if fill is not None:
                self._draw.rectangle(
                    [xx1, y1, xx2, y2], outline=color, width=width, fill=fill
                )
            else:
                self._draw.rectangle([xx1, y1, xx2, y2], outline=color, width=width)

            if label:
                font = self._get_font()
                text_w, text_h = self._draw.textbbox((0, 0), label, font=font)[2:]
                pad = 2

                # TOP по умолчанию
                place_top = True
                if position is not None and hasattr(position, "name"):
                    if position.name.upper() == "BOTTOM":
                        place_top = False

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

    def save(self, path: Union[str, Path]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._canvas.save(path)

    def to_pil(self) -> Image.Image:
        return self._canvas

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._canvas)
