from __future__ import annotations

import cv2
import numpy as np

from vision_core.config import ImagePreprocessorConfig


class ImagePreprocessor:
    """Препроцессинг изображений с адаптивной обработкой"""

    def __init__(
        self,
        config: ImagePreprocessorConfig | None = None,
        debug_image=None,
    ):
        self.cfg = config or ImagePreprocessorConfig()
        self._debug = debug_image

    def process(self, image: np.ndarray, *, page_number: int = 0) -> np.ndarray:
        """Анализирует и улучшает изображение."""
        if self._debug:
            self._debug.on_debug_image(src_image=image, stage="1_original", prefix="page", page_number=page_number)

        result = self._unsharp_mask(
            image,
            kernel_size=(self.cfg.kernel, self.cfg.kernel),
            sigma=self.cfg.sigma,
            amount=self.cfg.amount,
        )

        if self._debug:
            self._debug.on_debug_image(src_image=result, stage="2_preprocessed", prefix="page", page_number=page_number)

        return result

    def _unsharp_mask(self, image: np.ndarray, kernel_size=(5, 5), sigma=1.5, amount=1.0) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return sharpened
