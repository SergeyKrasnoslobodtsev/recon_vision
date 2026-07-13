"""Модуль для препроцессинга изображений с адаптивной обработкой.
Этот модуль включает в себя класс ImagePreprocessor, который применяет нечеткую маску
к изображению для улучшения его качества перед дальнейшей обработкой. Параметры маски
настраиваются через конфигурацию ImagePreprocessorConfig, что позволяет адаптировать
препроцессинг к различным типам изображений и условиям съемки.
"""

from __future__ import annotations

import numpy as np

from vision_core.config import ImagePreprocessorConfig
from vision_core.utils import image_utils


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
        """Применяет нечеткую маску к изображению для улучшения его качества."""
        if self._debug:
            self._debug.on_debug_image(src_image=image, stage="1_original", prefix="page", page_number=page_number)

        result = image_utils.unsharp_mask(
            image,
            kernel_size=(self.cfg.kernel, self.cfg.kernel),
            sigma=self.cfg.sigma,
            amount=self.cfg.amount,
        )

        if self._debug:
            self._debug.on_debug_image(src_image=result, stage="2_preprocessed", prefix="page", page_number=page_number)

        return result
