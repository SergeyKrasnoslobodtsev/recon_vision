"""
Модуль для предобработки изображений таблиц. Содержит класс TablePreprocessor,
который выполняет гамма-коррекцию и бинаризацию изображения для выделения таблиц.
"""

from __future__ import annotations

import numpy as np

from vision_core.config import TablePreprocessorConfig
from vision_core.debug_image_observer import DebugImageObserver
from vision_core.utils.image_utils import binary_threshold, gamma_correction


class TablePreprocessor:
    def __init__(
        self,
        cfg: TablePreprocessorConfig | None = None,
        debug_image: DebugImageObserver | None = None,
    ):
        """Предобработчик для таблиц

        Args:
            cfg: Конфигурация предобработчика таблиц
            debug_image: Наблюдатель для отладки изображений. Если None, отладка отключена.
        """
        if cfg is None:
            cfg = TablePreprocessorConfig()

        self.cfg = cfg
        self._debug_image = debug_image

    def process(self, image: np.ndarray, page_number: int = 0) -> np.ndarray:
        """Создание маски таблицы из изображения"""
        gamma_img = gamma_correction(image, self.cfg.gamma)
        binary_image = binary_threshold(gamma_img, block_size=self.cfg.block_size, C=self.cfg.C)

        if self._debug_image:
            self._debug_image.on_debug_image(
                src_image=binary_image,
                stage="4_table_preprocessor",
                prefix="binary",
                page_number=page_number,
            )

        return binary_image
