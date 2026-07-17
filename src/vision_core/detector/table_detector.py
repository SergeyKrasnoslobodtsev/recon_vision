import numpy as np
from loguru import logger

from vision_core.config import TableDetectorConfig, TablePreprocessorConfig
from vision_core.debug_image_observer import DebugImageObserver
from vision_core.detector import helper_table_detector as table_helper
from vision_core.entities.table import Table
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.utils import geometry_utils, image_utils


class TableDetector:
    """Детектор таблиц на изображении"""

    def __init__(
        self,
        table_detector_config: TableDetectorConfig | None = None,
        debug_image: DebugImageObserver | None = None,
    ):
        """
        Инициализация детектора таблиц.

        Args:
            preprocessor_config: Конфигурация для предобработки таблиц.
                Если None, используется конфигурация по умолчанию.
            table_detector_config: Конфигурация для детектора таблиц.
                Если None, используется конфигурация по умолчанию.
            debug_image: Наблюдатель для отладки изображений. Если None, отладка отключена.
        """
        self.cfg = table_detector_config or TableDetectorConfig()
        self._debug_image = debug_image

        self._orig_shape = None
        self._sharpness_img = None
        self._binary_img = None
        self._mask_raw_lines = None

    def _preprocess(self, image: np.ndarray, page_number: int = 0):
        self._orig_shape = image.shape
        self._sharpness_img = image_utils.apply_gamma_correction(image, gamma=0.3)
        gray_img = image_utils.to_grayscale(self._sharpness_img)
        self._binary_img = image_utils.binary_masked(gray_img, k_gauss=5, block_size=11, c=2)
        self._mask_raw_lines = image_utils.extract_lines_mask(self._binary_img, v_scale=10, h_scale=40)
        lines = geometry_utils.find_hough_lines(self._mask_raw_lines,
                                                threshold=10,
                                                min_line_length=int(self._orig_shape[0] * 0.1),
                                                max_line_gap=30,
                                                )



