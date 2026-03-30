import cv2
import numpy as np

from vision_core.config import ImagePreprocessorConfig


class ImagePreprocessor:
    """Препроцессинг изображений с адаптивной обработкой"""

    def __init__(self, config: ImagePreprocessorConfig | None = None):
        self.cfg = config or ImagePreprocessorConfig()

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Анализирует и улучшает изображение

        Returns:
            enhanced_image: Улучшенное изображение
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        normalized, _ = self._normalize_background(gray, kernel_size=self.cfg.kernel_size_morph)
        processed = self._darken_light_strokes(
            normalized,
            denoise_h=self.cfg.denoise_h,
            clip_limit=self.cfg.clip_limit,
            tile_size=self.cfg.tile_size,
            kernel_size=self.cfg.kernel_size,
            blackhat_gain=self.cfg.blackhat_gain,
        )

        return processed

    def _darken_light_strokes(
        self,
        gray: np.ndarray,
        denoise_h: int = 3,
        clip_limit: float = 1.9,
        tile_size: int = 8,
        kernel_size: int = 5,
        blackhat_gain: float = 0.3,
    ) -> np.ndarray:
        """Усиление светлых штрихов с адаптивными параметрами

        Args:
            gray (np.ndarray): входное изображение в оттенках серого
            denoise_h (int, optional): сила удаления шума (0 - без удаления, 3-5 - легкое удаление). Defaults to 3.
            clip_limit (float, optional): порог для CLAHE (1.0 - без усиления, 2.0 - сильное усиление). Defaults to 1.9.
            tile_size (int, optional): размер тайла для CLAHE (меньше - более локальный контраст). Defaults to 8.
            kernel_size (int, optional): размер ядра для blackhat (3-5 обычно достаточно). Defaults to 5.
            blackhat_gain (float, optional): коэффициент усиления для вычитания (0.2-0.3 может помочь,
            но зависит от качества скана). Defaults to 0.3.

        Returns:
            np.ndarray: обработанное изображение
        """
        denoised = cv2.fastNlMeansDenoising(gray, h=denoise_h)
        local = self._clahe_soft(denoised, clip_limit=clip_limit, tile_size=tile_size)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(local, cv2.MORPH_BLACKHAT, kernel)

        result = cv2.subtract(local, (blackhat.astype(np.float32) * blackhat_gain).astype(np.uint8))
        result = cv2.medianBlur(result, 3)  # легкое размытие для сглаживания артефактов
        return result

    def _clahe_soft(
        self,
        gray: np.ndarray,
        clip_limit: float = 1.8,
        tile_size: int = 8,
    ) -> np.ndarray:
        """Применяет CLAHE (Contrast Limited Adaptive Histogram Equalization) для улучшения контраста.

        Args:
            gray (np.ndarray): входное изображение в оттенках серого
            clip_limit (float, optional): порог для CLAHE (1.0 - без усиления, 2.0 - сильное усиление). Defaults to 1.8.
            tile_size (int, optional): размер тайла для CLAHE (меньше - более локальный контраст). Defaults to 8.

        Returns:
            np.ndarray: изображение с улучшенным контрастом
        """
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size),
        )
        return clahe.apply(gray)

    def _normalize_background(self, gray, kernel_size=21):
        """Нормализует фон, уменьшая влияние неровностей освещения
        Args:
            gray (np.ndarray): входное изображение в оттенках серого
            kernel_size (int, optional): размер ядра для морфологических операций
            для оценки фона (обычно 21). Defaults to 21.
        Returns:
            tuple[np.ndarray, np.ndarray]: нормализованное изображение и оценка фона
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        normalized = cv2.divide(gray, background, scale=255)
        return normalized, background
