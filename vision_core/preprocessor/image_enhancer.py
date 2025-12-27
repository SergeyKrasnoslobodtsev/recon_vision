import cv2
import numpy as np

from vision_core.preprocessor.image_analyzer import ImageQualityMetrics

from loguru import logger


class ImageEnhancer:
    """Адаптивное улучшение изображения"""

    def enhance(self, image: np.ndarray, metrics: ImageQualityMetrics) -> np.ndarray:
        """
        Улучшает изображение на основе выявленных проблем

        Args:
            image: Исходное изображение
            metrics: Метрики качества

        Returns:
            Улучшенное изображение
        """
        result = image.copy()
        logger.info("Процесс адаптивной обработки изображения начат")
        # Конвертируем в grayscale
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Обработка цветного фона
        if metrics.has_colored_bg:
            logger.debug("Удаление цветного фона из изображения")
            # result = self._remove_colored_background(result)
        # Исправление наклона
        if metrics.has_skew:
            logger.debug(f"Исправление наклона на {metrics.skew_angle:.2f} градусов")
            result = self._deskew(result, metrics.skew_angle)

        # Удаление шума
        if metrics.has_noise:
            logger.debug(f"Удаление шума с изображения {metrics.noise_ratio:.2f}")
            result = self._remove_noise_contours(result)

        # Повышение резкости
        if metrics.has_blur:
            if metrics.has_blur and metrics.has_noise:
                result = cv2.bilateralFilter(result, d=3, sigmaColor=75, sigmaSpace=75)
                logger.debug(
                    f"Удаление шума с помощью Non-Local Means Denoising {metrics.noise_ratio:.2f}"
                )
                # result = self._sharpen(result)
            else:
                logger.debug(f"Повышение резкости изображения {metrics.sharpness:.2f}")
                result = self._sharpen(result)

        # Улучшение контраста
        if metrics.contrast_level < 0.3:
            logger.debug(
                f"Улучшение контраста изображения {metrics.contrast_level:.2f}"
            )
            result = self._enhance_contrast(result)
        result = self._normalize(result)
        logger.info("Процесс адаптивной обработки изображения завершен")

        return result

    def _normalize(self, gray: np.ndarray) -> np.ndarray:
        """Нормализация гистограммы"""
        # norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
        )

    def _remove_noise_contours(self, gray: np.ndarray) -> np.ndarray:
        """Удаление зерна через анализ контуров"""
        # noise_removed = cv2.fastNlMeansDenoising(gray, h=15)
        blured = cv2.medianBlur(gray, 3)
        return blured

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE для локального выравнивания контраста"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        return clahe.apply(gray)

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        """Повышение резкости"""

        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, filter)

        return sharpened

    # def _remove_colored_background(self, gray: np.ndarray) -> np.ndarray:
    #     """Удаление цветного фона с помощью адаптивной бинаризации"""

    #     return cv2.adaptiveThreshold(
    #         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
    #     )

    def _deskew(self, gray: np.ndarray, angle: float) -> np.ndarray:
        """Выравнивание наклона изображения"""

        height, width = gray.shape
        center = (width / 2, height / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(gray, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
