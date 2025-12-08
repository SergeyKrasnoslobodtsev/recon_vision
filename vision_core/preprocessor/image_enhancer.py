import cv2
import numpy as np

from vision_core.preprocessor.image_analyzer import ImageQualityMetrics


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

        # Конвертируем в grayscale
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Нормализация (всегда применяем)
        result = self._normalize(result)

        # Адаптивная обработка по проблемам
        if metrics.has_colored_bg:
            result = self._remove_colored_background(result)

        if metrics.has_noise:
            result = self._remove_noise_contours(result)

        if metrics.has_blur:
            result = self._sharpen(result)

        if metrics.contrast_level < 0.3:
            result = self._enhance_contrast(result)

        if metrics.has_broken_chars:
            result = self._close_gaps(result)

        return result

    def _normalize(self, gray: np.ndarray) -> np.ndarray:
        """Нормализация гистограммы"""
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    def _remove_noise_contours(self, gray: np.ndarray) -> np.ndarray:
        """Удаление зерна через анализ контуров (ваш метод)"""
        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Поиск контуров
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Маска для удаления мелких контуров
        mask = np.ones_like(gray) * 255

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:  # Мелкий шум
                cv2.drawContours(mask, [contour], -1, 0, -1)

        # Применяем маску
        return cv2.bitwise_and(gray, gray, mask=mask)

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE для локального выравнивания контраста"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        """Повышение резкости"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(gray, -1, kernel)

    def _close_gaps(self, gray: np.ndarray) -> np.ndarray:
        """Морфологическое закрытие для устранения разрывов"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    def _remove_colored_background(self, gray: np.ndarray) -> np.ndarray:
        """Удаление цветного фона"""
        # Adaptive thresholding лучше справляется с неравномерным фоном
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
