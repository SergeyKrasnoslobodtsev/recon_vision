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

        # # Mild color correction
        # gamma = 1.1
        # inv_gamma = 1.0 / gamma
        # table = np.array(
        #     [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        # ).astype("uint8")
        # corrected = cv2.LUT(image, table)

        # # LAB Color Space
        # lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)

        # # Gentle CLAHE
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
        # l = clahe.apply(l)

        # # Merge channels
        # lab = cv2.merge([l, a, b])
        # enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # # Smart saturation boost
        # hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)

        # # Saturation mask
        # s_clean = cv2.bilateralFilter(s, 9, 75, 75)
        # s = cv2.addWeighted(s, 1.2, s_clean, 0.3, 0)
        # s = np.clip(s, 20, 230).astype(np.uint8)

        # hsv = cv2.merge([h, s, v])
        # enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # # Controlled sharpening
        # blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        # sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        # # Noise reduction
        # final = cv2.bilateralFilter(sharpened, 9, 175, 175)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if metrics.has_skew:
            gray = self._deskew(gray, metrics.skew_angle)
        return gray

        # if (
        #     not metrics.has_skew
        #     and not metrics.has_noise
        #     and not metrics.has_blur
        #     and metrics.contrast_level >= 0.5
        # ):
        #     logger.debug("Изображение выглядит хорошо — обработка пропущена")
        #     return result

        # is_color = len(result.shape) == 3

        # # Цветной фон → перевод в gray
        # if metrics.has_colored_bg and is_color:
        #     logger.debug("Удаление цветного фона (gray)")
        #     result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        #     is_color = False

        # if metrics.has_skew:
        #     logger.debug(f"Исправление наклона на {metrics.skew_angle:.2f} градусов")
        #     gray = result if not is_color else cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        #     result = self._deskew(gray, metrics.skew_angle)
        #     is_color = False

        # if metrics.has_noise:
        #     logger.debug(f"Удаление шума {metrics.noise_ratio:.2f}")
        #     result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        #     result = self._remove_noise_soft(result)
        #     is_color = False

        # if metrics.has_blur:
        #     logger.debug(f"Повышение резкости {metrics.sharpness:.2f}")
        #     result = self._unsharp_mask(
        #         result, is_color=is_color, amount=1.0, radius=1.0
        #     )

        # if metrics.contrast_level < 0.4:
        #     logger.debug(f"Улучшение контраста {metrics.contrast_level:.2f}")
        #     if is_color:
        #         result = self._enhance_contrast_lab(result)
        #     else:
        #         result = self._enhance_contrast(result)

        # # Мягкая нормализация через LUT (авто-гамма), вместо жесткого adaptiveThreshold
        # result = self._auto_gamma_lut(result, is_color=is_color)

        # # Бинаризация оставляем только если нужна (плохие контуры)
        # if not is_color and metrics.contrast_level < 0.3:
        #     result = self._normalize(result)

        # logger.info("Процесс адаптивной обработки изображения завершен")

        # return result

    def _normalize(self, gray: np.ndarray) -> np.ndarray:
        """Нормализация гистограммы"""
        # norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
        )

    def _remove_noise_soft(self, gray: np.ndarray) -> np.ndarray:
        """Щадящее шумоподавление: xphoto (если доступно) или bilateral/median."""
        dst = np.zeros(gray.shape, np.uint8)
        cv2.denoise_TVL1([gray], dst, lambda_=1.0, niters=1)
        return dst

    def _unsharp_mask(
        self, img: np.ndarray, is_color: bool, amount: float = 1.0, radius: float = 1.0
    ) -> np.ndarray:
        """Unsharp mask для четкости текста."""
        if is_color:
            blurred = cv2.GaussianBlur(img, (0, 0), radius)
            return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        blurred = cv2.GaussianBlur(img, (0, 0), radius)
        return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    def _enhance_contrast_lab(self, bgr: np.ndarray) -> np.ndarray:
        """CLAHE по каналу L в LAB."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _auto_gamma_lut(
        self, img: np.ndarray, is_color: bool, target_mean: float = 0.5
    ) -> np.ndarray:
        """Автогамма через LUT — мягкая нормализация без перешарпа хороших снимков."""
        arr = img if not is_color else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = np.clip(arr.mean() / 255.0, 1e-3, 0.999)
        gamma = np.clip(np.log(target_mean) / np.log(mean), 0.6, 1.6)
        lut = np.array(
            [np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
            dtype=np.uint8,
        )
        if is_color:
            return cv2.LUT(img, lut)
        return cv2.LUT(img, lut)

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
        sharpened = cv2.filter2D(gray, 0, filter)

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
