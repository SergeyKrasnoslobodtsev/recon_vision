from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class ImageQualityMetrics:
    """Метрики качества изображения"""

    has_noise: bool  # Есть ли зерно
    has_blur: bool  # Есть ли размытие
    has_colored_bg: bool  # Цветной фон
    has_skew: bool
    contrast_level: float  # Уровень контраста (0-1)
    text_density: float  # Плотность текста (0-1)
    noise_ratio: float  # Доля шума (0-1)
    sharpness: float  # Резкость (Laplacian variance)
    skew_angle: float


class ImageAnalyzer:
    """Анализирует изображение и выявляет проблемы"""

    def __init__(
        self,
        noise_area_threshold: int = 10,
        blur_threshold: float = 500.0,
        colored_bg_threshold: float = 15.0,
        contrast_threshold: float = 0.3,
        noise_ratio_threshold: float = 0.25,
        skew_threshold: float = 0.5,
        max_skew: int = 10,
    ):
        """
        Args:
            noise_area_threshold: Макс площадь контура для классификации как шум (пикс)
            blur_threshold: Порог Laplacian variance для размытия
            colored_bg_threshold: Порог std цветовых каналов для цветного фона
            contrast_threshold: Мин уровень контраста
            noise_ratio_threshold: Порог доли шума для has_noise
            skew_threshold: Порог угла наклона (градусы)
            max_skew: Максимальный угол наклона (градусы)
        """
        self.noise_area_threshold = noise_area_threshold
        self.blur_threshold = blur_threshold
        self.colored_bg_threshold = colored_bg_threshold
        self.contrast_threshold = contrast_threshold
        self.noise_ratio_threshold = noise_ratio_threshold
        self.skew_threshold = skew_threshold
        self.max_skew = max_skew

    def analyze(self, image: np.ndarray) -> ImageQualityMetrics:
        """
        Анализирует изображение и возвращает метрики

        Args:
            image: BGR изображение

        Returns:
            ImageQualityMetrics: Набор метрик качества
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        noise_ratio = self._measure_noise_ratio(gray)
        sharpness = self._measure_sharpness(gray)
        contrast_level = self._measure_contrast(gray)
        skew_angle = self._measure_skew_angle(gray)

        return ImageQualityMetrics(
            has_noise=noise_ratio > self.noise_ratio_threshold,
            has_blur=sharpness < self.blur_threshold,
            has_colored_bg=self._detect_colored_background(image),
            has_skew=abs(skew_angle) > self.skew_threshold,
            contrast_level=contrast_level,
            text_density=self._measure_text_density(gray),
            noise_ratio=noise_ratio,
            sharpness=sharpness,
            skew_angle=skew_angle,
        )

    def _detect_noise(self, gray: np.ndarray) -> bool:
        """Определяет наличие зерна по мелким контурам"""
        noise_ratio = self._measure_noise_ratio(gray)
        return noise_ratio > self.noise_ratio_threshold

    def _detect_blur(self, gray: np.ndarray) -> bool:
        """Определяет размытие по Laplacian variance"""
        sharpness = self._measure_sharpness(gray)
        return sharpness < self.blur_threshold

    def _detect_colored_background(self, bgr: np.ndarray) -> bool:
        """Проверяет наличие цветного фона"""
        b, g, r = cv2.split(bgr)

        # Среднее значение каналов
        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)

        # Максимальная разница между средними
        max_diff = max(abs(mean_b - mean_g), abs(mean_b - mean_r), abs(mean_g - mean_r))

        return max_diff > self.colored_bg_threshold

    def _measure_contrast(self, gray: np.ndarray) -> float:
        """Измеряет контраст (std / mean)"""
        mean = np.mean(gray)
        std = np.std(gray)

        if mean == 0:
            return 0.0

        # Нормализуем в диапазон 0-1
        contrast = std / mean
        return min(contrast, 1.0)

    def _measure_text_density(self, gray: np.ndarray) -> float:
        """Соотношение текст/фон через анализ контуров"""
        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Поиск контуров
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return 0.0

        # Фильтруем шумовые контуры (площадь < noise_area_threshold)
        text_contours = [
            c for c in contours if cv2.contourArea(c) >= self.noise_area_threshold
        ]

        if len(text_contours) == 0:
            return 0.0

        # Суммарная площадь текстовых контуров
        total_text_area = sum(cv2.contourArea(c) for c in text_contours)

        # Общая площадь изображения
        total_area = gray.shape[0] * gray.shape[1]

        return total_text_area / total_area

    def _measure_noise_ratio(self, gray: np.ndarray) -> float:
        """Доля шумовых контуров к общему числу"""
        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Поиск контуров
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return 0.0

        # Считаем мелкие контуры (шум)
        noise_contours = sum(
            1 for c in contours if cv2.contourArea(c) < self.noise_area_threshold
        )

        return noise_contours / len(contours)

    def _measure_sharpness(self, gray: np.ndarray) -> float:
        """Резкость по Laplacian variance"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return float(variance)

    def _measure_skew_angle(self, gray: np.ndarray) -> float:
        """Определяет угол наклона изображения"""
        _, width = gray.shape

        # Денойзинг для более точного определения линий
        denoised = cv2.fastNlMeansDenoising(gray, h=3)

        # Бинаризация
        _, binary = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        # Поиск линий
        lines = cv2.HoughLinesP(
            binary,
            1,
            np.pi / 180,
            200,
            minLineLength=width / 12,
            maxLineGap=width / 150,
        )

        if lines is None or len(lines) < 5:
            return 0.0

        # Вычисляем углы линий
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angles.append(np.arctan2(y2 - y1, x2 - x1))

        # Определяем ориентацию (альбомная/портретная)
        landscape = (
            np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2
        )

        # Фильтруем углы в зависимости от ориентации
        if landscape:
            angles = [
                angle
                for angle in angles
                if np.deg2rad(90 - self.max_skew)
                < abs(angle)
                < np.deg2rad(90 + self.max_skew)
            ]
        else:
            angles = [
                angle for angle in angles if abs(angle) < np.deg2rad(self.max_skew)
            ]

        if len(angles) < 5:
            return 0.0

        # Медианный угол
        angle_deg = np.rad2deg(np.median(angles))

        # Корректировка для альбомной ориентации
        if landscape:
            if angle_deg < 0:
                angle_deg += 90
            elif angle_deg > 0:
                angle_deg -= 90

        return float(angle_deg)
