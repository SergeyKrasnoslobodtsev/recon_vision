import numpy as np
from vision_core.preprocessor.image_analyzer import ImageAnalyzer
from vision_core.preprocessor.image_enhancer import ImageEnhancer


class ImagePreprocessor:
    """Препроцессинг изображений с адаптивной обработкой"""

    def __init__(self):
        self.analyzer = ImageAnalyzer()
        self.enhancer = ImageEnhancer()

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Анализирует и улучшает изображение

        Returns:
            enhanced_image: Улучшенное изображение
        """
        # 1. Анализ
        metrics = self.analyzer.analyze(image)

        # 2. Обработка
        enhanced = self.enhancer.enhance(image, metrics)

        return enhanced
