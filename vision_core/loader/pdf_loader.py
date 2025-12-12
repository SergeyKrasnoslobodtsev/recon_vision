import pymupdf
import numpy as np
from PIL import Image
import cv2

from vision_core.preprocessor.image_preprocessor import ImagePreprocessor


class PDFLoader:
    """Загрузчик PDF документов"""

    def __init__(self, pdf_bytes: bytes):
        """
        Инициализация загрузчика

        Args:
            pdf_bytes: PDF документ в байтах
        """
        self.pdf_bytes = pdf_bytes
        self.doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        self.preprocessor = ImagePreprocessor()

    @property
    def num_pages(self) -> int:
        """Количество страниц"""
        return len(self.doc)

    def get_page_image(self, page_num: int, dpi: int = 300) -> np.ndarray:
        """
        Рендерит страницу PDF в изображение (numpy array для OpenCV)

        Args:
            page_num: Номер страницы (0-indexed)
            dpi: Разрешение (300 для OCR, 150 для быстрого просмотра)

        Returns:
            numpy.ndarray: BGR изображение для OpenCV
        """
        page = self.doc[page_num]

        pix = page.get_pixmap(dpi=dpi, alpha=False)

        # Конвертируем в numpy array (RGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        return img_bgr

    def get_page_image_preprocessed(self, page_num: int, dpi: int = 300) -> np.ndarray:
        """
        Рендерит страницу PDF в изображение с препроцессингом

        Args:
            page_num: Номер страницы (0-indexed)
            dpi: Разрешение (300 для OCR, 150 для быстрого просмотра)

        Returns:
            numpy.ndarray: Препроцессированное BGR изображение для OpenCV
        """
        image = self.get_page_image(page_num, dpi=dpi)
        processed_image = self.preprocessor.process(image)

        return processed_image

    def get_page_size(self, page_num: int) -> tuple[float, float]:
        """
        Получить размер страницы в пикселях

        Returns:
            (width, height) в пикселях при 72 DPI
        """
        page = self.doc[page_num]
        rect = page.rect
        return (rect.width, rect.height)

    def get_text(self, page_num: int) -> str:
        """
        Извлечь текст со страницы (если есть text layer)

        Полезно для проверки: если текста нет → нужен OCR
        """
        page = self.doc[page_num]
        return page.get_text()

    def has_text_layer(self, page_num: int) -> bool:
        """Проверяет, есть ли текстовый слой на странице"""
        text = self.get_text(page_num)
        return len(text.strip()) > 10

    def close(self):
        """Закрыть документ"""
        self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
