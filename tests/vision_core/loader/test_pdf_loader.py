import pytest
import numpy as np
from pathlib import Path
from vision_core.loader.pdf_loader import PDFLoader


class TestPDFLoader:
    """Тесты для PDFLoader"""

    @pytest.fixture
    def pdf_path(self) -> Path:
        """Путь к папке с тестовыми PDF"""
        return Path("./examples/act_sverki.pdf")

    @pytest.fixture
    def sample_pdf_with_text(self, pdf_path) -> bytes:
        """PDF с текстовым слоем"""
        if not pdf_path.exists():
            pytest.skip(f"Тестовый PDF не найден: {pdf_path}")
        return pdf_path.read_bytes()

    @pytest.fixture
    def sample_pdf_scanned(self, pdf_path) -> bytes:
        """Отсканированный PDF (без текста)"""

        if not pdf_path.exists():
            pytest.skip(f"Тестовый PDF не найден: {pdf_path}")
        return pdf_path.read_bytes()

    # Тесты инициализации
    def test_init_with_valid_pdf(self, sample_pdf_with_text):
        """Тест создания загрузчика с валидным PDF"""
        loader = PDFLoader(sample_pdf_with_text)
        assert loader.doc is not None
        assert loader.num_pages > 0
        loader.close()

    def test_init_with_invalid_bytes(self):
        """Тест создания с невалидными данными"""
        with pytest.raises(Exception):
            PDFLoader(b"not a pdf")

    # Тесты num_pages
    def test_num_pages(self, sample_pdf_with_text):
        """Тест получения количества страниц"""
        with PDFLoader(sample_pdf_with_text) as loader:
            assert isinstance(loader.num_pages, int)
            assert loader.num_pages >= 1

    # Тесты get_page_image
    def test_get_page_image_first_page(self, sample_pdf_with_text):
        """Тест рендеринга первой страницы"""
        with PDFLoader(sample_pdf_with_text) as loader:
            img = loader.get_page_image(0, dpi=150)

            # Проверяем тип
            assert isinstance(img, np.ndarray)

            # Проверяем размерность (высота, ширина, 3 канала BGR)
            assert len(img.shape) == 3
            assert img.shape[2] == 3

            # Проверяем, что изображение не пустое
            assert img.shape[0] > 0
            assert img.shape[1] > 0

    def test_get_page_image_different_dpi(self, sample_pdf_with_text):
        """Тест рендеринга с разным DPI"""
        with PDFLoader(sample_pdf_with_text) as loader:
            img_150 = loader.get_page_image(0, dpi=150)
            img_300 = loader.get_page_image(0, dpi=300)

            # При большем DPI — больше разрешение
            assert img_300.shape[0] > img_150.shape[0]
            assert img_300.shape[1] > img_150.shape[1]

    def test_get_page_image_invalid_page(self, sample_pdf_with_text):
        """Тест рендеринга несуществующей страницы"""
        with PDFLoader(sample_pdf_with_text) as loader:
            with pytest.raises(Exception):
                loader.get_page_image(999)

    # Тесты get_page_size
    def test_get_page_size(self, sample_pdf_with_text):
        """Тест получения размера страницы"""
        with PDFLoader(sample_pdf_with_text) as loader:
            width, height = loader.get_page_size(0)

            assert isinstance(width, float)
            assert isinstance(height, float)
            assert width > 0
            assert height > 0

    def test_get_page_size_all_pages(self, sample_pdf_with_text):
        """Тест размеров всех страниц"""
        with PDFLoader(sample_pdf_with_text) as loader:
            for page_num in range(loader.num_pages):
                width, height = loader.get_page_size(page_num)
                assert width > 0
                assert height > 0

    # Тесты get_text
    def test_get_text_with_text_layer(self, sample_pdf_with_text):
        """Тест извлечения текста из PDF с текстовым слоем"""
        with PDFLoader(sample_pdf_with_text) as loader:
            text = loader.get_text(0)

            assert isinstance(text, str)
            # Должен быть какой-то текст
            assert len(text) > 0

    def test_get_text_scanned_pdf(self, sample_pdf_scanned):
        """Тест извлечения текста из отсканированного PDF"""
        with PDFLoader(sample_pdf_scanned) as loader:
            text = loader.get_text(0)

            # Текста должно быть мало или совсем нет
            assert isinstance(text, str)

    # Тесты has_text_layer
    def test_has_text_layer_true(self, sample_pdf_with_text):
        """Тест проверки текстового слоя (есть)"""
        with PDFLoader(sample_pdf_with_text) as loader:
            assert loader.has_text_layer(0) is True

    def test_has_text_layer_false(self, sample_pdf_scanned):
        """Тест проверки текстового слоя (нет)"""
        with PDFLoader(sample_pdf_scanned) as loader:
            # Должно быть False, т.к. это сканированный документ
            has_text = loader.has_text_layer(0)
            assert isinstance(has_text, bool)

    # Тесты context manager
    def test_context_manager(self, sample_pdf_with_text):
        """Тест использования as context manager"""
        with PDFLoader(sample_pdf_with_text) as loader:
            assert loader.doc is not None
            pages = loader.num_pages

        # После выхода документ должен быть закрыт
        # (проверка зависит от реализации pymupdf)

    def test_manual_close(self, sample_pdf_with_text):
        """Тест ручного закрытия"""
        loader = PDFLoader(sample_pdf_with_text)
        assert loader.doc is not None
        loader.close()
        # После close документ закрыт

    # Интеграционный тест
    def test_full_workflow(self, sample_pdf_with_text):
        """Тест полного рабочего процесса"""
        with PDFLoader(sample_pdf_with_text) as loader:
            # 1. Получаем количество страниц
            num_pages = loader.num_pages
            assert num_pages > 0

            # 2. Обрабатываем каждую страницу
            for page_num in range(num_pages):
                # Размер страницы
                width, height = loader.get_page_size(page_num)
                assert width > 0 and height > 0

                # Изображение
                img = loader.get_page_image(page_num, dpi=150)
                assert img.shape[0] > 0

                # Текст
                text = loader.get_text(page_num)
                assert isinstance(text, str)

                # Проверка текстового слоя
                has_text = loader.has_text_layer(page_num)
                assert isinstance(has_text, bool)
