import pytest
from vision_core.entities.bbox import BBox
from vision_core.entities.paragraph import Paragraph, ParagraphType


class TestParagraphType:
    """Тесты для enum ParagraphType"""

    def test_paragraph_type_values(self):
        """Тест значений enum"""
        assert ParagraphType.HEADER.value == 0
        assert ParagraphType.FOOTER.value == 1
        assert ParagraphType.NONE.value == 2


class TestParagraph:
    """Тесты для класса Paragraph"""

    @pytest.fixture
    def simple_paragraph(self):
        """Простой параграф в центре страницы"""
        return Paragraph(
            id="para_1",
            text="This is a simple paragraph",
            bbox=BBox(x_min=50, y_min=200, x_max=550, y_max=250),
        )

    @pytest.fixture
    def header_paragraph(self):
        """Параграф в верхней части страницы (header)"""
        return Paragraph(
            id="header_1",
            text="Header Text",
            type=ParagraphType.HEADER,
            bbox=BBox(x_min=50, y_min=10, x_max=550, y_max=50),
        )

    @pytest.fixture
    def footer_paragraph(self):
        """Параграф в нижней части страницы (footer)"""
        return Paragraph(
            id="footer_1",
            text="Footer Text - Page 1",
            type=ParagraphType.FOOTER,
            bbox=BBox(x_min=50, y_min=750, x_max=550, y_max=790),
        )

    @pytest.fixture
    def paragraph_with_blobs(self):
        """Параграф с текстовыми блоками"""
        return Paragraph(
            id="para_blobs",
            text="Text with blobs",
            bbox=BBox(x_min=50, y_min=200, x_max=550, y_max=250),
            blobs=[
                BBox(x_min=60, y_min=210, x_max=200, y_max=230),
                BBox(x_min=210, y_min=210, x_max=350, y_max=230),
                BBox(x_min=360, y_min=210, x_max=540, y_max=230),
            ],
        )

    @pytest.fixture
    def empty_paragraph(self):
        """Пустой параграф"""
        return Paragraph(
            id="empty",
            text="",
            bbox=BBox(x_min=50, y_min=200, x_max=550, y_max=250),
        )

    # Тесты свойств
    def test_area(self, simple_paragraph):
        """Тест расчёта площади параграфа"""
        assert simple_paragraph.area == 25000  # 500 * 50

    def test_width(self, simple_paragraph):
        """Тест расчёта ширины параграфа"""
        assert simple_paragraph.width == 500

    def test_height(self, simple_paragraph):
        """Тест расчёта высоты параграфа"""
        assert simple_paragraph.height == 50

    def test_center(self, simple_paragraph):
        """Тест расчёта центра параграфа"""
        assert simple_paragraph.center == (300.0, 225.0)

    def test_is_empty_false(self, simple_paragraph):
        """Тест проверки пустоты (не пустой)"""
        assert not simple_paragraph.is_empty

    def test_is_empty_true(self, empty_paragraph):
        """Тест проверки пустоты (пустой)"""
        assert empty_paragraph.is_empty

    def test_is_empty_whitespace(self):
        """Тест проверки пустоты (только пробелы)"""
        para = Paragraph(
            id="whitespace",
            text="   \n\t  ",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
        )
        assert para.is_empty

    def test_is_empty_none(self):
        """Тест проверки пустоты (None)"""
        para = Paragraph(
            id="none",
            text="",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
        )
        assert para.is_empty

    # Тесты методов
    def test_padding(self, simple_paragraph):
        """Тест добавления отступа"""
        padded = simple_paragraph.padding(10)
        assert padded.bbox.x_min == 40
        assert padded.bbox.y_min == 190
        assert padded.bbox.x_max == 560
        assert padded.bbox.y_max == 260

    def test_padding_with_blobs(self, paragraph_with_blobs):
        """Тест добавления отступа с блобами"""
        padded = paragraph_with_blobs.padding(5)
        assert len(padded.blobs) == 3
        assert padded.blobs[0].x_min == 55  # 60 - 5
        assert padded.blobs[0].x_max == 205  # 200 + 5

    def test_intersect(self):
        """Тест пересечения параграфов"""
        para1 = Paragraph(
            id="p1",
            text="Para 1",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=100),
        )
        para2 = Paragraph(
            id="p2",
            text="Para 2",
            bbox=BBox(x_min=50, y_min=50, x_max=150, y_max=150),
        )
        # Пересечение: 50x50 = 2500
        assert para1.intersect(para2) == 2500

    def test_intersect_no_overlap(self):
        """Тест пересечения без наложения"""
        para1 = Paragraph(
            id="p1",
            text="Para 1",
            bbox=BBox(x_min=0, y_min=0, x_max=50, y_max=50),
        )
        para2 = Paragraph(
            id="p2",
            text="Para 2",
            bbox=BBox(x_min=100, y_min=100, x_max=150, y_max=150),
        )
        assert para1.intersect(para2) == 0

    # Тесты позиционирования
    def test_is_at_top_true(self, header_paragraph):
        """Тест проверки позиции сверху (header)"""
        page_height = 800
        assert header_paragraph.is_at_top(page_height)

    def test_is_at_top_false(self, simple_paragraph):
        """Тест проверки позиции сверху (не header)"""
        page_height = 800
        assert not simple_paragraph.is_at_top(page_height)

    def test_is_at_top_custom_threshold(self):
        """Тест проверки позиции сверху с кастомным порогом"""
        para = Paragraph(
            id="custom",
            text="Test",
            bbox=BBox(x_min=0, y_min=150, x_max=100, y_max=200),
        )
        page_height = 800
        assert not para.is_at_top(page_height, threshold=0.15)  # 15% = 120px
        assert para.is_at_top(page_height, threshold=0.25)  # 25% = 200px

    def test_is_at_bottom_true(self, footer_paragraph):
        """Тест проверки позиции снизу (footer)"""
        page_height = 800
        assert footer_paragraph.is_at_bottom(page_height)

    def test_is_at_bottom_false(self, simple_paragraph):
        """Тест проверки позиции снизу (не footer)"""
        page_height = 800
        assert not simple_paragraph.is_at_bottom(page_height)

    def test_is_at_bottom_custom_threshold(self):
        """Тест проверки позиции снизу с кастомным порогом"""
        para = Paragraph(
            id="custom",
            text="Test",
            bbox=BBox(x_min=0, y_min=600, x_max=100, y_max=650),
        )
        page_height = 800
        assert not para.is_at_bottom(page_height, threshold=0.15)  # 85% = 680px
        assert para.is_at_bottom(page_height, threshold=0.25)  # 75% = 600px

    # Тесты классификации
    def test_classify_by_position_header(self):
        """Тест автоматической классификации как HEADER"""
        para = Paragraph(
            id="auto_header",
            text="Top text",
            bbox=BBox(x_min=50, y_min=20, x_max=550, y_max=60),
        )
        classified = para.classify_by_position(page_height=800)
        assert classified.type == ParagraphType.HEADER

    def test_classify_by_position_footer(self):
        """Тест автоматической классификации как FOOTER"""
        para = Paragraph(
            id="auto_footer",
            text="Bottom text",
            bbox=BBox(x_min=50, y_min=740, x_max=550, y_max=780),
        )
        classified = para.classify_by_position(page_height=800)
        assert classified.type == ParagraphType.FOOTER

    def test_classify_by_position_none(self, simple_paragraph):
        """Тест автоматической классификации как NONE"""
        classified = simple_paragraph.classify_by_position(page_height=800)
        assert classified.type == ParagraphType.NONE

    def test_classify_preserves_other_fields(self, paragraph_with_blobs):
        """Тест что классификация сохраняет остальные поля"""
        classified = paragraph_with_blobs.classify_by_position(page_height=800)
        assert classified.id == paragraph_with_blobs.id
        assert classified.text == paragraph_with_blobs.text
        assert classified.bbox == paragraph_with_blobs.bbox
        assert len(classified.blobs) == len(paragraph_with_blobs.blobs)

    # Тесты объединения
    def test_merge_with(self):
        """Тест объединения двух параграфов"""
        para1 = Paragraph(
            id="p1",
            text="First paragraph",
            bbox=BBox(x_min=50, y_min=100, x_max=300, y_max=150),
        )
        para2 = Paragraph(
            id="p2",
            text="Second paragraph",
            bbox=BBox(x_min=50, y_min=160, x_max=300, y_max=210),
        )

        merged = para1.merge_with(para2)

        assert merged.id == "p1+p2"
        assert merged.text == "First paragraph Second paragraph"
        assert merged.bbox.x_min == 50
        assert merged.bbox.y_min == 100
        assert merged.bbox.x_max == 300
        assert merged.bbox.y_max == 210

    def test_merge_with_blobs(self):
        """Тест объединения параграфов с блобами"""
        para1 = Paragraph(
            id="p1",
            text="First",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
            blobs=[BBox(x_min=10, y_min=10, x_max=50, y_max=30)],
        )
        para2 = Paragraph(
            id="p2",
            text="Second",
            bbox=BBox(x_min=0, y_min=60, x_max=100, y_max=110),
            blobs=[BBox(x_min=10, y_min=70, x_max=50, y_max=90)],
        )

        merged = para1.merge_with(para2)

        assert len(merged.blobs) == 2
        assert merged.blobs[0].y_min == 10
        assert merged.blobs[1].y_min == 70

    def test_merge_preserves_type(self, header_paragraph):
        """Тест что объединение сохраняет тип первого параграфа"""
        para2 = Paragraph(
            id="p2",
            text="More text",
            bbox=BBox(x_min=50, y_min=60, x_max=550, y_max=100),
        )

        merged = header_paragraph.merge_with(para2)
        assert merged.type == ParagraphType.HEADER

    def test_merge_overlapping_paragraphs(self):
        """Тест объединения пересекающихся параграфов"""
        para1 = Paragraph(
            id="p1",
            text="First",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=100),
        )
        para2 = Paragraph(
            id="p2",
            text="Second",
            bbox=BBox(x_min=50, y_min=50, x_max=150, y_max=150),
        )

        merged = para1.merge_with(para2)

        # Bbox должен охватывать оба параграфа
        assert merged.bbox.x_min == 0
        assert merged.bbox.y_min == 0
        assert merged.bbox.x_max == 150
        assert merged.bbox.y_max == 150

    def test_merge_empty_text(self):
        """Тест объединения с пустым текстом"""
        para1 = Paragraph(
            id="p1",
            text="",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
        )
        para2 = Paragraph(
            id="p2",
            text="Second",
            bbox=BBox(x_min=0, y_min=60, x_max=100, y_max=110),
        )

        merged = para1.merge_with(para2)
        assert merged.text == "Second"

    # Тесты инициализации
    def test_default_type(self):
        """Тест типа параграфа по умолчанию"""
        para = Paragraph(
            id="default",
            text="Default paragraph",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
        )
        assert para.type == ParagraphType.NONE

    def test_default_blobs(self):
        """Тест пустого списка блобов по умолчанию"""
        para = Paragraph(
            id="no_blobs",
            text="No blobs",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
        )
        assert para.blobs == []

    # Тесты строкового представления
    def test_str_simple(self, simple_paragraph):
        """Тест строкового представления простого параграфа"""
        result = str(simple_paragraph)
        assert "id='para_1'" in result
        assert "text='This is a simple paragraph'" in result

    def test_str_header(self, header_paragraph):
        """Тест строкового представления header параграфа"""
        result = str(header_paragraph)
        assert "type=<ParagraphType.HEADER: 0>" in result

    def test_str_with_blobs(self, paragraph_with_blobs):
        """Тест строкового представления с блобами"""
        result = str(paragraph_with_blobs)
        assert "blobs=" in result
