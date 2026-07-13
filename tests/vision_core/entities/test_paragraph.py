import pytest
from vision_core.entities.bbox import BBox
from vision_core.entities.paragraph import Paragraph, ParagraphType


class TestParagraphType:
    def test_paragraph_type_has_expected_members(self):
        assert ParagraphType.PAGE_HEADER
        assert ParagraphType.PAGE_FOOTER
        assert ParagraphType.SECTION_TITLE
        assert ParagraphType.TABLE_CAPTION
        assert ParagraphType.BODY_TEXT
        assert ParagraphType.UNKNOWN


class TestParagraph:
    @pytest.fixture
    def simple_paragraph(self):
        return Paragraph(
            id="para_1",
            text="This is a simple paragraph",
            bbox=BBox(x_min=50, y_min=200, x_max=550, y_max=250),
        )

    @pytest.fixture
    def header_paragraph(self):
        return Paragraph(
            id="header_1",
            text="Header Text",
            type=ParagraphType.PAGE_HEADER,
            bbox=BBox(x_min=50, y_min=10, x_max=550, y_max=50),
        )

    @pytest.fixture
    def footer_paragraph(self):
        return Paragraph(
            id="footer_1",
            text="Footer Text - Page 1",
            type=ParagraphType.PAGE_FOOTER,
            bbox=BBox(x_min=50, y_min=750, x_max=550, y_max=790),
        )

    @pytest.fixture
    def paragraph_with_blobs(self):
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
        return Paragraph(
            id="empty",
            text="",
            bbox=BBox(x_min=50, y_min=200, x_max=550, y_max=250),
        )

    # Свойства
    def test_area(self, simple_paragraph):
        assert simple_paragraph.area == 25000

    def test_width(self, simple_paragraph):
        assert simple_paragraph.width == 500

    def test_height(self, simple_paragraph):
        assert simple_paragraph.height == 50

    def test_center(self, simple_paragraph):
        assert simple_paragraph.center == (300.0, 225.0)

    def test_is_empty_false(self, simple_paragraph):
        assert not simple_paragraph.is_empty

    def test_is_empty_true(self, empty_paragraph):
        assert empty_paragraph.is_empty

    def test_is_empty_whitespace(self):
        para = Paragraph(id="ws", text="   \n\t  ", bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50))
        assert para.is_empty

    # Методы
    def test_padding(self, simple_paragraph):
        padded = simple_paragraph.padding(10)
        assert padded.bbox.x_min == 40
        assert padded.bbox.y_min == 190
        assert padded.bbox.x_max == 560
        assert padded.bbox.y_max == 260

    def test_padding_with_blobs(self, paragraph_with_blobs):
        padded = paragraph_with_blobs.padding(5)
        assert len(padded.blobs) == 3
        assert padded.blobs[0].x_min == 55
        assert padded.blobs[0].x_max == 205

    def test_intersect(self):
        para1 = Paragraph(id="p1", text="A", bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=100))
        para2 = Paragraph(id="p2", text="B", bbox=BBox(x_min=50, y_min=50, x_max=150, y_max=150))
        assert para1.intersect(para2) == 2500

    def test_intersect_no_overlap(self):
        para1 = Paragraph(id="p1", text="A", bbox=BBox(x_min=0, y_min=0, x_max=50, y_max=50))
        para2 = Paragraph(id="p2", text="B", bbox=BBox(x_min=100, y_min=100, x_max=150, y_max=150))
        assert para1.intersect(para2) == 0

    # Позиционирование
    def test_is_at_top_true(self, header_paragraph):
        assert header_paragraph.is_at_top(800)

    def test_is_at_top_false(self, simple_paragraph):
        assert not simple_paragraph.is_at_top(800)

    def test_is_at_top_custom_threshold(self):
        para = Paragraph(id="p", text="T", bbox=BBox(x_min=0, y_min=150, x_max=100, y_max=200))
        assert not para.is_at_top(800, threshold=0.15)
        assert para.is_at_top(800, threshold=0.25)

    def test_is_at_bottom_true(self, footer_paragraph):
        assert footer_paragraph.is_at_bottom(800)

    def test_is_at_bottom_false(self, simple_paragraph):
        assert not simple_paragraph.is_at_bottom(800)

    def test_is_centered_true(self):
        para = Paragraph(id="p", text="T", bbox=BBox(x_min=200, y_min=100, x_max=600, y_max=150))
        assert para.is_centered(page_width=800, margin=0.2)

    def test_is_centered_false(self):
        para = Paragraph(id="p", text="T", bbox=BBox(x_min=0, y_min=100, x_max=50, y_max=150))
        assert not para.is_centered(page_width=800, margin=0.1)

    # Классификация
    def test_classify_header(self):
        para = Paragraph(id="p", text="Top", bbox=BBox(x_min=50, y_min=20, x_max=550, y_max=60))
        classified = para.classify_by_position(page_shape=(800, 600), tables=None, median_height_row=20.0)
        assert classified.type == ParagraphType.PAGE_HEADER

    def test_classify_footer(self):
        para = Paragraph(id="p", text="Bot", bbox=BBox(x_min=50, y_min=740, x_max=550, y_max=780))
        classified = para.classify_by_position(page_shape=(800, 600), tables=None, median_height_row=20.0)
        assert classified.type == ParagraphType.PAGE_FOOTER

    def test_classify_body_text(self):
        para = Paragraph(id="p", text="Left text", bbox=BBox(x_min=0, y_min=200, x_max=100, y_max=250))
        classified = para.classify_by_position(page_shape=(800, 600), tables=None, median_height_row=20.0)
        assert classified.type == ParagraphType.BODY_TEXT

    def test_classify_preserves_fields(self, paragraph_with_blobs):
        classified = paragraph_with_blobs.classify_by_position(page_shape=(800, 600), tables=None, median_height_row=20.0)
        assert classified.id == paragraph_with_blobs.id
        assert classified.text == paragraph_with_blobs.text
        assert len(classified.blobs) == len(paragraph_with_blobs.blobs)

    # Слияние
    def test_merge_with(self):
        para1 = Paragraph(id="p1", text="First", bbox=BBox(x_min=50, y_min=100, x_max=300, y_max=150))
        para2 = Paragraph(id="p2", text="Second", bbox=BBox(x_min=50, y_min=160, x_max=300, y_max=210))
        merged = para1.merge_with(para2)
        assert merged.id == "p1+p2"
        assert merged.text == "First Second"
        assert merged.bbox.y_min == 100
        assert merged.bbox.y_max == 210

    def test_merge_with_blobs(self):
        para1 = Paragraph(id="p1", text="A", bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
                          blobs=[BBox(x_min=10, y_min=10, x_max=50, y_max=30)])
        para2 = Paragraph(id="p2", text="B", bbox=BBox(x_min=0, y_min=60, x_max=100, y_max=110),
                          blobs=[BBox(x_min=10, y_min=70, x_max=50, y_max=90)])
        merged = para1.merge_with(para2)
        assert len(merged.blobs) == 2

    def test_merge_preserves_type(self, header_paragraph):
        para2 = Paragraph(id="p2", text="More", bbox=BBox(x_min=50, y_min=60, x_max=550, y_max=100))
        merged = header_paragraph.merge_with(para2)
        assert merged.type == ParagraphType.PAGE_HEADER

    def test_merge_empty_text(self):
        para1 = Paragraph(id="p1", text="", bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50))
        para2 = Paragraph(id="p2", text="Second", bbox=BBox(x_min=0, y_min=60, x_max=100, y_max=110))
        merged = para1.merge_with(para2)
        assert merged.text == "Second"

    # Инициализация
    def test_default_type(self):
        para = Paragraph(id="d", text="Default", bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50))
        assert para.type == ParagraphType.UNKNOWN

    def test_default_blobs(self):
        para = Paragraph(id="d", text="No blobs", bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50))
        assert para.blobs == []

    # to_markdown
    def test_to_markdown_header(self, header_paragraph):
        assert header_paragraph.to_markdown() == "<small>Header Text</small>"

    def test_to_markdown_footer(self, footer_paragraph):
        assert footer_paragraph.to_markdown() == "<small>Footer Text - Page 1</small>"

    def test_to_markdown_title(self):
        para = Paragraph(id="p", text="Заголовок", type=ParagraphType.SECTION_TITLE,
                         bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50))
        assert para.to_markdown() == "**Заголовок**"

    def test_to_markdown_caption(self):
        para = Paragraph(id="p", text="Таблица 1", type=ParagraphType.TABLE_CAPTION,
                         bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50))
        assert para.to_markdown() == "*Таблица 1*"

    def test_to_markdown_body_text(self, simple_paragraph):
        assert simple_paragraph.to_markdown() == "This is a simple paragraph"

    def test_to_markdown_empty(self, empty_paragraph):
        assert empty_paragraph.to_markdown() == ""
