import pytest
from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell


class TestCell:
    """Тесты для класса Cell"""

    @pytest.fixture
    def simple_cell(self):
        """Простая ячейка без объединения"""
        return Cell(
            row=0, col=0, bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50), value="Test"
        )

    @pytest.fixture
    def merged_cell(self):
        """Объединённая ячейка"""
        return Cell(
            row=0,
            col=0,
            colspan=2,
            rowspan=2,
            bbox=BBox(x_min=0, y_min=0, x_max=200, y_max=100),
            value="Merged",
        )

    @pytest.fixture
    def cell_with_blobs(self):
        """Ячейка с текстовыми блоками"""
        return Cell(
            row=0,
            col=0,
            bbox=BBox(x_min=0, y_min=0, x_max=200, y_max=100),
            value="Text with blobs",
            blobs=[
                BBox(x_min=10, y_min=10, x_max=80, y_max=30),
                BBox(x_min=10, y_min=40, x_max=100, y_max=60),
            ],
        )

    @pytest.fixture
    def empty_cell(self):
        """Пустая ячейка"""
        return Cell(row=0, col=0, bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50))

    # Тесты свойств
    def test_area(self, simple_cell):
        """Тест расчёта площади"""
        assert simple_cell.area == 5000  # 100 * 50 * 1 * 1

    def test_area_merged(self, merged_cell):
        """Тест расчёта площади объединённой ячейки"""
        assert merged_cell.area == 80000  # 200 * 100 * 2 * 2

    def test_width(self, simple_cell):
        """Тест расчёта ширины"""
        assert simple_cell.width == 100

    def test_width_merged(self, merged_cell):
        """Тест ширины объединённой ячейки"""
        assert merged_cell.width == 400  # 200 * 2

    def test_height(self, simple_cell):
        """Тест расчёта высоты"""
        assert simple_cell.height == 50

    def test_height_merged(self, merged_cell):
        """Тест высоты объединённой ячейки"""
        assert merged_cell.height == 200  # 100 * 2

    def test_center(self, simple_cell):
        """Тест расчёта центра"""
        assert simple_cell.center == (50.0, 25.0)

    def test_is_merged_false(self, simple_cell):
        """Тест проверки объединения (не объединена)"""
        assert not simple_cell.is_merged

    def test_is_merged_true(self, merged_cell):
        """Тест проверки объединения (объединена)"""
        assert merged_cell.is_merged

    def test_is_empty_false(self, simple_cell):
        """Тест проверки пустоты (не пустая)"""
        assert not simple_cell.is_empty

    def test_is_empty_true(self, empty_cell):
        """Тест проверки пустоты (пустая)"""
        assert empty_cell.is_empty

    def test_is_empty_whitespace(self):
        """Тест проверки пустоты (только пробелы)"""
        cell = Cell(
            row=0, col=0, bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50), value="   "
        )
        assert cell.is_empty

    # Тесты методов
    def test_padding(self, simple_cell):
        """Тест добавления отступа"""
        padded = simple_cell.padding(5)
        assert padded.bbox.x_min == -5
        assert padded.bbox.y_min == -5
        assert padded.bbox.x_max == 105
        assert padded.bbox.y_max == 55

    def test_padding_with_blobs(self, cell_with_blobs):
        """Тест добавления отступа с блобами"""
        padded = cell_with_blobs.padding(5)
        assert len(padded.blobs) == 2
        assert padded.blobs[0].x_min == 5  # 10 - 5
        assert padded.blobs[1].x_min == 5

    def test_intersect(self):
        """Тест пересечения ячеек"""
        cell1 = Cell(row=0, col=0, bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50))
        cell2 = Cell(row=0, col=1, bbox=BBox(x_min=50, y_min=0, x_max=150, y_max=50))
        # Пересечение: 50x50 = 2500
        assert cell1.intersect(cell2) == 2500

    def test_intersect_no_overlap(self):
        """Тест пересечения без наложения"""
        cell1 = Cell(row=0, col=0, bbox=BBox(x_min=0, y_min=0, x_max=50, y_max=50))
        cell2 = Cell(row=0, col=1, bbox=BBox(x_min=100, y_min=0, x_max=150, y_max=50))
        assert cell1.intersect(cell2) == 0

    # Тесты get_largest_free_space
    def test_get_largest_free_space_empty_cell(self, empty_cell):
        """Тест свободного места в пустой ячейке"""
        free_space = empty_cell.get_largest_free_space()
        assert free_space == empty_cell.bbox

    def test_get_largest_free_space_with_blobs(self, cell_with_blobs):
        """Тест свободного места с блобами"""
        free_space = cell_with_blobs.get_largest_free_space(padding=2.0)

        assert free_space is not None
        # Должно быть либо справа, либо снизу от блобов
        assert free_space.width > 0 or free_space.height > 0

    def test_get_largest_free_space_right(self):
        """Тест свободного места справа от blob"""
        cell = Cell(
            row=0,
            col=0,
            bbox=BBox(x_min=0, y_min=0, x_max=200, y_max=100),
            blobs=[BBox(x_min=10, y_min=10, x_max=80, y_max=90)],
        )
        free_space = cell.get_largest_free_space(padding=2.0)

        assert free_space is not None
        assert free_space.x_min == 82  # 80 + 2
        assert free_space.x_max == 200
        assert free_space.width == 118

    def test_get_largest_free_space_bottom(self):
        """Тест свободного места снизу от blob"""
        cell = Cell(
            row=0,
            col=0,
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=200),
            blobs=[BBox(x_min=10, y_min=10, x_max=90, y_max=80)],
        )
        free_space = cell.get_largest_free_space(padding=2.0)

        assert free_space is not None
        assert free_space.y_min == 82  # 80 + 2
        assert free_space.y_max == 200
        assert free_space.height == 118

    def test_get_largest_free_space_no_space(self):
        """Тест когда свободного места нет"""
        cell = Cell(
            row=0,
            col=0,
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
            blobs=[
                BBox(x_min=0, y_min=0, x_max=100, y_max=50)  # Занимает всю ячейку
            ],
        )
        free_space = cell.get_largest_free_space(padding=2.0)

        assert free_space is None

    # Тест строкового представления
    def test_str(self, simple_cell):
        """Тест строкового представления"""
        result = str(simple_cell)
        assert "Cell(row=0" in result
        assert "col=0" in result
        assert "value=Test" in result
