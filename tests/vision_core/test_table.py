import pytest
from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.table import Table


class TestTable:
    """Тесты для класса Table"""

    @pytest.fixture
    def simple_table(self):
        """Простая таблица 3x3"""
        cells = []
        for row in range(3):
            for col in range(3):
                cells.append(
                    Cell(
                        row=row,
                        col=col,
                        bbox=BBox(
                            x_min=col * 100,
                            y_min=row * 50,
                            x_max=(col + 1) * 100,
                            y_max=(row + 1) * 50,
                        ),
                        value=f"R{row}C{col}",
                    )
                )
        return Table(
            id="table_1",
            bbox=BBox(x_min=0, y_min=0, x_max=300, y_max=150),
            num_rows=3,
            num_cols=3,
            cells=cells,
            start_page=1,
            end_page=1,
        )

    @pytest.fixture
    def table_with_merged_cells(self):
        """Таблица с объединёнными ячейками"""
        cells = [
            # Объединённая ячейка 2x2 в верхнем левом углу
            Cell(
                row=0,
                col=0,
                colspan=2,
                rowspan=2,
                bbox=BBox(x_min=0, y_min=0, x_max=200, y_max=100),
                value="Merged",
            ),
            # Обычные ячейки
            Cell(
                row=0,
                col=2,
                bbox=BBox(x_min=200, y_min=0, x_max=300, y_max=50),
                value="R0C2",
            ),
            Cell(
                row=1,
                col=2,
                bbox=BBox(x_min=200, y_min=50, x_max=300, y_max=100),
                value="R1C2",
            ),
            Cell(
                row=2,
                col=0,
                bbox=BBox(x_min=0, y_min=100, x_max=100, y_max=150),
                value="R2C0",
            ),
            Cell(
                row=2,
                col=1,
                bbox=BBox(x_min=100, y_min=100, x_max=200, y_max=150),
                value="R2C1",
            ),
            Cell(
                row=2,
                col=2,
                bbox=BBox(x_min=200, y_min=100, x_max=300, y_max=150),
                value="R2C2",
            ),
        ]
        return Table(
            id="table_merged",
            bbox=BBox(x_min=0, y_min=0, x_max=300, y_max=150),
            num_rows=3,
            num_cols=3,
            cells=cells,
            start_page=1,
            end_page=1,
        )

    @pytest.fixture
    def table_with_blobs(self):
        """Таблица с текстовыми блоками в ячейках"""
        cells = [
            Cell(
                row=0,
                col=0,
                bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
                value="Cell1",
                blobs=[
                    BBox(x_min=10, y_min=10, x_max=50, y_max=20),
                    BBox(x_min=10, y_min=25, x_max=50, y_max=35),
                ],
            ),
            Cell(
                row=0,
                col=1,
                bbox=BBox(x_min=100, y_min=0, x_max=200, y_max=50),
                value="Cell2",
                blobs=[BBox(x_min=110, y_min=10, x_max=150, y_max=25)],
            ),
            Cell(
                row=1,
                col=0,
                bbox=BBox(x_min=0, y_min=50, x_max=100, y_max=100),
                value="Cell3",
            ),
            Cell(
                row=1,
                col=1,
                bbox=BBox(x_min=100, y_min=50, x_max=200, y_max=100),
                value="Cell4",
                blobs=[BBox(x_min=110, y_min=60, x_max=150, y_max=80)],
            ),
        ]
        return Table(
            id="table_blobs",
            bbox=BBox(x_min=0, y_min=0, x_max=200, y_max=100),
            num_rows=2,
            num_cols=2,
            cells=cells,
            start_page=1,
            end_page=1,
        )

    @pytest.fixture
    def multipage_table(self):
        """Многостраничная таблица"""
        cells = [
            Cell(
                row=0,
                col=0,
                bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
                value="R0C0",
            ),
            Cell(
                row=1,
                col=0,
                bbox=BBox(x_min=0, y_min=50, x_max=100, y_max=100),
                value="R1C0",
            ),
        ]
        return Table(
            id="table_multipage",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=100),
            num_rows=2,
            num_cols=2,
            cells=cells,
            start_page=1,
            end_page=3,
        )

    @pytest.fixture
    def invalid_table(self):
        """Невалидная таблица (1 строка)"""
        return Table(
            id="invalid",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
            num_rows=1,
            num_cols=2,
            cells=[],
            start_page=1,
            end_page=1,
        )

    # Тесты свойств
    def test_area(self, simple_table):
        """Тест расчёта площади таблицы"""
        assert simple_table.area == 45000  # 300 * 150

    def test_width(self, simple_table):
        """Тест расчёта ширины таблицы"""
        assert simple_table.width == 300

    def test_height(self, simple_table):
        """Тест расчёта высоты таблицы"""
        assert simple_table.height == 150

    def test_center(self, simple_table):
        """Тест расчёта центра таблицы"""
        assert simple_table.center == (150.0, 75.0)

    # Тесты get_cell
    def test_get_cell_exists(self, simple_table):
        """Тест получения существующей ячейки"""
        cell = simple_table.get_cell(1, 1)
        assert cell is not None
        assert cell.row == 1
        assert cell.col == 1
        assert cell.value == "R1C1"

    def test_get_cell_not_exists(self, simple_table):
        """Тест получения несуществующей ячейки"""
        cell = simple_table.get_cell(10, 10)
        assert cell is None

    def test_get_cell_first(self, simple_table):
        """Тест получения первой ячейки"""
        cell = simple_table.get_cell(0, 0)
        assert cell is not None
        assert cell.value == "R0C0"

    def test_get_cell_last(self, simple_table):
        """Тест получения последней ячейки"""
        cell = simple_table.get_cell(2, 2)
        assert cell is not None
        assert cell.value == "R2C2"

    def test_cell_index_caching(self, simple_table):
        """Тест кэширования индекса ячеек"""
        # Первый вызов создаёт индекс
        cell1 = simple_table.get_cell(1, 1)
        # Второй вызов использует кэш
        cell2 = simple_table.get_cell(1, 1)
        assert cell1 is cell2

    # Тесты методов
    def test_padding(self, simple_table):
        """Тест добавления отступа к таблице"""
        padded = simple_table.padding(10)
        assert padded.bbox.x_min == -10
        assert padded.bbox.y_min == -10
        assert padded.bbox.x_max == 310
        assert padded.bbox.y_max == 160
        assert len(padded.cells) == len(simple_table.cells)

    def test_intersect(self):
        """Тест пересечения двух таблиц"""
        table1 = Table(
            id="t1",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=100),
            num_rows=2,
            num_cols=2,
            cells=[],
            start_page=1,
            end_page=1,
        )
        table2 = Table(
            id="t2",
            bbox=BBox(x_min=50, y_min=50, x_max=150, y_max=150),
            num_rows=2,
            num_cols=2,
            cells=[],
            start_page=1,
            end_page=1,
        )
        # Пересечение: 50x50 = 2500
        assert table1.intersect(table2) == 2500

    def test_intersect_no_overlap(self):
        """Тест пересечения без наложения"""
        table1 = Table(
            id="t1",
            bbox=BBox(x_min=0, y_min=0, x_max=50, y_max=50),
            num_rows=2,
            num_cols=2,
            cells=[],
            start_page=1,
            end_page=1,
        )
        table2 = Table(
            id="t2",
            bbox=BBox(x_min=100, y_min=100, x_max=150, y_max=150),
            num_rows=2,
            num_cols=2,
            cells=[],
            start_page=1,
            end_page=1,
        )
        assert table1.intersect(table2) == 0

    # Тесты get_rows
    def test_get_rows_simple(self, simple_table):
        """Тест группировки по строкам"""
        rows = simple_table.get_rows()
        assert len(rows) == 3
        assert len(rows[0]) == 3  # 3 ячейки в первой строке
        assert len(rows[1]) == 3
        assert len(rows[2]) == 3
        assert rows[0][0].value == "R0C0"
        assert rows[1][1].value == "R1C1"

    def test_get_rows_with_merged(self, table_with_merged_cells):
        """Тест группировки по строкам с объединёнными ячейками (без дублирования)"""
        rows = table_with_merged_cells.get_rows(include_merged=False)
        assert len(rows) == 3
        # Merged ячейка попадает только в строку 0
        assert len(rows[0]) == 2  # Merged + R0C2
        assert len(rows[1]) == 1  # R1C2
        assert len(rows[2]) == 3  # R2C0, R2C1, R2C2

    def test_get_rows_include_merged(self, table_with_merged_cells):
        """Тест группировки по строкам с дублированием merged ячеек"""
        rows = table_with_merged_cells.get_rows(include_merged=True)
        assert len(rows) == 3
        # Merged ячейка попадает в строки 0 и 1
        assert len(rows[0]) == 2  # Merged + R0C2
        assert len(rows[1]) == 2  # Merged + R1C2
        assert len(rows[2]) == 3  # R2C0, R2C1, R2C2

    # Тесты get_cols
    def test_get_cols_simple(self, simple_table):
        """Тест группировки по столбцам"""
        cols = simple_table.get_cols()
        assert len(cols) == 3
        assert len(cols[0]) == 3  # 3 ячейки в первом столбце
        assert len(cols[1]) == 3
        assert len(cols[2]) == 3
        assert cols[0][0].value == "R0C0"
        assert cols[1][1].value == "R1C1"

    def test_get_cols_with_merged(self, table_with_merged_cells):
        """Тест группировки по столбцам с объединёнными ячейками (без дублирования)"""
        cols = table_with_merged_cells.get_cols(include_merged=False)
        assert len(cols) == 3
        # Merged ячейка попадает только в столбец 0
        assert len(cols[0]) == 2  # Merged + R2C0
        assert len(cols[1]) == 1  # R2C1
        assert len(cols[2]) == 3  # R0C2, R1C2, R2C2

    def test_get_cols_include_merged(self, table_with_merged_cells):
        """Тест группировки по столбцам с дублированием merged ячеек"""
        cols = table_with_merged_cells.get_cols(include_merged=True)
        assert len(cols) == 3
        # Merged ячейка попадает в столбцы 0 и 1
        assert len(cols[0]) == 2  # Merged + R2C0
        assert len(cols[1]) == 2  # Merged + R2C1
        assert len(cols[2]) == 3  # R0C2, R1C2, R2C2

    # Тесты median_height_blobs_per_cells
    def test_median_height_blobs_empty(self, simple_table):
        """Тест медианной высоты для таблицы без блобов"""
        assert simple_table.median_height_blobs_per_cells() == 0.0

    def test_median_height_blobs_with_data(self, table_with_blobs):
        """Тест медианной высоты для таблицы с блобами"""
        median = table_with_blobs.median_height_blobs_per_cells()
        # Cell1: (10 + 10) / 2 = 10
        # Cell2: 15
        # Cell4: 20
        # Медиана из [10, 15, 20] = 15
        assert median == 15.0

    def test_median_height_blobs_single_cell(self):
        """Тест медианной высоты для одной ячейки с блобами"""
        cells = [
            Cell(
                row=0,
                col=0,
                bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
                blobs=[BBox(x_min=10, y_min=10, x_max=50, y_max=30)],  # height = 20
            )
        ]
        table = Table(
            id="single",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50),
            num_rows=2,
            num_cols=2,
            cells=cells,
            start_page=1,
            end_page=1,
        )
        assert table.median_height_blobs_per_cells() == 20.0

    # Тесты is_valid
    def test_is_valid_true(self, simple_table):
        """Тест валидности нормальной таблицы"""
        assert simple_table.is_valid()

    def test_is_valid_false_one_row(self, invalid_table):
        """Тест валидности таблицы с одной строкой"""
        assert not invalid_table.is_valid()

    def test_is_valid_false_one_col(self):
        """Тест валидности таблицы с одним столбцом"""
        table = Table(
            id="invalid",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=100),
            num_rows=2,
            num_cols=1,
            cells=[],
            start_page=1,
            end_page=1,
        )
        assert not table.is_valid()

    def test_is_valid_false_no_cells(self):
        """Тест валидности таблицы без ячеек"""
        table = Table(
            id="empty",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=100),
            num_rows=2,
            num_cols=2,
            cells=[],
            start_page=1,
            end_page=1,
        )
        assert not table.is_valid()

    # Тесты validate_structure
    def test_validate_structure_valid(self, simple_table):
        """Тест структурной валидности нормальной таблицы"""
        assert simple_table.validate_structure()

    def test_validate_structure_cell_out_of_bounds_row(self):
        """Тест структурной валидности с ячейкой за пределами строк"""
        cells = [
            Cell(
                row=5, col=0, bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50)
            )  # row=5 > num_rows=2
        ]
        table = Table(
            id="invalid",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=100),
            num_rows=2,
            num_cols=2,
            cells=cells,
            start_page=1,
            end_page=1,
        )
        assert not table.validate_structure()

    def test_validate_structure_cell_out_of_bounds_col(self):
        """Тест структурной валидности с ячейкой за пределами столбцов"""
        cells = [
            Cell(
                row=0, col=5, bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=50)
            )  # col=5 > num_cols=2
        ]
        table = Table(
            id="invalid",
            bbox=BBox(x_min=0, y_min=0, x_max=100, y_max=100),
            num_rows=2,
            num_cols=2,
            cells=cells,
            start_page=1,
            end_page=1,
        )
        assert not table.validate_structure()

    def test_validate_structure_merged_cell_overflow(self):
        """Тест структурной валидности с merged ячейкой, выходящей за границы"""
        cells = [
            Cell(
                row=1,
                col=1,
                colspan=2,
                rowspan=2,  # Выходит за границы 2x2 таблицы
                bbox=BBox(x_min=100, y_min=50, x_max=300, y_max=150),
            )
        ]
        table = Table(
            id="overflow",
            bbox=BBox(x_min=0, y_min=0, x_max=200, y_max=100),
            num_rows=2,
            num_cols=2,
            cells=cells,
            start_page=1,
            end_page=1,
        )
        assert not table.validate_structure()

    # Тесты multipage
    def test_multipage_table(self, multipage_table):
        """Тест многостраничной таблицы"""
        assert multipage_table.start_page == 1
        assert multipage_table.end_page == 3
        assert multipage_table.is_valid()

    # Тесты строкового представления
    def test_str(self, simple_table):
        """Тест строкового представления"""
        result = str(simple_table)
        assert "Table(id=table_1" in result
        assert "rows=3" in result
        assert "cols=3" in result
        assert "pages=1-1" in result

    def test_str_multipage(self, multipage_table):
        """Тест строкового представления многостраничной таблицы"""
        result = str(multipage_table)
        assert "pages=1-3" in result
