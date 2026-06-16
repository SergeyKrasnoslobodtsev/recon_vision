"""Сборка логических таблиц из сырых: детекция шапки дебет/кредит и объединение.

Заменяет DcColsResolver. Делает первый шаг пайплайна структуры документа:
детектирует шапку у корневой (головной) сырой таблицы, проставляет типы колонок
(DEBIT/CREDIT/OTHER) на канонических колонках логической таблицы и сшивает все
куски цепочки в одну LogicalTable, отображая локальные колонки каждого куска в
канонические по геометрии (перекрытию по оси X). Строка-шапка в данные не попадает.

Порядок в пайплайне: после continuation_linker, до row_splitter.
dc_cols выводится из типов колонок (см. dc_cols ниже), отдельно не хранится.

Допущения о сырых сущностях. Если в проекте интерфейс иной, правьте только
тонкие обёртки `_rows`, `_num_cols`, `_table_y` — остальной код от деталей Table
не зависит:
  * Table: .id: str, .continuation_of: str | None, .num_cols: int, .bbox: BBox,
           .get_rows() -> list[list[Cell]]  (строки сверху вниз; ячейки строки)
  * Page:  .tables: list[Table]
  * Cell:  .row, .col: int; .value: str | None; .bbox: BBox (.x_min/.x_max)
  * Текст уже нормализован в ВЕРХНИЙ регистр (см. text_normalizer) — поэтому
    ключевые слова заданы в верхнем регистре и значения не перенормализуются.
  * Номер страницы = позиция Page в списке `pages` (0-based).
"""

from __future__ import annotations

from loguru import logger

from vision_core.entities.cell import Cell
from vision_core.entities.logical_table import Column, ColumnType, LogicalTable, Row
from vision_core.entities.page import Page
from vision_core.entities.table import Table
from vision_core.exceptions import (
    DcColsInvalidPositionError,
    DcColsNotFoundError,
    DcColsResolutionError,
)

# Инвариант: текст уже в верхнем регистре. Если инвариант исчезнет — матчить
# через `kw in value.casefold().upper()` или хранить слова в обоих регистрах.
_DEBIT_KEYWORDS = {"ДЕБЕТ"}
_CREDIT_KEYWORDS = {"КРЕДИТ"}
_HEADER_SCAN_ROWS = 4
# Минимальная доля перекрытия колонки куска с канонической, ниже которой
# сопоставление считается неуверенным (берём ближайшую по центру + warning).
_MIN_OVERLAP_RATIO = 0.3


# --------------------------------------------------------------------------- #
# Тонкие обёртки над сырыми сущностями — единственное место, зависящее от Table
# --------------------------------------------------------------------------- #
def _rows(table: Table) -> list[list[Cell]]:
    return table.get_rows()


def _num_cols(table: Table) -> int:
    return table.num_cols


def _table_y(table: Table) -> float:
    return table.bbox.y_min


# --------------------------------------------------------------------------- #
# Детекция шапки
# --------------------------------------------------------------------------- #
def _match(value: str | None, keywords: set[str]) -> bool:
    return bool(value) and any(kw in value for kw in keywords)


def has_dc_header(table: Table) -> bool:
    """Есть ли в первых строках таблицы шапка дебет/кредит."""
    for row in _rows(table)[:_HEADER_SCAN_ROWS]:
        for cell in row:
            if _match(cell.value, _DEBIT_KEYWORDS) or _match(cell.value, _CREDIT_KEYWORDS):
                return True
    return False


def detect_dc_columns(table: Table) -> tuple[set[int], set[int], set[int]]:
    """Ищет колонки дебета и кредита плюс индексы строк-шапок.

    Возвращает (debit_cols, credit_cols, header_row_indices). Индексы строк —
    позиции в `get_rows()`, чтобы их можно было исключить из данных при сборке.

    Восстановление недостающей стороны выполняется ТОЛЬКО когда одна из сторон
    пуста (типичный OCR-промах по одному слову): дебет слева от кредита, кредит
    справа от дебета. Если найдены обе стороны — доверяем как есть и не
    достраиваем смежные пары, иначе при неполной/шумной детекции можно
    нафабриковать фантомные колонки.

    Raises:
        DcColsNotFoundError: ни дебет, ни кредит не найдены.
        DcColsInvalidPositionError: восстановленная колонка вне диапазона.
    """
    debit: set[int] = set()
    credit: set[int] = set()
    header_rows: set[int] = set()

    for ridx, row in enumerate(_rows(table)[:_HEADER_SCAN_ROWS]):
        for cell in row:
            if _match(cell.value, _DEBIT_KEYWORDS):
                debit.add(cell.col)
                header_rows.add(ridx)
            elif _match(cell.value, _CREDIT_KEYWORDS):
                credit.add(cell.col)
                header_rows.add(ridx)

    if not debit and not credit:
        raise DcColsNotFoundError(table_id=table.id)

    num_cols = _num_cols(table)
    if debit and not credit:
        for col in sorted(debit):
            partner = col + 1
            if partner in debit:
                continue
            if partner >= num_cols:
                raise DcColsInvalidPositionError(table_id=table.id, col=partner, num_cols=num_cols)
            credit.add(partner)
    elif credit and not debit:
        for col in sorted(credit):
            partner = col - 1
            if partner in credit:
                continue
            if partner < 0:
                raise DcColsInvalidPositionError(table_id=table.id, col=partner, num_cols=num_cols)
            debit.add(partner)

    return debit, credit, header_rows


# --------------------------------------------------------------------------- #
# Геометрия колонок: отображение локальных колонок куска в канонические
# --------------------------------------------------------------------------- #
def _column_xranges(table: Table) -> dict[int, tuple[float, float]]:
    """Диапазон [x_min, x_max] каждой колонки таблицы по bbox её ячеек."""
    ranges: dict[int, tuple[float, float]] = {}
    for row in _rows(table):
        for cell in row:
            x0, x1 = cell.bbox.x_min, cell.bbox.x_max
            if cell.col in ranges:
                lo, hi = ranges[cell.col]
                ranges[cell.col] = (min(lo, x0), max(hi, x1))
            else:
                ranges[cell.col] = (x0, x1)
    return ranges


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _map_columns(
    local: dict[int, tuple[float, float]],
    canonical: dict[int, tuple[float, float]],
    table_id: str,
) -> dict[int, int]:
    """Сопоставляет локальные колонки куска с каноническими по перекрытию X."""
    mapping: dict[int, int] = {}
    for lcol, lr in local.items():
        best_col: int | None = None
        best_ov = 0.0
        for ccol, cr in canonical.items():
            ov = _overlap(lr, cr)
            if ov > best_ov:
                best_col, best_ov = ccol, ov

        width = lr[1] - lr[0]
        weak = width > 0 and best_ov / width < _MIN_OVERLAP_RATIO
        if best_col is None or weak:
            lcenter = (lr[0] + lr[1]) / 2
            best_col = min(
                canonical,
                key=lambda cc: abs((canonical[cc][0] + canonical[cc][1]) / 2 - lcenter),
            )
            logger.warning(
                "Колонка %s таблицы %s сопоставлена по центру (слабое перекрытие по X)",
                lcol,
                table_id,
            )
        mapping[lcol] = best_col
    return mapping


def _moved(cell: Cell, row: int, col: int) -> Cell:
    """Копия ячейки с новыми каноническими координатами (без мутации оригинала)."""
    return Cell(
        row=row,
        col=col,
        colspan=cell.colspan,
        rowspan=cell.rowspan,
        value=cell.value,
        bbox=cell.bbox,
        blobs=cell.blobs,
    )


# --------------------------------------------------------------------------- #
# Сборщик
# --------------------------------------------------------------------------- #
class LogicalTableBuilder:
    """Строит LogicalTable из сырых таблиц, уже связанных в цепочки continuation_of."""

    def build(self, pages: list[Page]) -> list[LogicalTable]:
        page_of: dict[str, int] = {}
        tables: dict[str, Table] = {}
        for pidx, page in enumerate(pages):
            for table in page.tables:
                tables[table.id] = table
                page_of[table.id] = pidx

        chains = self._chains(tables, page_of)
        built: list[LogicalTable] = []
        failures: list[Exception] = []
        for chain in chains:
            try:
                built.append(self._build_one(chain, page_of))
            except (DcColsNotFoundError, DcColsInvalidPositionError) as exc:
                # Не падаем на первой же таблице: на плохом скане их может быть
                # несколько. Собираем все провалы и поднимаем единую ошибку ниже.
                failures.append(exc)

        if failures:
            raise DcColsResolutionError(
                table_id=", ".join(getattr(e, "table_id", "?") for e in failures),
                details="; ".join(str(e) for e in failures),
            )
        return built

    def _chains(self, tables: dict[str, Table], page_of: dict[str, int]) -> list[list[Table]]:
        """Группирует сырые таблицы в цепочки [голова, продолжение, ...] по порядку чтения."""
        children: dict[str, list[Table]] = {}
        for table in tables.values():
            if table.continuation_of is not None:
                children.setdefault(table.continuation_of, []).append(table)

        chains: list[list[Table]] = []
        for root in (t for t in tables.values() if t.continuation_of is None):
            chain: list[Table] = []
            stack = [root]
            while stack:
                cur = stack.pop()
                chain.append(cur)
                stack.extend(children.get(cur.id, []))
            chain.sort(key=lambda t: (page_of[t.id], _table_y(t)))
            chains.append(chain)
        return chains

    def _build_one(self, chain: list[Table], page_of: dict[str, int]) -> LogicalTable:
        head = next(t for t in chain if t.continuation_of is None)
        debit, credit, header_rows = detect_dc_columns(head)

        canonical_xr = _column_xranges(head)
        col_type = dict.fromkeys(canonical_xr, ColumnType.OTHER)
        for c in debit:
            col_type[c] = ColumnType.DEBIT
        for c in credit:
            col_type[c] = ColumnType.CREDIT

        rows_by_index: dict[int, list[Cell]] = {}
        global_row = 0
        for part in chain:
            # Кросс-проверка границы: продолжение со своей шапкой почти наверняка
            # означает ошибку группировки (это отдельная логическая таблица).
            if part is not head and has_dc_header(part):
                logger.warning(
                    "Продолжение %s содержит собственную шапку дебет/кредит — "
                    "вероятна ошибка группировки в continuation_linker",
                    part.id,
                )

            mapping = _map_columns(_column_xranges(part), canonical_xr, part.id)
            skip = header_rows if part is head else set()
            for ridx, row in enumerate(_rows(part)):
                if ridx in skip:
                    continue
                for cell in row:
                    canon_col = mapping.get(cell.col, cell.col)
                    rows_by_index.setdefault(global_row, []).append(_moved(cell, global_row, canon_col))
                global_row += 1

        logical_rows = [
            Row(index=r, cells=sorted(cells, key=lambda c: c.col)) for r, cells in sorted(rows_by_index.items())
        ]
        logical_cols = [Column(index=c, type=col_type[c]) for c in sorted(canonical_xr)]
        pages_in = [page_of[t.id] for t in chain]

        return LogicalTable(
            id=head.id,
            rows=logical_rows,
            columns=logical_cols,
            start_page=min(pages_in),
            end_page=max(pages_in),
        )
