"""Интеграционный тест токенизатора на реальных PDF-документах.

Показывает структуру документа с аннотированными токенами прямо в тексте.
Запуск: pytest tests/extractor/integration/ -v -s
"""

from __future__ import annotations

from pathlib import Path

from extractor.normalize import transform_text
from extractor.tokenize import (
    AnyReference,
    CurrencyReference,
    DateReference,
    DigitalReference,
    OrganizationReference,
    tokenize,
)
from vision_core.pipelines.build_document import DocumentBuildPipeline

# ---------------------------------------------------------------------------
# Аннотирование — токены оборачиваются символами прямо в тексте
# ---------------------------------------------------------------------------


def _annotate(normalized: str, tokens: list[AnyReference]) -> str:
    """Вставляет маркеры токенов прямо в нормализованный текст."""
    parts: list[str] = []
    prev = 0
    for ref in tokens:
        parts.append(normalized[prev : ref.token.start])
        if isinstance(ref, DateReference):
            end = f"..{ref.date_end}" if ref.date_end else ""
            parts.append(f"[DATE:{ref.date}{end}]")
        elif isinstance(ref, OrganizationReference):
            form = f"/{ref.org_form}" if ref.org_form else ""
            parts.append(f"[ORG:{ref.name}{form}]")
        elif isinstance(ref, CurrencyReference):
            parts.append(f"[CUR:{ref.value:.2f}]")
        elif isinstance(ref, DigitalReference):
            parts.append(f"[NUM:{int(ref.value)}]")
        prev = ref.token.end
    parts.append(normalized[prev:])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Таблица — ячейки в формате R{row}:C{col}
# ---------------------------------------------------------------------------


def _print_table(table, page_idx: int) -> list[AnyReference]:
    cont = f"-> {table.continuation_of}" if table.continuation_of else "корень"
    header_row = table.get_dc_header_row()
    rows = table.get_rows()

    print(
        f"\n  [стр {page_idx}] Таблица {table.id}"
        f"  {table.num_rows}x{table.num_cols}"
        f"  dc_cols={sorted(table.dc_cols)}"
        f"  header_row={header_row}"
        f"  ({cont})"
    )

    all_tokens: list[AnyReference] = []

    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            raw = cell.value or ""
            if not raw.strip():
                continue
            norm = transform_text(raw)
            tokens = tokenize(norm)
            all_tokens.extend(tokens)
            text = _annotate(norm, tokens) if tokens else norm
            print(f"  R{row_idx + 1}:C{col_idx + 1} - {text}")

    return all_tokens


# ---------------------------------------------------------------------------
# Тест
# ---------------------------------------------------------------------------


def test_tokenize_real(pdf_file: Path) -> None:
    document = DocumentBuildPipeline().build(pdf_file.read_bytes())

    page_count = len(document.pages)
    table_count = sum(len(p.tables) for p in document.pages)
    para_count = sum(len(p.paragraphs) for p in document.pages)

    print(f"\n{'=' * 80}")
    print(f"  {pdf_file.name}")
    print(f"  страниц: {page_count}  таблиц: {table_count}  абзацев: {para_count}")
    print(f"{'=' * 80}")

    # --- Абзацы ---
    all_tokens: list[AnyReference] = []
    print("\n--- АБЗАЦЫ ---")
    for page_idx, page in enumerate(document.pages):
        for p in page.paragraphs:
            if not p.text.strip():
                continue
            norm = transform_text(p.text)
            tokens = tokenize(norm)
            all_tokens.extend(tokens)
            annotated = _annotate(norm, tokens) if tokens else norm
            print(f"  [стр {page_idx}] {annotated}")

    # --- Таблицы ---
    print("\n--- ТАБЛИЦЫ ---")
    for page_idx, page in enumerate(document.pages):
        for table in page.tables:
            table_tokens = _print_table(table, page_idx)
            all_tokens.extend(table_tokens)

    # --- Итог ---
    org_tokens = [t for t in all_tokens if isinstance(t, OrganizationReference)]
    date_tokens = [t for t in all_tokens if isinstance(t, DateReference)]
    curr_tokens = [t for t in all_tokens if isinstance(t, CurrencyReference)]
    num_tokens = [t for t in all_tokens if isinstance(t, DigitalReference)]

    print("\n--- ИТОГ ---")
    print(f"  орг   : {len(org_tokens)}")
    print(f"  даты  : {len(date_tokens)}")
    print(f"  суммы : {len(curr_tokens)}")
    print(f"  числа : {len(num_tokens)}")

    assert all_tokens, f"{pdf_file.name}: не найдено ни одного токена"
