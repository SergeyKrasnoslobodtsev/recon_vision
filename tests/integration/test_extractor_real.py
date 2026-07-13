"""Интеграционные тесты извлечения данных акта сверки на реальных PDF."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.infrastructure.services.structured_data_extractor import ReconciliationActExtractor
from vision_core.pipelines.build_document import DocumentBuildPipeline


@pytest.mark.asyncio
async def test_extract_real(pdf_file: Path):
    document = DocumentBuildPipeline().build(pdf_file.read_bytes())
    result = await ReconciliationActExtractor().extract(document)

    debit_by_row = {(e.row_reference.id_table, e.row_reference.id_row): e for e in result.debit}
    credit_by_row = {(e.row_reference.id_table, e.row_reference.id_row): e for e in result.credit}
    all_keys = sorted(debit_by_row.keys() | credit_by_row.keys(), key=lambda k: (k[0], int(k[1])))

    print(f"\n--- {pdf_file.name} ---")
    print(f"seller : {result.seller}")
    print(f"buyer  : {result.buyer}")
    print(f"period : {result.period}")
    print(f"{'ref':<12}  {'debit':>15}  {'credit':>15}  record")
    for key in all_keys:
        d = debit_by_row.get(key)
        c = credit_by_row.get(key)
        ref = f"{key[0]}:r{key[1]}"
        dval = f"{d.value:>15.2f}" if d else f"{'':>15}"
        cval = f"{c.value:>15.2f}" if c else f"{'':>15}"
        record = (d or c).record
        print(f"{ref:<12}  {dval}  {cval}  {record!r}")

    assert result.seller, f"{pdf_file.name}: продавец не определён"
    assert result.buyer, f"{pdf_file.name}: покупатель не определён"
    for entry in result.debit + result.credit:
        assert entry.value >= 0, f"{pdf_file.name}: отрицательное значение {entry.value}"
    assert result.debit or result.credit, f"{pdf_file.name}: не извлечено ни одной записи дебета или кредита"
