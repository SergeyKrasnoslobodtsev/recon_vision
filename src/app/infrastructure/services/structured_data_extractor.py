"""Реализует извлечение данных акта сверки из канонического документа."""

from __future__ import annotations

from app.domain.entities.reconciliation_data import ReconciliationData
from app.infrastructure.services.extractor.company_ext import extract_companies
from app.infrastructure.services.extractor.dc_ext import extract_dc
from app.infrastructure.services.extractor.period_ext import extract_period
from vision_core.entities.document import Document


class ReconciliationActExtractor:
    """Оркестрирует извлечение данных акта сверки из Document."""

    async def extract(self, document: Document) -> ReconciliationData:
        companies = extract_companies(document)
        period    = extract_period(document)
        debit, credit = extract_dc(document, companies)

        sellers = [c for c in companies if c.role == "seller"]
        buyers  = [c for c in companies if c.role == "buyer"]

        return ReconciliationData(
            seller=sellers[0].display_name if sellers else "",
            buyer=buyers[0].display_name if buyers else "",
            period=period,
            debit=debit,
            credit=credit,
        )
