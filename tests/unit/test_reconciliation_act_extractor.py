"""Тесты для ReconciliationActExtractor и вспомогательных функций."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.infrastructure.services.extractor.company_ext import (
    _Role,
    _assign_roles,
    _deduplicate_orgs,
    _extract_org_names,
    _find_working_pair,
    _normalize_text,
)
from app.infrastructure.services.structured_data_extractor import ReconciliationActExtractor


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_empty_string(self):
        assert _normalize_text("") == ""

    def test_latin_to_cyrillic(self):
        assert _normalize_text("PYCAЛ") == "РУСАЛ"

    def test_uppercase(self):
        assert _normalize_text("русал братск") == "РУСАЛ БРАТСК"

    def test_yo_to_e(self):
        assert _normalize_text("ОБЩЕСТВО") == "ОБЩЕСТВО"
        assert _normalize_text("ЁМКОСТЬ") == "ЕМКОСТЬ"

    def test_000_to_ooo(self):
        assert _normalize_text('000 "Ромашка"') == 'ООО "РОМАШКА "'

    def test_quotes_normalized(self):
        # «» и Unicode-кавычки приводятся к прямым, trailing space перед закрывающей
        assert _normalize_text('«БРАТСК»') == '"БРАТСК "'
        assert _normalize_text('\u201cБРАТСК\u201d') == '"БРАТСК "'

    def test_double_quotes_collapsed(self):
        assert _normalize_text('ОБЩЕСТВО""ФОРВАРД"') == 'ОБЩЕСТВО "ФОРВАРД "'

    def test_space_before_quote(self):
        assert _normalize_text('ОБЩЕСТВО"ФОРВАРД"') == 'ОБЩЕСТВО "ФОРВАРД "'

    def test_whitespace_collapsed(self):
        assert _normalize_text("РУСАЛ   БРАТСК") == "РУСАЛ БРАТСК"

    def test_combined_ocr_artifacts(self):
        result = _normalize_text('PAO "PYCAЛ БРАТCK"')
        assert result == 'РАО "РУСАЛ БРАТСК "'


# ---------------------------------------------------------------------------
# _extract_org_names
# ---------------------------------------------------------------------------

class TestExtractOrgNames:
    def test_ao_form(self):
        names = _extract_org_names('АО "БРАТСКОЕ ПРЕДПРИЯТИЕ"')
        assert names == ["БРАТСКОЕ ПРЕДПРИЯТИЕ, АО"]

    def test_pao_form(self):
        names = _extract_org_names('ПАО "РУСГИДРО"')
        assert names == ["РУСГИДРО, ПАО"]

    def test_fgup_form(self):
        names = _extract_org_names('ФГУП "ВНИИМ ИМ.Д.И.МЕНДЕЛЕЕВА"')
        assert names == ["ВНИИМ ИМ.Д.И.МЕНДЕЛЕЕВА, ФГУП"]

    def test_full_form_mapped_to_short(self):
        names = _extract_org_names('АКЦИОНЕРНОЕ ОБЩЕСТВО "БРАТСКОЕ ПРЕДПРИЯТИЕ"')
        assert names == ["БРАТСКОЕ ПРЕДПРИЯТИЕ, АО"]

    def test_rusal_without_form(self):
        names = _extract_org_names('"РУСАЛ БРАТСК"')
        assert names == ["РУСАЛ БРАТСК, "]

    def test_fix_rusal_glued_word(self):
        names = _extract_org_names('"РУСАЛБРАТСК"')
        assert names == ["РУСАЛ БРАТСК, "]

    def test_multiple_orgs(self):
        text = 'МЕЖДУ ПАО "РУСГИДРО" И АО "БРАТСКОЕ ПРЕДПРИЯТИЕ"'
        names = _extract_org_names(text)
        assert "РУСГИДРО, ПАО" in names
        assert "БРАТСКОЕ ПРЕДПРИЯТИЕ, АО" in names

    def test_deduplication_same_name(self):
        text = 'ПАО "РУСГИДРО" и снова ПАО "РУСГИДРО"'
        names = _extract_org_names(text)
        assert names.count("РУСГИДРО, ПАО") == 1

    def test_no_org_found(self):
        assert _extract_org_names("АКТ СВЕРКИ ЗА 2024 ГОД") == []

    def test_real_case_vniim_rusal(self):
        text = (
            'МЕЖДУ ВНИИМ - ФИЛИАЛ ФГУП "ВНИИМ ИМ.Д.И.МЕНДЕЛЕЕВА" '
            '(ИНН 7809022120) И "РУСАЛ КАНДАЛАКША" (ИНН 6612005052)'
        )
        names = _extract_org_names(text)
        assert any("ВНИИМ ИМ.Д.И.МЕНДЕЛЕЕВА" in n for n in names)
        assert any("РУСАЛ КАНДАЛАКША" in n for n in names)


# ---------------------------------------------------------------------------
# _deduplicate_orgs
# ---------------------------------------------------------------------------

class TestDeduplicateOrgs:
    def test_removes_substring_token(self):
        orgs = [
            "ФЕДЕРАЛЬНАЯ ГИДРОГЕНЕРИРУЮЩАЯ КОМПАНИЯ - РУСГИДРО, ПАО",
            "ОБЪЕДИНЕННАЯ КОМПАНИЯ РУСАЛ УРАЛЬСКИЙ АЛЮМИНИЙ, АО",
            "РУСГИДРО, ПАО",
            "ОБЪЕДИНЕННАЯ, АО",
        ]
        result = _deduplicate_orgs(orgs)
        tokens = [o.split(",")[0].strip() for o in result]
        assert "РУСГИДРО" not in tokens
        assert "ОБЪЕДИНЕННАЯ" not in tokens
        assert "ФЕДЕРАЛЬНАЯ ГИДРОГЕНЕРИРУЮЩАЯ КОМПАНИЯ - РУСГИДРО" in tokens
        assert "ОБЪЕДИНЕННАЯ КОМПАНИЯ РУСАЛ УРАЛЬСКИЙ АЛЮМИНИЙ" in tokens

    def test_no_duplicates_unchanged(self):
        orgs = ["РУСГИДРО, ПАО", "БРАТСК, АО"]
        assert _deduplicate_orgs(orgs) == orgs

    def test_empty_list(self):
        assert _deduplicate_orgs([]) == []


# ---------------------------------------------------------------------------
# _find_working_pair
# ---------------------------------------------------------------------------

class TestFindWorkingPair:
    def _events(self, text: str):
        from app.infrastructure.services.extractor.company_ext import _find_events
        return _find_events(text)

    def test_uses_mezhdu_anchor(self):
        text = 'МЕЖДУ ПАО "РУСГИДРО" И АО "РУСАЛ БРАТСК" СОСТАВИЛИ'
        orgs = ["РУСГИДРО, ПАО", "РУСАЛ БРАТСК, АО"]
        events = self._events(text)
        pair = _find_working_pair(text, orgs, events)
        assert set(pair) == {"РУСГИДРО, ПАО", "РУСАЛ БРАТСК, АО"}

    def test_returns_first_two_by_position(self):
        # РУСГИДРО стоит раньше БРАТСК в окне после МЕЖДУ
        text = 'МЕЖДУ ПАО "РУСГИДРО" И АО "БРАТСК" И ООО "ЛИШНЯЯ"'
        orgs = ["РУСГИДРО, ПАО", "БРАТСК, АО", "ЛИШНЯЯ, ООО"]
        events = self._events(text)
        pair = _find_working_pair(text, orgs, events)
        assert len(pair) == 2
        assert "ЛИШНЯЯ, ООО" not in pair

    def test_fallback_without_mezhdu(self):
        text = 'АКТ СВЕРКИ ПАО "РУСГИДРО" И АО "БРАТСК"'
        orgs = ["РУСГИДРО, ПАО", "БРАТСК, АО", "ЛИШНЯЯ, ООО"]
        events = self._events(text)
        pair = _find_working_pair(text, orgs, events)
        assert pair == ["РУСГИДРО, ПАО", "БРАТСК, АО"]

    def test_fallback_single_org(self):
        text = 'АКТ СВЕРКИ'
        orgs = ["РУСГИДРО, ПАО"]
        events = self._events(text)
        pair = _find_working_pair(text, orgs, events)
        assert pair == ["РУСГИДРО, ПАО"]


# ---------------------------------------------------------------------------
# _assign_roles
# ---------------------------------------------------------------------------

class TestAssignRoles:
    def test_rusal_always_buyer(self):
        text = 'МЕЖДУ АО "РУСАЛ БРАТСК" И ПАО "РУСГИДРО"'
        orgs = ["РУСАЛ БРАТСК, АО", "РУСГИДРО, ПАО"]
        roles = _assign_roles(text, orgs)
        assert roles["РУСАЛ БРАТСК, АО"] == _Role.BUYER
        assert roles["РУСГИДРО, ПАО"] == _Role.SELLER

    def test_rusal_overrides_explicit_seller_anchor(self):
        text = (
            'МЕЖДУ АО "РУСАЛ БРАТСК" И ПАО "РУСГИДРО" '
            'ОТ ПРОДАВЦА АО "РУСАЛ БРАТСК" ОТ ПОКУПАТЕЛЯ ПАО "РУСГИДРО"'
        )
        orgs = ["РУСАЛ БРАТСК, АО", "РУСГИДРО, ПАО"]
        roles = _assign_roles(text, orgs)
        assert roles["РУСАЛ БРАТСК, АО"] == _Role.BUYER
        assert roles["РУСГИДРО, ПАО"] == _Role.SELLER

    def test_explicit_buyer_anchor(self):
        text = (
            'МЕЖДУ ПАО "ЭНЕРГО" И ПАО "БОГУЧАНСКАЯ ГЭС" '
            'ОТ ПРОДАВЦА ПАО "БОГУЧАНСКАЯ ГЭС" ОТ ПОКУПАТЕЛЯ ПАО "ЭНЕРГО"'
        )
        orgs = ["ЭНЕРГО, ПАО", "БОГУЧАНСКАЯ ГЭС, ПАО"]
        roles = _assign_roles(text, orgs)
        assert roles["ЭНЕРГО, ПАО"] == _Role.BUYER
        assert roles["БОГУЧАНСКАЯ ГЭС, ПАО"] == _Role.SELLER

    def test_explicit_seller_anchor(self):
        text = (
            'МЕЖДУ ПАО "БОГУЧАНСКАЯ ГЭС" И ПАО "ЭНЕРГО" '
            'ОТ ПРОДАВЦА ПАО "БОГУЧАНСКАЯ ГЭС" ОТ ПОКУПАТЕЛЯ ПАО "ЭНЕРГО"'
        )
        orgs = ["БОГУЧАНСКАЯ ГЭС, ПАО", "ЭНЕРГО, ПАО"]
        roles = _assign_roles(text, orgs)
        assert roles["БОГУЧАНСКАЯ ГЭС, ПАО"] == _Role.SELLER
        assert roles["ЭНЕРГО, ПАО"] == _Role.BUYER

    def test_positional_fallback_first_seller_second_buyer(self):
        text = 'АКТ СВЕРКИ ПАО "ЭНЕРГО" И ПАО "ФОРВАРД"'
        orgs = ["ЭНЕРГО, ПАО", "ФОРВАРД, ПАО"]
        roles = _assign_roles(text, orgs)
        assert roles["ЭНЕРГО, ПАО"] == _Role.SELLER
        assert roles["ФОРВАРД, ПАО"] == _Role.BUYER

    def test_symmetry_after_rusal(self):
        text = 'АКТ СВЕРКИ АО "РУСАЛ КАНДАЛАКША" И ФГУП "ВНИИМ"'
        orgs = ["РУСАЛ КАНДАЛАКША, АО", "ВНИИМ, ФГУП"]
        roles = _assign_roles(text, orgs)
        assert roles["РУСАЛ КАНДАЛАКША, АО"] == _Role.BUYER
        assert roles["ВНИИМ, ФГУП"] == _Role.SELLER

    def test_real_case_vniim_rusal_kand(self):
        text = (
            "АКТ СВЕРКИ ВЗАИМНЫХ РАСЧЕТОВ ЗА ПЕРИОД: 2024 Г. "
            'МЕЖДУ ВНИИМ - ФИЛИАЛ ФГУП "ВНИИМ ИМ.Д.И.МЕНДЕЛЕЕВА" '
            "(ИНН 7809022120) И "
            '"РУСАЛ КАНДАЛАКША" (ИНН 6612005052) '
            'МЫ, НИЖЕПОДПИСАВШИЕСЯ, ГЛАВНЫЙ БУХГАЛТЕР ВНИИМ - ФИЛИАЛ ФГУП "ВНИИМ ИМ.Д.И.МЕНДЕЛЕЕВА" '
            "КОМИНА НАТАЛЬЯ ВЛАДИМИРОВНА, С ОДНОЙ СТОРОНЫ, "
            'И "РУСАЛ КАНДАЛАКША", С ДРУГОЙ СТОРОНЫ'
        )
        orgs = ["ВНИИМ ИМ.Д.И.МЕНДЕЛЕЕВА, ФГУП", "РУСАЛ КАНДАЛАКША, "]
        roles = _assign_roles(text, orgs)
        assert roles["РУСАЛ КАНДАЛАКША, "] == _Role.BUYER
        assert roles["ВНИИМ ИМ.Д.И.МЕНДЕЛЕЕВА, ФГУП"] == _Role.SELLER

    def test_real_case_rushydro_rusal_ural(self):
        text = (
            "АКТ СВЕРКИ "
            'МЕЖДУ ПАО "ФЕДЕРАЛЬНАЯ ГИДРОГЕНЕРИРУЮЩАЯ КОМПАНИЯ - РУСГИДРО" '
            'И АО "ОБЪЕДИНЕННАЯ КОМПАНИЯ РУСАЛ УРАЛЬСКИЙ АЛЮМИНИЙ" '
            'ОТ ПРОДАВЦА ПАО "РУСГИДРО" '
            'ОТ ПОКУПАТЕЛЯ АО "ОБЪЕДИНЕННАЯ КОМПАНИЯ РУСАЛ УРАЛЬСКИЙ АЛЮМИНИЙ"'
        )
        orgs = [
            "ФЕДЕРАЛЬНАЯ ГИДРОГЕНЕРИРУЮЩАЯ КОМПАНИЯ - РУСГИДРО, ПАО",
            "ОБЪЕДИНЕННАЯ КОМПАНИЯ РУСАЛ УРАЛЬСКИЙ АЛЮМИНИЙ, АО",
        ]
        roles = _assign_roles(text, orgs)
        assert roles["ОБЪЕДИНЕННАЯ КОМПАНИЯ РУСАЛ УРАЛЬСКИЙ АЛЮМИНИЙ, АО"] == _Role.BUYER
        assert roles["ФЕДЕРАЛЬНАЯ ГИДРОГЕНЕРИРУЮЩАЯ КОМПАНИЯ - РУСГИДРО, ПАО"] == _Role.SELLER

    def test_empty_orgs_returns_empty(self):
        assert _assign_roles("текст", []) == {}

    def test_single_org_gets_seller_by_positional(self):
        roles = _assign_roles("АКТ СВЕРКИ", ["ЭНЕРГО, ПАО"])
        assert roles["ЭНЕРГО, ПАО"] == _Role.SELLER


# ---------------------------------------------------------------------------
# ReconciliationActExtractor.extract (интеграция с моком Document)
# ---------------------------------------------------------------------------

def _make_document(paragraph_texts: list[str], cell_header_texts: list[str] | None = None):
    """Строит мок Document с заданными параграфами и текстами заголовка dc-колонок."""
    paragraphs = [MagicMock(text=t) for t in paragraph_texts]

    tables = []
    if cell_header_texts:
        cells = []
        for i, text in enumerate(cell_header_texts):
            cell = MagicMock()
            cell.value = text
            cell.row = 0
            cell.col = i
            cell.colspan = 1
            cells.append(cell)

        table = MagicMock()
        table.continuation_of = None
        table.dc_cols = {i for i in range(len(cell_header_texts))}
        table.get_dc_header_row.return_value = 1
        table.get_rows.return_value = [cells]
        tables.append(table)

    page = MagicMock()
    page.paragraphs = paragraphs
    page.tables = tables

    document = MagicMock()
    document.pages = [page]
    return document


class TestReconciliationActExtractor:
    @pytest.mark.asyncio
    async def test_extracts_buyer_and_seller_from_paragraphs(self):
        text = (
            'АКТ СВЕРКИ МЕЖДУ ПАО "РУСГИДРО" '
            'И АО "РУСАЛ БРАТСК" ОТ ПРОДАВЦА ПАО "РУСГИДРО" '
            'ОТ ПОКУПАТЕЛЯ АО "РУСАЛ БРАТСК"'
        )
        document = _make_document([text])
        result = await ReconciliationActExtractor().extract(document)
        assert "РУСАЛ БРАТСК" in result.buyer
        assert "РУСГИДРО" in result.seller

    @pytest.mark.asyncio
    async def test_prefers_orgs_from_cell_headers_when_two_found(self):
        cell_texts = ['ПО ДАННЫМ ПАО "РУСГИДРО"', 'ПО ДАННЫМ АО "РУСАЛ БРАТСК"']
        paragraph_text = "АКТ СВЕРКИ МЕЖДУ ПАО РУСГИДРО И АО РУСАЛ БРАТСК"
        document = _make_document([paragraph_text], cell_texts)
        result = await ReconciliationActExtractor().extract(document)
        assert "РУСАЛ БРАТСК" in result.buyer
        assert "РУСГИДРО" in result.seller

    @pytest.mark.asyncio
    async def test_returns_empty_strings_when_no_orgs_found(self):
        document = _make_document(["АКТ СВЕРКИ ЗА 2024 ГОД"])
        result = await ReconciliationActExtractor().extract(document)
        assert result.seller == ""
        assert result.buyer == ""

    @pytest.mark.asyncio
    async def test_multipage_paragraphs_joined(self):
        page1 = MagicMock()
        page1.paragraphs = [MagicMock(text='АКТ СВЕРКИ МЕЖДУ ПАО "РУСГИДРО"')]
        page1.tables = []

        page2 = MagicMock()
        page2.paragraphs = [MagicMock(text='И АО "РУСАЛ БРАТСК" ОТ ПРОДАВЦА ПАО "РУСГИДРО" ОТ ПОКУПАТЕЛЯ АО "РУСАЛ БРАТСК"')]
        page2.tables = []

        document = MagicMock()
        document.pages = [page1, page2]

        result = await ReconciliationActExtractor().extract(document)
        assert "РУСАЛ БРАТСК" in result.buyer
        assert "РУСГИДРО" in result.seller

    @pytest.mark.asyncio
    async def test_table_without_dc_cols_skipped(self):
        table = MagicMock()
        table.continuation_of = None
        table.dc_cols = set()
        table.get_dc_header_row.return_value = -1

        page = MagicMock()
        page.paragraphs = [MagicMock(text='МЕЖДУ ПАО "РУСГИДРО" И АО "РУСАЛ БРАТСК"')]
        page.tables = [table]

        document = MagicMock()
        document.pages = [page]

        result = await ReconciliationActExtractor().extract(document)
        assert result.buyer != "" or result.seller != ""

    @pytest.mark.asyncio
    async def test_continuation_table_skipped(self):
        table = MagicMock()
        table.continuation_of = "table-1"

        page = MagicMock()
        page.paragraphs = [MagicMock(text='МЕЖДУ ПАО "РУСГИДРО" И АО "РУСАЛ БРАТСК"')]
        page.tables = [table]

        document = MagicMock()
        document.pages = [page]

        result = await ReconciliationActExtractor().extract(document)
        assert "РУСАЛ БРАТСК" in result.buyer
