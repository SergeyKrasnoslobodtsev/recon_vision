"""Тесты токенизатора: даты, организации, суммы, перекрытия."""

import pytest

from extractor.tokenize import (
    AnyReference,
    CurrencyReference,
    DateReference,
    OrganizationReference,
    tokenize,
)


def dates(refs: list[AnyReference]) -> list[DateReference]:
    return [r for r in refs if isinstance(r, DateReference)]


def orgs(refs: list[AnyReference]) -> list[OrganizationReference]:
    return [r for r in refs if isinstance(r, OrganizationReference)]


def currencies(refs: list[AnyReference]) -> list[CurrencyReference]:
    return [r for r in refs if isinstance(r, CurrencyReference)]


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


class TestDates:
    def test_numeric_date(self):
        result = dates(tokenize("САЛЬДО НА 01.07.2023 ГОДА"))
        assert len(result) == 1
        assert result[0].date == "01.07.2023"
        assert result[0].date_end is None

    def test_numeric_date_slash(self):
        result = dates(tokenize("ОТ 31/12/2022"))
        assert result[0].date == "31.12.2022"

    def test_ru_date_full(self):
        result = dates(tokenize("САЛЬДО НА 1 ИЮЛЯ 2023 Г."))
        assert len(result) == 1
        assert result[0].date == "01.07.2023"

    def test_ru_date_no_suffix(self):
        result = dates(tokenize("НА 30 СЕНТЯБРЯ 2023"))
        assert result[0].date == "30.09.2023"

    @pytest.mark.parametrize(
        "month_str, expected_num",
        [
            ("ЯНВАРЯ", "01"),
            ("ФЕВРАЛЯ", "02"),
            ("МАРТА", "03"),
            ("АПРЕЛЯ", "04"),
            ("МАЯ", "05"),
            ("МАЙ", "05"),
            ("ИЮНЯ", "06"),
            ("ИЮЛЯ", "07"),
            ("АВГУСТА", "08"),
            ("СЕНТЯБРЯ", "09"),
            ("ОКТЯБРЯ", "10"),
            ("НОЯБРЯ", "11"),
            ("ДЕКАБРЯ", "12"),
        ],
    )
    def test_ru_date_all_months(self, month_str, expected_num):
        result = dates(tokenize(f"НА 15 {month_str} 2024"))
        assert result[0].date == f"15.{expected_num}.2024"

    @pytest.mark.parametrize(
        "q_str, start, end",
        [
            ("1 КВАРТАЛ 2023", "01.01.2023", "31.03.2023"),
            ("2 КВАРТАЛ 2023", "01.04.2023", "30.06.2023"),
            ("3 КВАРТАЛ 2023", "01.07.2023", "30.09.2023"),
            ("4 КВАРТАЛ 2023", "01.10.2023", "31.12.2023"),
            ("III КВАРТАЛ 2024", "01.07.2024", "30.09.2024"),
        ],
    )
    def test_quarter(self, q_str, start, end):
        result = dates(tokenize(q_str))
        assert len(result) == 1
        assert result[0].date == start
        assert result[0].date_end == end

    def test_year(self):
        result = dates(tokenize("ЗА 2023 ГОД"))
        assert len(result) == 1
        assert result[0].date == "01.01.2023"
        assert result[0].date_end == "31.12.2023"

    def test_token_positions(self):
        text = "НА 01.07.2023 ГОДА"
        result = dates(tokenize(text))
        ref = result[0]
        assert text[ref.token.start : ref.token.end] == ref.token.text
        assert ref.token.text == "01.07.2023"


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


class TestOrganizations:
    def test_ooo_with_form(self):
        result = orgs(tokenize('ООО "РОМАШКА"'))
        assert len(result) == 1
        assert result[0].name == "РОМАШКА"
        assert result[0].org_form == "ООО"

    def test_ao_with_form(self):
        result = orgs(tokenize('АО "СБЕРБАНК"'))
        assert result[0].org_form == "АО"
        assert result[0].name == "СБЕРБАНК"

    def test_rusal_without_form(self):
        result = orgs(tokenize('"РУСАЛ БРАТСК"'))
        assert len(result) == 1
        assert result[0].name == "РУСАЛ БРАТСК"
        assert result[0].org_form is None

    def test_rusal_name_space_fix(self):
        # РУСАЛБРАТСК -> РУСАЛ БРАТСК (пробел после РУСАЛ)
        result = orgs(tokenize('"РУСАЛБРАТСК"'))
        assert result[0].name == "РУСАЛ БРАТСК"

    def test_rusal_inside_ooo(self):
        # РУСАЛ внутри ООО: org_form должна быть ООО
        result = orgs(tokenize('ООО "РУСАЛ ЭНЕРГОСБЫТ"'))
        assert result[0].org_form == "ООО"
        assert result[0].name == "РУСАЛ ЭНЕРГОСБЫТ"

    def test_ooo_with_dot_before_quote(self):
        # ООО. "НАЗВАНИЕ" — точка между формой и кавычкой (OCR-артефакт)
        result = orgs(tokenize('ООО. "ЮНИГРИН ПАУЗР"'))
        assert len(result) == 1
        assert result[0].org_form == "ООО"
        assert result[0].name == "ЮНИГРИН ПАУЗР"

    def test_full_form_with_ocr_garbage_word(self):
        # "Я" — обрывок слова "ОБЩЕСТВА", OCR-артефакт между формой и кавычкой
        result = orgs(tokenize('АКЦИОНЕРНОЕ ОБЩЕСТВО Я "ОБЪЕДИНЕННАЯ КОМПАНИЯ РУСАЛ УРАЛЬСКИЙ АЛЮМИНИЙ"'))
        assert len(result) == 1
        assert result[0].org_form == "АО"
        assert result[0].name == "ОБЪЕДИНЕННАЯ КОМПАНИЯ РУСАЛ УРАЛЬСКИЙ АЛЮМИНИЙ"

    @pytest.mark.parametrize(
        "full_form, expected_abbr",
        [
            ("АКЦИОНЕРНОЕ ОБЩЕСТВО", "АО"),
            ("ОТКРЫТОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО", "ОАО"),
            ("ЗАКРЫТОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО", "ЗАО"),
            ("ПУБЛИЧНОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО", "ПАО"),
            ("ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ", "ООО"),
            ("ИНДИВИДУАЛЬНЫЙ ПРЕДПРИНИМАТЕЛЬ", "ИП"),
            ("ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ УНИТАРНОЕ ПРЕДПРИЯТИЕ", "ФГУП"),
        ],
    )
    def test_full_org_form_recognized(self, full_form, expected_abbr):
        result = orgs(tokenize(f'{full_form} "РОМАШКА"'))
        assert len(result) == 1
        assert result[0].name == "РОМАШКА"
        assert result[0].org_form == expected_abbr

    def test_token_positions(self):
        text = 'МЕЖДУ ООО "РОМАШКА" И АО "ВАСИЛЁК"'
        result = orgs(tokenize(text))
        assert len(result) == 2
        for ref in result:
            assert text[ref.token.start : ref.token.end] == ref.token.text


# ---------------------------------------------------------------------------
# Currencies
# ---------------------------------------------------------------------------


class TestCurrencies:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("23 035 017,97", 23035017.97),
            ("47 761,70", 47761.70),
            ("0,00", 0.0),
            ("1 000,00", 1000.0),
        ],
    )
    def test_parse(self, text, expected):
        result = currencies(tokenize(text))
        assert len(result) == 1
        assert result[0].value == pytest.approx(expected)

    def test_token_positions(self):
        text = "ИТОГО 23 035 017,97 РУБЛЕЙ"
        result = currencies(tokenize(text))
        ref = result[0]
        assert text[ref.token.start : ref.token.end] == ref.token.text


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------


class TestOverlaps:
    def test_date_beats_currency(self):
        # "01.07.2023" — длиннее чем потенциальный "01.07" как валюта
        result = tokenize("01.07.2023")
        assert len(result) == 1
        assert isinstance(result[0], DateReference)

    def test_ooo_beats_rusal_inner(self):
        # ООО "РУСАЛ Х" — _RE_ORG длиннее _RE_RUSAL, должен победить
        result = orgs(tokenize('ООО "РУСАЛ БРАТСК"'))
        assert len(result) == 1
        assert result[0].org_form == "ООО"

    def test_sorted_by_position(self):
        text = 'ООО "РОМАШКА" САЛЬДО НА 01.01.2023 ИТОГО 100 000,00'
        result = tokenize(text)
        starts = [r.token.start for r in result]
        assert starts == sorted(starts)
