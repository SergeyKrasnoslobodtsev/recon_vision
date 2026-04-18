"""Токенизатор: находит ссылки на даты, организации и суммы в нормализованном тексте."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Token base
# ---------------------------------------------------------------------------


@dataclass
class Token:
    """Ссылка на фрагмент нормализованного текста."""

    start: int
    end: int
    text: str


# ---------------------------------------------------------------------------
# Reference types
# ---------------------------------------------------------------------------


@dataclass
class DateReference:
    """Ссылка на дату в тексте документа."""

    token: Token
    date: str  # DD.MM.YYYY
    date_end: str | None = None  # DD.MM.YYYY — для диапазонов (квартал, год)


@dataclass
class OrganizationReference:
    """Ссылка на организацию в тексте документа."""

    token: Token
    name: str
    org_form: str | None = None  # ООО, АО, ПАО, ... или None


@dataclass
class CurrencyReference:
    """Ссылка на денежное значение в тексте документа."""

    token: Token
    value: float


@dataclass
class DigitalReference:
    """Ссылка на цифровое значение в тексте документа."""

    token: Token
    value: str


AnyReference = DateReference | OrganizationReference | CurrencyReference | DigitalReference


# ---------------------------------------------------------------------------
# Date recognition
# ---------------------------------------------------------------------------

_MONTHS_RU = (
    r"ЯНВАР[А-ЯЁ]*|ФЕВРАЛ[А-ЯЁ]*|МАРТ[А-ЯЁ]*|АПРЕЛ[А-ЯЁ]*"
    r"|МА[А-ЯЁ]+|ИЮН[А-ЯЁ]*|ИЮЛ[А-ЯЁ]*|АВГУСТ[А-ЯЁ]*"
    r"|СЕНТЯБР[А-ЯЁ]*|ОКТЯБР[А-ЯЁ]*|НОЯБР[А-ЯЁ]*|ДЕКАБР[А-ЯЁ]*"
)
# Порядок важен: МАРТ проверяется до МА, чтобы не перехватить март как май
_MONTH_NUM: list[tuple[str, str]] = [
    ("ЯНВАР", "01"),
    ("ФЕВРАЛ", "02"),
    ("МАРТ", "03"),
    ("АПРЕЛ", "04"),
    ("МА", "05"),
    ("ИЮН", "06"),
    ("ИЮЛ", "07"),
    ("АВГУСТ", "08"),
    ("СЕНТЯБР", "09"),
    ("ОКТЯБР", "10"),
    ("НОЯБР", "11"),
    ("ДЕКАБР", "12"),
]
_QUARTER_BOUNDS: dict[str, tuple[str, str]] = {
    "1": ("01.01", "31.03"),
    "I": ("01.01", "31.03"),
    "2": ("01.04", "30.06"),
    "II": ("01.04", "30.06"),
    "3": ("01.07", "30.09"),
    "III": ("01.07", "30.09"),
    "4": ("01.10", "31.12"),
    "IV": ("01.10", "31.12"),
}

_RE_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b")
_RE_RU_DATE = re.compile(r"\b(\d{1,2})\s+(" + _MONTHS_RU + r")\s+(\d{4})(?:\s*Г\.?)?")
_RE_QUARTER = re.compile(r"\b([1-4]|I{1,3}V?|VI{0,3})\s+КВАРТАЛ[А-ЯЁ]*\s+(\d{4})\b")
_RE_YEAR = re.compile(r"\bЗА\s+(\d{4})\s+ГОД\b")


def _month_num(name: str) -> str:
    for prefix, num in _MONTH_NUM:
        if name.startswith(prefix):
            return num
    return "01"


def _fmt(d: str, m: str, y: str) -> str:
    return f"{int(d):02d}.{int(m):02d}.{'20' + y if len(y) == 2 else y}"


def _find_dates(text: str) -> list[DateReference]:
    refs: list[DateReference] = []

    for m in _RE_NUMERIC_DATE.finditer(text):
        refs.append(
            DateReference(
                token=Token(m.start(), m.end(), m.group()),
                date=_fmt(m.group(1), m.group(2), m.group(3)),
            )
        )

    for m in _RE_RU_DATE.finditer(text):
        refs.append(
            DateReference(
                token=Token(m.start(), m.end(), m.group()),
                date=f"{int(m.group(1)):02d}.{_month_num(m.group(2))}.{m.group(3)}",
            )
        )

    for m in _RE_QUARTER.finditer(text):
        q = m.group(1).upper().lstrip("0") or "1"
        year = m.group(2)
        bounds = _QUARTER_BOUNDS.get(q)
        if bounds:
            refs.append(
                DateReference(
                    token=Token(m.start(), m.end(), m.group()),
                    date=f"{bounds[0]}.{year}",
                    date_end=f"{bounds[1]}.{year}",
                )
            )

    for m in _RE_YEAR.finditer(text):
        year = m.group(1)
        refs.append(
            DateReference(
                token=Token(m.start(), m.end(), m.group()),
                date=f"01.01.{year}",
                date_end=f"31.12.{year}",
            )
        )

    return refs


# ---------------------------------------------------------------------------
# Organization recognition
# ---------------------------------------------------------------------------

# Аббревиатуры (длиннее идут первыми — порядок важен для regex alternation)
_ORG_ABBR = ("ФГУП", "МУП", "ГУП", "ПАО", "ОАО", "ЗАО", "ООО", "АО", "НП", "ИП")

# Полные формы -> аббревиатура (сортируем по убыванию длины, чтобы длинные паттерны
# проверялись раньше: ОТКРЫТОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО раньше чем АКЦИОНЕРНОЕ ОБЩЕСТВО)
_ORG_FULL: dict[str, str] = {
    "ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ УНИТАРНОЕ ПРЕДПРИЯТИЕ": "ФГУП",
    "МУНИЦИПАЛЬНОЕ УНИТАРНОЕ ПРЕДПРИЯТИЕ": "МУП",
    "ГОСУДАРСТВЕННОЕ УНИТАРНОЕ ПРЕДПРИЯТИЕ": "ГУП",
    "ПУБЛИЧНОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО": "ПАО",
    "ОТКРЫТОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО": "ОАО",
    "ЗАКРЫТОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО": "ЗАО",
    "ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ": "ООО",
    "АКЦИОНЕРНОЕ ОБЩЕСТВО": "АО",
    "НЕКОММЕРЧЕСКОЕ ПАРТНЕРСТВО": "НП",
    "ИНДИВИДУАЛЬНЫЙ ПРЕДПРИНИМАТЕЛЬ": "ИП",
}

_OPEN_QUOTE = r'\.?[\s"\]]+'  # опциональная точка, затем пробел/кавычка/]
_CLOSE_QUOTE = r'["\]]'  # " или ]
_NAME_INNER = r'(?:[^"]|"(?=\w))+'  # содержимое имени — всё кроме кавычек

_RE_ORG = re.compile(r"\b(" + "|".join(_ORG_ABBR) + r")" + _OPEN_QUOTE + r"(" + _NAME_INNER + r")" + _CLOSE_QUOTE)


# def _fuzzy_org_pattern(phrase: str) -> str:
#     """Генерирует regex, допускающий OCR-ошибку в одном слове фразы.

#     Для фраз из 1-2 слов — точное совпадение.
#     Для 3+ слов — каждое слово поочерёдно может быть любым кириллическим словом.
#     """
#     words = phrase.split()
#     if len(words) <= 2:
#         return re.escape(phrase)
#     variants = [
#         r"\s+".join(r"[А-ЯЁ]+" if j == i else re.escape(w) for j, w in enumerate(words)) for i in range(len(words))
#     ]
#     return "(?:" + "|".join(variants) + ")"


# _ORG_FULL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
#     (
#         re.compile(r"\b" + _fuzzy_org_pattern(key) + _OPEN_QUOTE + r"(" + _NAME_INNER + r")" + _CLOSE_QUOTE),
#         abbr,
#     )
#     for key, abbr in sorted(_ORG_FULL.items(), key=lambda x: len(x[0]), reverse=True)
# ]
_RE_ORG_FULL = re.compile(
    r"\b(" + "|".join(sorted(_ORG_FULL, key=len, reverse=True)) + r")"
    r'(?:\s+[А-ЯЁ]{1,3})?\s*\.?\s*"([^"]+)"'
)

_RE_RUSAL = re.compile(r'["\]](РУСАЛ' + _NAME_INNER + r')["\]]')
_RE_RUSAL_SPACE = re.compile(r"РУСАЛ([А-ЯЁ])")


def _clean_org_name(raw: str) -> str:
    """Убирает кавычки и лишние пробелы из имени организации после извлечения."""
    return re.sub(r'"+', " ", raw).strip()


def _find_orgs(text: str) -> list[OrganizationReference]:
    refs: list[OrganizationReference] = []

    for m in _RE_ORG.finditer(text):
        name = _RE_RUSAL_SPACE.sub(r"РУСАЛ \1", _clean_org_name(m.group(2)))
        refs.append(
            OrganizationReference(
                token=Token(m.start(), m.end(), m.group()),
                name=name,
                org_form=m.group(1),
            )
        )

    # for pattern, abbr in _ORG_FULL_PATTERNS:
    #     for m in pattern.finditer(text):
    #         name = _RE_RUSAL_SPACE.sub(r"РУСАЛ \1", m.group(1).strip())  # group(1) — имя
    #         refs.append(
    #             OrganizationReference(
    #                 token=Token(m.start(), m.end(), m.group()),
    #                 name=name,
    #                 org_form=abbr,
    #             )
    #         )

    for m in _RE_ORG_FULL.finditer(text):
        name = _RE_RUSAL_SPACE.sub(r"РУСАЛ \1", _clean_org_name(m.group(2)))
        org_form = _ORG_FULL[m.group(1)]
        refs.append(
            OrganizationReference(
                token=Token(m.start(), m.end(), m.group()),
                name=name,
                org_form=org_form,
            )
        )

    for m in _RE_RUSAL.finditer(text):
        name = _RE_RUSAL_SPACE.sub(r"РУСАЛ \1", _clean_org_name(m.group(1)))
        refs.append(
            OrganizationReference(
                token=Token(m.start(), m.end(), m.group()),
                name=name,
                org_form=None,
            )
        )

    return refs


# ---------------------------------------------------------------------------
# Digital value recognition
# ---------------------------------------------------------------------------

_RE_DIGITAL = re.compile(r"(?<!\S)\d+(?!\S)")


def _find_digitals(text: str) -> list[DigitalReference]:
    refs: list[DigitalReference] = []
    for m in _RE_DIGITAL.finditer(text):
        refs.append(
            DigitalReference(
                token=Token(m.start(), m.end(), m.group()),
                value=m.group(),
            )
        )
    return refs


# ---------------------------------------------------------------------------
# Currency recognition
# ---------------------------------------------------------------------------

_OCR_DIGIT_FIX = str.maketrans("ОоOolI|", "0000111")

# Числовой кандидат: захватывает весь блок целиком включая OCR-разделители
# Alt 1: минимум 2 цифры с чем угодно между ними (пробел, , ; .)
# Alt 2: одиночная цифра
_RE_CURRENCY_CANDIDATE = re.compile(r"\b\d[\d\s,;.]*\d\b|\b\d\b")

# DD.MM без года — валидные месяцы 01-12
_RE_DATE_LIKE = re.compile(r"^\d{1,2}[./](?:0[1-9]|1[0-2])$")

_CURRENCY_CONTEXT_WINDOW = 80

_RE_CURRENCY_MARKER_BEFORE = re.compile(
    r"\b(СУММ[А-ЯЁ]*|ИТОГО|ВСЕГО|ОСТАТК[А-ЯЁ]*|ЗАДОЛЖЕННОСТ[А-ЯЁ]*"
    r"|ОБОРОТ[А-ЯЁ]*|ДОЛГ[А-ЯЁ]*|БАЛАНС[А-ЯЁ]*|НАЧИСЛЕН[А-ЯЁ]*"
    r"|ОПЛАЧЕН[А-ЯЁ]*|ПЕРЕЧИСЛЕН[А-ЯЁ]*|ВЫПЛАЧЕН[А-ЯЁ]*|ПОГАШЕН[А-ЯЁ]*)\b"
)
_RE_CURRENCY_UNIT_AFTER = re.compile(r"^\s*(РУБЛЕЙ|РУБ[А-ЯЁ.]*|RUB)\b")


def _is_non_currency(text: str, start: int, end: int) -> bool:
    if start > 0 and text[start - 1] in "-/":
        return True
    if start > 0 and text[start - 1] == ".":
        if start >= 2 and (text[start - 2].isdigit() or text[start - 2].isalpha()):
            return True
    if end < len(text) and text[end] in "-/":
        return True
    return False


def _has_currency_marker(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - _CURRENCY_CONTEXT_WINDOW) : start]
    after = text[end : min(len(text), end + 20)]
    return bool(_RE_CURRENCY_MARKER_BEFORE.search(before)) or bool(_RE_CURRENCY_UNIT_AFTER.match(after))


def _has_words_in_context(text: str, start: int, end: int) -> bool:
    window = (
        text[max(0, start - _CURRENCY_CONTEXT_WINDOW) : start]
        + text[end : min(len(text), end + _CURRENCY_CONTEXT_WINDOW)]
    )
    return bool(re.search(r"[А-ЯЁ]{2,}", window))


def _parse_currency(raw: str) -> float:
    """Парсит денежное значение: все цифры, последние 2 — копейки."""
    digits = re.sub(r"\D", "", raw.translate(_OCR_DIGIT_FIX))
    if not digits:
        return 0.0
    if len(digits) < 3:
        digits = digits.zfill(3)
    rubles = digits[:-2].lstrip("0") or "0"
    kopecks = digits[-2:]
    return float(f"{rubles}.{kopecks}")


def _find_currencies(text: str) -> list[CurrencyReference]:
    result: list[CurrencyReference] = []
    covered: list[tuple[int, int]] = []

    for m in _RE_CURRENCY_CANDIDATE.finditer(text):
        start, end = m.start(), m.end()
        if any(s <= start < e or s < end <= e for s, e in covered):
            continue
        if re.fullmatch(r"\d+", m.group()):
            continue  # чистое целое -> DigitalReference, не Currency
        if _is_non_currency(text, start, end):
            continue
        if _RE_DATE_LIKE.match(m.group()):
            continue
        # Числовой контекст — только цифры вокруг -> всегда деньги
        if not _has_words_in_context(text, start, end):
            result.append(CurrencyReference(token=Token(start, end, m.group()), value=_parse_currency(m.group())))
            covered.append((start, end))
            continue
        # Текстовый контекст — нужен маркер
        if _has_currency_marker(text, start, end):
            result.append(CurrencyReference(token=Token(start, end, m.group()), value=_parse_currency(m.group())))
            covered.append((start, end))

    return result


# ---------------------------------------------------------------------------
# Digital reference recognition
# ---------------------------------------------------------------------------

_RE_DIGITAL = re.compile(r"(?<!\S)\d+(?!\S)")


def _find_digital(text: str) -> list[DigitalReference]:
    return [
        DigitalReference(token=Token(m.start(), m.end(), m.group()), value=float(m.group()))
        for m in _RE_DIGITAL.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Overlap resolution — при перекрытии побеждает более длинный токен
# ---------------------------------------------------------------------------


def _remove_overlaps(refs: list[AnyReference]) -> list[AnyReference]:
    result: list[AnyReference] = []
    for ref in refs:
        if result and ref.token.start < result[-1].token.end:
            if (ref.token.end - ref.token.start) > (result[-1].token.end - result[-1].token.start):
                result[-1] = ref
        else:
            result.append(ref)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[AnyReference]:
    """Находит все токены в нормализованном тексте (ВЕРХНИЙ регистр).

    Порядок приоритета при перекрытии: побеждает более длинный токен.
    """
    refs: list[AnyReference] = _find_dates(text) + _find_orgs(text) + _find_currencies(text) + _find_digital(text)
    refs.sort(key=lambda r: r.token.start)
    return _remove_overlaps(refs)
