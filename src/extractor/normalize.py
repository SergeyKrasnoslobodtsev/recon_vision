import re
import unicodedata


def _build_extended_cyrillic_map() -> dict:
    russian = set(range(0x0410, 0x0450)) | {0x0401, 0x0451}  # А-Яа-я + Ёё
    result = {}
    for cp in range(0x0400, 0x0500):
        if cp in russian:
            continue
        decomposed = unicodedata.normalize("NFD", chr(cp))
        base = decomposed[0]
        if ord(base) in russian:
            result[cp] = base  # Ӧ -> О, Ӓ -> А, Ӗ -> Е ...
    return result


_NORM_MAP = str.maketrans(
    {
        # Двойные кавычки -> стандартная “
        0x00AB: '"',  # «
        0x00BB: '"',  # »
        0x201C: '"',  # “
        0x201D: '"',  # ”
        0x201E: '"',  # „
        0x003C: '"',  # <
        0x003E: '"',  # >
        # Одиночные кавычки и OCR-мусор
        0x0027: " ",  # ‘
        0x0060: " ",  # `
        0x00B7: " ",  # ·
        0x2018: " ",  # ‘
        0x2019: " ",  # ‘
        0x201A: " ",  # ‚
        0x2039: " ",  # ‹
        0x203A: " ",  # ›
        0x2021: " ",  # ‡
        0x2020: " ",  # †
        0x0404: "Е",  # Є  Ukrainian
        0x0454: "Е",  # є
        0x0490: "Г",  # Ґ  Ukrainian
        0x0491: "Г",  # ґ
        0x040E: "У",  # Ў  Belarusian
        0x045E: "У",  # ў
        0x040F: "Т",  # Џ  -> Т (Тел.)
        0x040C: "К",  # Ќ
        0x0403: "Г",  # Ѓ
        0x0405: "З",  # Ѕ
        0x0408: "Й",  # Ј
        0x0498: "З",  # ҙ
        0x04AE: "У",  # Ү
        0x04AF: "У",  # ү
        0x04B0: "У",  # Ұ
        0x04B1: "У",  # ұ
        0x0492: "Г",  # Ғ
        0x0493: "Г",  # ғ
        0x049A: "К",  # Қ
        0x049B: "К",  # қ
        0x04A2: "Н",  # Ң
        0x04A3: "Н",  # ң
        0x04E8: "О",  # Ө
        0x04E7: "О",  # ө
        0x04E9: "О",  # ө
        0x04E4: "И",  # Ӥ
        0x04E5: "И",  # ӥ
        0x04E2: "И",  # Ӣ
        0x04E3: "И",  # ӣ
        0x0499: "З",  # ҙ
        0x0462: "Е",  # Ѣ  ять
        0x0472: "Ф",  # Ѳ  фита
        0x0474: "И",  # Ѵ  ижица
        0x005B: None,  # [
        0x005D: None,  # ]
        0x007C: None,  # |
        ord("Ё"): "Е",
    }
)


_RULES_SPACE: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\s+", re.UNICODE), " ", "collapse_spaces"),  # "слово   слово" -> "слово слово"
    (re.compile(r"\s+([,.:;])"), r"\1", "space_before_punct"),  # "слово , слово" -> "слово, слово"
    (re.compile(r"([,;:])(?=[^\s\d])"), r"\1 ", "space_after_punct"),  # "слово,слово" -> "слово, слово"
    (
        re.compile(r"(?<=\w\w)\.(?=[А-ЯЁA-Za-z])"),
        ". ",
        "space_after_dot",
    ),  # 0.00 -> 0.00, но "слово.слово" -> "слово. слово"
]

_RULES_QUOTES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r'"{2,}'), '"', "collapse_quotes"),  # """" -> "
    (re.compile(r'(\S)"(?=\w)'), r'\1 "', "space_before_quote"),  # слово" -> слово "
    (re.compile(r"\.{2,}"), ".", "collapse_dots"),  # ....... -> .
    (re.compile(r",{2,}"), ",", "collapse_commas"),  # ,,,,, -> ,
    (re.compile(r"(?<![А-ЯЁA-Z0-9.])[.:]"), "", "leading_punct"),  # .0.00 -> 0.00 :0.00 -> 0.00
    (re.compile(r"([,;:.!])\s*\1+"), r"\1", "collapse_dup_punct"),
]


def _apply_rules(text: str, rules: list[tuple[re.Pattern, str, str]]) -> str:
    for pattern, repl, _ in rules:
        text = pattern.sub(repl, text)
    return text


# Символы, которые OCR путает с цифрами (после upper())
_LAT_TO_DIGIT = str.maketrans("OI", "01")

# Числовой контекст: последовательность цифро-подобных символов через . или ,
# \b не используем — он не работает с кириллицей, поэтому (?<!\w)
RE_NUMERIC = re.compile(r"(?<![А-ЯЁA-Z])(?:[0-9OI]+[.,])+[0-9OI]+(?![А-ЯЁA-Z])")


def _fix_numeric(m: re.Match) -> str:
    return m.group(0).translate(_LAT_TO_DIGIT)


_LAT_TO_CYR = str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ")

RE_SMART_LAT = re.compile(r"(?<=[А-ЯЁ])[A-Z]|[A-Z](?=[А-ЯЁ])")

# 1. Сначала защищаем числовой контекст: цифра-пробел-ООО-точка/запятая-цифра
RE_OOO_THOUSANDS = re.compile(r"(?<=\d )[О0]{3}(?=[.,]\d)")  # 20 ООО.25 -> 20 000.25

# 2. Потом заменяем ООО как форму собственности
RE_OOO_COMPANY = re.compile(r'[О0]{3}(?=[\s.]*[«"""]|\s+[А-ЯЁ])')  # 000 "Рога" -> ООО "Рога"


RE_DATE = re.compile(r"(\d{1,2})[^\s\d](\d{1,2})[^\s\d](\d{2}(?:\d{2})?)([А-ЯЁA-Z]\.?)?")


def _fix_date(m: re.Match) -> str:
    result = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    if m.group(4):
        result += " Г."
    return result


def transform_text(text: str) -> str:
    if not text:
        return ""

    maps = [
        _NORM_MAP,
        _build_extended_cyrillic_map(),
    ]
    for m in maps:
        text = text.upper().translate(m)
    text = RE_DATE.sub(_fix_date, text)  # сначала даты
    text = RE_OOO_THOUSANDS.sub("000", text)  # потом числовой контекст
    text = RE_OOO_COMPANY.sub("ООО", text)  # потом ООО как форма собственности
    text = RE_NUMERIC.sub(_fix_numeric, text)  # потом остальные числа
    text = RE_SMART_LAT.sub(lambda m: m.group(0).translate(_LAT_TO_CYR), text)
    text = _apply_rules(text, _RULES_SPACE)
    text = _apply_rules(text, _RULES_QUOTES)
    return text.strip()


_OCR_FIX = str.maketrans("ОоOolI|", "0000111")


_RE_DECIMAL_TAIL = re.compile(r"[.,;](\d{1,2})$")


def transform_currency(text: str | None) -> float:
    """
    Парсит денежное значение из OCR-строки в float.

    Если в конце строки есть разделитель (, . ;) перед 1-2 цифрами — это граница
    копеек (при одной цифре дополняется нулём справа: ",4" -> 40 коп.), а всё до
    него — рубли. Без такого разделителя — старое правило: последние 2 цифры
    всей строки — копейки, остальные — рубли.

    Примеры:
      "1 649 669,47" -> 1649669.47
      "47 761,70"    -> 47761.70
      "1 636 276,4"  -> 1636276.40  (одна цифра после запятой -> копейки *10)
      "1 649 669,4О" -> 1649669.40  (кириллическая О -> 0)
      "100"          -> 1.00        (без разделителя — старое правило)
      "-"            -> 0.0
      None           -> 0.0
    """
    if not text:
        return 0.0

    fixed = text.translate(_OCR_FIX)

    tail = _RE_DECIMAL_TAIL.search(fixed)
    if tail:
        rubles_digits = re.sub(r"\D", "", fixed[: tail.start()])
        if not rubles_digits:
            if text.strip() not in ("", "-", "—", "–"):
                raise ValueError(f"parse_currency: не удалось извлечь цифры из {text!r}")
            return 0.0
        kopecks = tail.group(1).ljust(2, "0")
        return int(rubles_digits) + int(kopecks) / 100

    digits = re.sub(r"\D", "", fixed)

    if not digits:
        if text.strip() not in ("", "-", "—", "–"):
            raise ValueError(f"parse_currency: не удалось извлечь цифры из {text!r}")
        return 0.0

    digits = digits.zfill(3)
    return int(digits[:-2]) + int(digits[-2:]) / 100
