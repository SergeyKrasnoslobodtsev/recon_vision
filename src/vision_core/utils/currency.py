import re

_OCR_FIX = str.maketrans("ОоOolI|", "0000111")


def parse_currency(value: str | None) -> float:
    """
    Парсит денежное значение из OCR-строки в float.
    Последние 2 цифры — копейки, остальные — рубли.

    Примеры:
      "1 649 669,47" -> 1649669.47
      "47 761,70"    -> 47761.70
      "1 649 669,4О" -> 1649669.40  (кириллическая О -> 0)
      "-"            -> 0.0
      None           -> 0.0
    """
    if not value:
        return 0.0

    fixed = value.translate(_OCR_FIX)
    digits = re.sub(r"\D", "", fixed)

    if not digits:
        if value.strip() not in ("", "-", "—", "–"):
            raise ValueError(f"parse_currency: не удалось извлечь цифры из {value!r}")
        return 0.0

    digits = digits.zfill(3)
    return int(digits[:-2]) + int(digits[-2:]) / 100
