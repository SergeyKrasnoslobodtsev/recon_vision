"""Извлекает организации и их роли из канонического документа."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from extractor.process import extract
from extractor.tokenize import OrganizationReference
from vision_core.entities.document import Document
from vision_core.entities.table import Table

from .exceptions import CompanyNotFoundError, RoleAssignmentError, RoleAssignmentNotFoundError

# ---------------------------------------------------------------------------
# Сущность Company
# ---------------------------------------------------------------------------


class _Role(Enum):
    BUYER = 0
    SELLER = 1
    UNKNOWN = -1


@dataclass(frozen=True)
class Company:
    """Организация, извлечённая из акта сверки.

    Attributes:
        name: Название без организационной формы.
        org_form: Аббревиатура формы (ООО, АО, ...) или пустая строка.
        role: Роль в акте — 'seller', 'buyer' или 'unknown'.
    """

    name: str
    org_form: str
    role: str

    @property
    def display_name(self) -> str:
        return f"{self.name}, {self.org_form}" if self.org_form else self.name


# ---------------------------------------------------------------------------
# Извлечение текста из Document
# ---------------------------------------------------------------------------


def _cell_covering(table: Table, row: int, col: int):
    rows = table.get_rows()
    if row >= len(rows):
        return None
    for cell in rows[row]:
        if cell.col <= col < cell.col + cell.colspan:
            return cell
    return None


def _build_summary_text(document: Document) -> str:
    parts: list[str] = []
    for page in document.pages:
        text = " ".join(p.text for p in page.paragraphs if p.text.strip())
        if text:
            parts.append(text)
    result = " ".join(parts)
    logger.debug(f"summary_text: {len(result)} символов из {len(document.pages)} страниц")
    return result


def _build_summary_cell_texts(document: Document) -> list[str]:
    result: list[str] = []
    for page in document.pages:
        for table in page.tables:
            if table.continuation_of is not None:
                continue
            num_row = table.get_dc_header_row()
            if num_row == -1 or not table.dc_cols:
                continue
            seen: set[int] = set()
            for col in sorted(table.dc_cols):
                for j in range(num_row):
                    cell = _cell_covering(table, j, col)
                    if cell is None or id(cell) in seen:
                        continue
                    seen.add(id(cell))
                    if cell.value and cell.value.strip():
                        normalized = re.sub(r"\s+", " ", cell.value.strip())
                        result.append(normalized)
    logger.debug(f"summary_cell_texts: {result}")
    return result


# ---------------------------------------------------------------------------
# Извлечение организаций через токенизатор
# ---------------------------------------------------------------------------


def _org_full_name(ref: OrganizationReference) -> str:
    return f"{ref.name}, {ref.org_form}" if ref.org_form else ref.name


def _org_token(full_name: str) -> str:
    return full_name.split(",", 1)[0].strip()


def _org_counts(raw: str) -> dict[str, tuple[str, int]]:
    """token -> (полное имя с формой, число упоминаний токена в тексте)."""
    result = extract(raw)
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    for ref in result.tokens:
        if isinstance(ref, OrganizationReference):
            full = _org_full_name(ref)
            token = _org_token(full)
            counts[token] = counts.get(token, 0) + 1
            names.setdefault(token, full)
    return {token: (names[token], counts[token]) for token in names}


def _merge_org_counts(*sources: dict[str, tuple[str, int]]) -> dict[str, tuple[str, int]]:
    merged: dict[str, tuple[str, int]] = {}
    for src in sources:
        for token, (full, count) in src.items():
            prev_full, prev_count = merged.get(token, (full, 0))
            merged[token] = (prev_full, prev_count + count)
    return merged


def _dominant_orgs(counts: dict[str, tuple[str, int]]) -> list[str]:
    """Уникальные организации, отсортированные по числу упоминаний (убывание).

    Если токен одной организации — подстрока токена другой (например, мусорный
    токен из-за регэкспа, склеившего два разных `АО "..."` через текст подписи
    между ними), побеждает не более длинный, а более часто встречающийся —
    частые упоминания сильнее говорят о реальной компании, чем длина совпадения.
    """
    tokens = list(counts.keys())
    kept = [
        (full, count)
        for token, (full, count) in counts.items()
        if not any(token != other and token in other and counts[other][1] >= count for other in tokens)
    ]
    kept.sort(key=lambda x: x[1], reverse=True)
    return [full for full, _ in kept]


# ---------------------------------------------------------------------------
# Назначение ролей
# ---------------------------------------------------------------------------

_ROLE_GLOSSARY: dict[str, str] = {
    r"\bОТ ПОКУПАТЕЛЯ\b": "BUYER_SOURCE",
    r"\bОТ ПРОДАВЦА\b": "SELLER_SOURCE",
    r"\bМЕЖДУ\b": "PARTICIPATION_SCOPE",
}


def _find_events(text: str) -> list[dict]:
    found = []
    for pattern, role in _ROLE_GLOSSARY.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            found.append({"keyword": m.group(), "role": role, "start_index": m.start(), "end_index": m.end()})
    return sorted(found, key=lambda x: x["start_index"])


def _find_orgs_in_span(text: str, left: int, right: int, orgs: list[str]) -> list[str]:
    span = text[left:right]
    return [o for o in orgs if _org_token(o) and _org_token(o) in span]


def _find_working_pair(text: str, orgs: list[str], events: list[dict]) -> list[str]:
    anchor = next((e for e in events if e["role"] == "PARTICIPATION_SCOPE"), None)
    if anchor:
        window = text[anchor["end_index"] : anchor["end_index"] + 400]
        hits = sorted([(window.find(_org_token(o)), o) for o in orgs if _org_token(o) in window])
        pair = [o for _, o in hits[:2]]
        if len(pair) == 2:
            return pair
    return orgs[:2]


def _apply_symmetry(roles: dict[str, _Role]) -> None:
    buyers = [o for o, r in roles.items() if r == _Role.BUYER]
    sellers = [o for o, r in roles.items() if r == _Role.SELLER]
    unknowns = [o for o, r in roles.items() if r == _Role.UNKNOWN]
    if buyers and unknowns and not sellers:
        for o in unknowns:
            roles[o] = _Role.SELLER
    elif sellers and unknowns and not buyers:
        for o in unknowns:
            roles[o] = _Role.BUYER


def _assign_roles(text: str, orgs: list[str]) -> dict[str, _Role]:
    events = _find_events(text)
    logger.debug(f"события: {[e['keyword'] for e in events]}")
    pair = _find_working_pair(text, orgs, events)
    logger.debug(f"рабочая пара: {pair}")
    roles: dict[str, _Role] = {o: _Role.UNKNOWN for o in pair}

    rusals = [o for o in pair if "РУСАЛ" in _org_token(o)]

    if len(rusals) == 1:
        roles[rusals[0]] = _Role.BUYER
        logger.debug(f"РУСАЛ-правило: {rusals[0]} -> BUYER")
    elif len(rusals) == 2:
        roles[rusals[0]] = _Role.SELLER
        roles[rusals[1]] = _Role.BUYER
        logger.debug(f"двойной РУСАЛ: {rusals[0]} -> SELLER, {rusals[1]} -> BUYER")

    _apply_symmetry(roles)

    for idx, event in enumerate(events):
        if event["role"] not in {"BUYER_SOURCE", "SELLER_SOURCE"}:
            continue
        next_start = events[idx + 1]["start_index"] if idx + 1 < len(events) else len(text)
        target = _Role.BUYER if event["role"] == "BUYER_SOURCE" else _Role.SELLER
        for o in _find_orgs_in_span(text, event["end_index"], next_start, pair):
            if roles[o] == _Role.UNKNOWN:
                roles[o] = target
                logger.debug(f"якорь '{event['keyword']}': {o} -> {target.name}")
    _apply_symmetry(roles)

    unknowns = [o for o, r in roles.items() if r == _Role.UNKNOWN]
    if unknowns:
        roles[unknowns[0]] = _Role.SELLER
        logger.debug(f"позиционный фоллбек: {unknowns[0]} -> SELLER")
        for o in unknowns[1:]:
            roles[o] = _Role.BUYER
            logger.debug(f"позиционный фоллбек: {o} -> BUYER")

    return roles


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def _parse_company(full_name: str, role: _Role) -> Company:
    if "," in full_name:
        name, form = full_name.rsplit(",", 1)
        return Company(name=name.strip(), org_form=form.strip(), role=role.name.lower())
    return Company(name=full_name.strip(), org_form="", role=role.name.lower())


def extract_companies(document: Document) -> list[Company]:
    """Извлекает компании с ролями seller/buyer из документа."""
    summary_text = _build_summary_text(document)
    cell_texts = _build_summary_cell_texts(document)

    text_counts = _org_counts(summary_text)
    cell_counts = _merge_org_counts(*(_org_counts(t) for t in cell_texts))

    orgs_from_cells = _dominant_orgs(cell_counts)
    if len(orgs_from_cells) >= 2:
        merged_counts = cell_counts
    elif orgs_from_cells:
        merged_counts = _merge_org_counts(cell_counts, text_counts)
    else:
        merged_counts = text_counts

    candidates = _dominant_orgs(merged_counts)
    logger.info(f"кандидаты: {candidates}")

    if not candidates:
        logger.error("не найдено организаций")
        raise CompanyNotFoundError()

    normalized = extract(summary_text).text
    roles = _assign_roles(normalized, candidates)

    if all(r == _Role.UNKNOWN for r in roles.values()):
        logger.error("не удалось определить роли организаций")
        raise RoleAssignmentNotFoundError()

    for org, role in roles.items():
        logger.info(f"определённая роль: {org} -> {role.name}")
        if role == _Role.UNKNOWN:
            logger.error(f"Не удалось определить роль для организации '{org}'")
            raise RoleAssignmentError(org=org)

    companies = [_parse_company(org, role) for org, role in roles.items()]
    sellers = [c for c in companies if c.role == "seller"]
    buyers = [c for c in companies if c.role == "buyer"]
    logger.info(
        f"seller={sellers[0].display_name if sellers else '—'}  buyer={buyers[0].display_name if buyers else '—'}"
    )
    return companies
