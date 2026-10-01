"""Deterministic, conservative claim parsing for the news evidence pipeline.

This module deliberately does not use a language model or external entity lookup.
It records only details that can be read from the submitted text, leaving fields
unset or ``UNKNOWN`` when the text does not establish them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

from backend.schemas import Claim, ClaimContext, ClaimNumber, ClaimType


_NUMBER_PATTERN = r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"

_UNIT_ALIASES = {
    "g": "gram",
    "gram": "gram",
    "grams": "gram",
    "kg": "kilogram",
    "kilogram": "kilogram",
    "kilograms": "kilogram",
    "mg": "milligram",
    "milligram": "milligram",
    "milligrams": "milligram",
    "oz": "ounce",
    "ounce": "ounce",
    "ounces": "ounce",
    "lb": "pound",
    "lbs": "pound",
    "pound": "pound",
    "pounds": "pound",
    "ml": "milliliter",
    "milliliter": "milliliter",
    "milliliters": "milliliter",
    "l": "liter",
    "litre": "liter",
    "litres": "liter",
    "liter": "liter",
    "liters": "liter",
    "km": "kilometer",
    "kilometer": "kilometer",
    "kilometers": "kilometer",
    "m": "meter",
    "meter": "meter",
    "meters": "meter",
    "°c": "celsius",
    "celsius": "celsius",
    "°f": "fahrenheit",
    "fahrenheit": "fahrenheit",
    "%": "percent",
    "percent": "percent",
    "percentage": "percent",
    "million": "million",
    "billion": "billion",
    "thousand": "thousand",
    "people": "people",
    "cases": "cases",
    "votes": "votes",
    "units": "units",
    "doses": "doses",
    "employees": "employees",
    "students": "students",
}

_MEASUREMENT_UNIT_PATTERN = (
    r"(?:°\s?[CF]|grams?|g|kilograms?|kg|milligrams?|mg|ounces?|oz|pounds?|lbs?|"
    r"milliliters?|ml|lit(?:er|re)s?|l|kilometers?|km|meters?|m|"
    r"celsius|fahrenheit|percent(?:age)?|million|billion|thousand|"
    r"people|cases|votes|units|doses|employees|students)"
)

_CURRENCY_VALUE_RE = re.compile(
    rf"(?P<currency>₹|\$|€|£|\b(?:INR|USD|EUR|GBP|Rs\.?)\b)\s*"
    rf"(?P<value>{_NUMBER_PATTERN})"
    rf"(?:\s*(?:/|per)\s*(?P<unit>{_MEASUREMENT_UNIT_PATTERN}))?",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    rf"(?P<value>{_NUMBER_PATTERN})\s*(?P<unit>%|percent(?:age)?)",
    re.IGNORECASE,
)
_MEASUREMENT_RE = re.compile(
    rf"(?P<value>{_NUMBER_PATTERN})\s*(?P<unit>{_MEASUREMENT_UNIT_PATTERN})",
    re.IGNORECASE,
)
_PRICE_NUMBER_RE = re.compile(
    rf"(?P<value>{_NUMBER_PATTERN})"
    rf"(?:\s*(?:/|per)\s*(?P<unit>{_MEASUREMENT_UNIT_PATTERN}))?",
    re.IGNORECASE,
)
_BARE_VALUE_RE = re.compile(rf"(?P<value>{_NUMBER_PATTERN})")
_PURITY_RE = re.compile(r"\b(?P<purity>\d{1,2})\s*(?:K|karat|carat)\b", re.IGNORECASE)

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_PATTERN = "|".join(_MONTHS)
_ISO_DATE_RE = re.compile(r"\b(?P<year>19\d{2}|20\d{2})[-/](?P<month>0?[1-9]|1[0-2])[-/](?P<day>0?[1-9]|[12]\d|3[01])\b")
_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\s+(?P<day>[12]?\d|3[01])(?:,\s*|\s+)(?P<year>19\d{{2}}|20\d{{2}})\b",
    re.IGNORECASE,
)
_DAY_MONTH_YEAR_RE = re.compile(
    rf"\b(?P<day>[12]?\d|3[01])\s+(?P<month>{_MONTH_PATTERN})\s+(?P<year>19\d{{2}}|20\d{{2}})\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19\d{2}|20\d{2})\b")

_LOCATION_PREPOSITION_RE = re.compile(
    r"\b(?:in|at|near|within|across)\s+(?:the\s+)?"
    r"(?P<location>[A-Z][A-Za-z'’\-]*(?:\s+[A-Z][A-Za-z'’\-]*){0,3})"
)
_LOCATION_POSSESSIVE_RE = re.compile(r"\b(?P<location>[A-Z][A-Za-z'’\-]*(?:\s+[A-Z][A-Za-z'’\-]*){0,3})(?:'s|’s)\b")
_LOCATION_BASED_RE = re.compile(r"\b(?P<location>[A-Z][A-Za-z'’\-]*(?:\s+[A-Z][A-Za-z'’\-]*){0,3})-based\b")
_NON_LOCATION_TOKENS = frozenset({*{month.title() for month in _MONTHS}, "Today", "Yesterday", "Current", "Latest"})

_FINANCIAL_PRICE_RE = re.compile(
    r"\b(?:price|market\s+rate|exchange\s+rate|trading\s+at|spot\s+price|costs?|priced\s+at)\b",
    re.IGNORECASE,
)
_PER_UNIT_PRICE_RE = re.compile(r"(?:₹|\$|€|£|\b(?:INR|USD|EUR|GBP)\b)\s*" + _NUMBER_PATTERN + r"\s*(?:/|per)\s*", re.IGNORECASE)
_LEGAL_RE = re.compile(r"\b(?:court|judge|ruling|verdict|lawsuit|convicted|sentenced|appeal|petition|legal)\b", re.IGNORECASE)
_SCIENTIFIC_RE = re.compile(r"\b(?:study|research|scientists?|scientific|clinical\s+trial|peer[-\s]?reviewed|journal|researchers?|experiment)\b", re.IGNORECASE)
_SPORTS_RE = re.compile(r"\b(?:match|game|tournament|championship|league|final|semi[-\s]?final|goal|score)\b|\b\d+\s*[-–]\s*\d+\b", re.IGNORECASE)
_SPORTS_RESULT_RE = re.compile(r"\b(?:won|defeated|beat|drew)\b", re.IGNORECASE)
_ANNOUNCEMENT_RE = re.compile(r"\b(?:announced|announce|announcing|declared|issued|released|confirmed|approved)\b", re.IGNORECASE)
_PRODUCT_ACTION_RE = re.compile(r"\b(?:launched|launches|launching|unveiled|unveils|introduced|introduces|released|releases)\b", re.IGNORECASE)
_PRODUCT_CUE_RE = re.compile(
    r"\b(?:product|phone|smartphone|device|app|application|software|service|model|vehicle|car|tablet|laptop|platform|version|feature)\b",
    re.IGNORECASE,
)
_GOVERNMENT_RE = re.compile(
    r"\b(?:government|ministry|minister|parliament|cabinet|president|prime\s+minister|state\s+government|city\s+council|municipality|department)\b",
    re.IGNORECASE,
)
_COMPANY_RE = re.compile(r"\b(?:company|corporation|corp\.?|inc\.?|ltd\.?|limited|firm|business|brand)\b", re.IGNORECASE)
_ORGANIZATION_RE = re.compile(
    r"\b(?:organization|organisation|association|foundation|university|college|committee|agency|institution|union|nonprofit|non-profit)\b",
    re.IGNORECASE,
)
_PERSON_STATEMENT_RE = re.compile(
    r"^(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?)?\s*[A-Z][A-Za-z'’\-]+\s+[A-Z][A-Za-z'’\-]+\s+"
    r"(?:said|stated|claimed|told|wrote|announced|confirmed)\b"
)
_EVENT_RE = re.compile(
    r"\b(?:happened|occurred|took\s+place|began|started|ended|opened|closed|collapsed|exploded|struck|hit)\b",
    re.IGNORECASE,
)
_FACTUAL_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|will|would|can|could|does|did|contains|"
    r"produces|increased|decreased|rose|fell|grew|declined|costs?)\b",
    re.IGNORECASE,
)
_OTHER_ACTION_RE = re.compile(r"\b(?:reported|expects?|plans?|built|made|found|moved|died|arrived|left|won)\b", re.IGNORECASE)
_SUBJECT_VERB_RE = re.compile(
    r"^(?P<subject>(?:the\s+)?[A-Za-z][A-Za-z0-9&.'’\-]*(?:\s+[A-Za-z][A-Za-z0-9&.'’\-]*){0,5}?)\s+"
    r"(?:has\s+|have\s+|had\s+|will\s+|would\s+|is\s+|are\s+|was\s+|were\s+)?"
    r"(?:announced|announce|launched|launches|unveiled|introduced|released|confirmed|issued|"
    r"approved|said|stated|claimed|told|ruled|won|defeated|beat|is|are|was|were|has|have|"
    r"costs?|priced|contains|produces|increased|decreased|rose|fell|grew|declined|happened|"
    r"occurred|began|started|ended|opened|closed|collapsed|exploded|struck|hit|found)\b",
    re.IGNORECASE,
)
# _LEADING_TIME_RE = re.compile(
#     rf"^(?:(?:today|yesterday|currently|latest)\s*,\s*|(?:on\s+)?(?:{_MONTH_PATTERN})\s+\d{{1,2}},?\s+(?:19\d{{2}}|20\d{{2}})\s*,\s*|(?:19\d{{2}}|20\d{{2}})[-/]\d{{1,2}}[-/]\d{{1,2}\s*,\s*)",
#     re.IGNORECASE,
# )


_LEADING_TIME_RE = re.compile(
    rf"^(?:(?:today|yesterday|currently|latest)\s*,\s*|(?:on\s+)?(?:{_MONTH_PATTERN})\s+\d{{1,2}},?\s+(?:19\d{{2}}|20\d{{2}})\s*,\s*|(?:19\d{{2}}|20\d{{2}})[-/]\d{{1,2}}[-/]\d{{1,2}}\s*,\s*)",
    re.IGNORECASE,
)



_COMMODITIES = ("gold", "silver", "crude oil", "oil", "bitcoin", "ethereum", "platinum", "copper", "natural gas")


def _normalise_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _normalise_unit(raw_unit: str | None) -> str | None:
    if not raw_unit:
        return None
    compact = re.sub(r"\s+", "", raw_unit).casefold()
    return _UNIT_ALIASES.get(compact)


def _number_from_text(raw_value: str) -> float:
    return float(raw_value.replace(",", ""))


def _spans_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _is_covered(span: tuple[int, int], used_spans: list[tuple[int, int]]) -> bool:
    return any(_spans_overlap(span, used) for used in used_spans)


def _currency_code(raw_currency: str) -> str | None:
    return {
        "₹": "INR",
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "inr": "INR",
        "usd": "USD",
        "eur": "EUR",
        "gbp": "GBP",
        "rs": "INR",
        "rs.": "INR",
    }.get(raw_currency.strip().casefold())


def _number_attribute_for_unit(unit: str | None) -> str:
    if unit == "percent":
        return "percentage"
    if unit in {"celsius", "fahrenheit"}:
        return "temperature"
    if unit in {"people", "cases", "votes", "units", "doses", "employees", "students"}:
        return "quantity"
    return "quantity"


def _extract_numbers(text: str) -> tuple[list[ClaimNumber], dict[str, str]]:
    """Extract explicit, unambiguous number/unit pairs from a claim."""
    entries: list[tuple[int, ClaimNumber]] = []
    used_spans: list[tuple[int, int]] = []
    attributes: dict[str, str] = {}

    def add(
        match: re.Match[str],
        *,
        attribute: str,
        unit: str | None = None,
        currency: str | None = None,
    ) -> None:
        span = match.span()
        if _is_covered(span, used_spans):
            return
        try:
            value = _number_from_text(match.group("value"))
        except (TypeError, ValueError):
            return
        canonical_unit = unit if unit is not None else _normalise_unit(match.groupdict().get("unit"))
        entries.append((span[0], ClaimNumber(raw_text=match.group(0).strip(), value=value, unit=canonical_unit, attribute=attribute)))
        used_spans.append(span)
        if currency and "currency" not in attributes:
            attributes["currency"] = currency

    for match in _CURRENCY_VALUE_RE.finditer(text):
        add(
            match,
            attribute="price",
            currency=_currency_code(match.group("currency")),
        )

    for match in _PERCENT_RE.finditer(text):
        add(match, attribute="percentage")

    for match in _PRICE_NUMBER_RE.finditer(text):
        span = match.span()
        if _is_covered(span, used_spans):
            continue
        before = text[max(0, span[0] - 60):span[0]].casefold()
        is_price_context = bool(match.groupdict().get("unit")) or bool(
            re.search(
                r"\b(?:price|cost|costs|costing|priced|worth|rate)\b[^.!?]{0,40}"
                r"(?:\b(?:is|was|at|of)\b\s*|:\s*|\b(?:costs?|worth)\s*)$",
                before,
            )
        )
        if is_price_context:
            add(match, attribute="price")

    for match in _MEASUREMENT_RE.finditer(text):
        unit = _normalise_unit(match.group("unit"))
        add(match, attribute=_number_attribute_for_unit(unit), unit=unit)

    for match in _BARE_VALUE_RE.finditer(text):
        span = match.span()
        if _is_covered(span, used_spans):
            continue
        before = text[max(0, span[0] - 28):span[0]].casefold()
        if re.search(r"\b(?:is|are|was|were|has|have|had|reached|totalled|totaled)\s*$", before):
            add(match, attribute="value", unit=None)

    purity_match = _PURITY_RE.search(text)
    if purity_match:
        attributes["purity"] = f"{purity_match.group('purity')}K"

    entries.sort(key=lambda entry: entry[0])
    return [entry for _, entry in entries], attributes


def _validated_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _extract_time_reference(text: str) -> str | None:
    for pattern in (_ISO_DATE_RE, _MONTH_DAY_YEAR_RE, _DAY_MONTH_YEAR_RE):
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groupdict()
        month = groups["month"]
        month_number = _MONTHS[month.casefold()] if isinstance(month, str) and not month.isdigit() else int(month)
        parsed = _validated_date(int(groups["year"]), month_number, int(groups["day"]))
        if parsed:
            return parsed

    lower_text = text.casefold()
    for phrase in ("today", "yesterday", "currently", "latest", "this week", "last week", "this month", "last month"):
        if re.search(rf"\b{re.escape(phrase)}\b", lower_text):
            return phrase
    if re.search(r"\b(?:historical|historically|formerly)\b|\bin\s+the\s+past\b", lower_text):
        return "historical"
    year_match = _YEAR_RE.search(text)
    return year_match.group(0) if year_match else None


def _is_plausible_location(candidate: str) -> bool:
    words = candidate.split()
    if not words or candidate in _NON_LOCATION_TOKENS:
        return False
    return not all(word in _NON_LOCATION_TOKENS for word in words)


def _extract_location(text: str) -> str | None:
    for pattern in (_LOCATION_PREPOSITION_RE, _LOCATION_POSSESSIVE_RE, _LOCATION_BASED_RE):
        match = pattern.search(text)
        if not match:
            continue
        candidate = match.group("location").strip(" ,;:.")
        if _is_plausible_location(candidate):
            return candidate
    return None


def _has_capitalized_product_after_action(text: str) -> bool:
    action = _PRODUCT_ACTION_RE.search(text)
    if not action:
        return False
    suffix = text[action.end():]
    return bool(re.match(r"\s+(?:the\s+|a\s+|an\s+)?[A-Z][A-Za-z0-9'’\-]*(?:\s+[A-Z0-9][A-Za-z0-9'’\-]*){0,3}\b", suffix))


def _classify_claim(text: str) -> ClaimType:
    if not text or text.endswith("?"):
        return ClaimType.UNKNOWN
    if (_FINANCIAL_PRICE_RE.search(text) and (re.search(r"\b(?:price|rate|cost)\b", text, re.IGNORECASE) or _PER_UNIT_PRICE_RE.search(text))):
        return ClaimType.FINANCIAL_PRICE
    if _LEGAL_RE.search(text):
        return ClaimType.LEGAL_COURT
    if _SCIENTIFIC_RE.search(text):
        return ClaimType.SCIENTIFIC
    if _SPORTS_RE.search(text) and _SPORTS_RESULT_RE.search(text):
        return ClaimType.SPORTS_RESULT
    if _GOVERNMENT_RE.search(text) and (_ANNOUNCEMENT_RE.search(text) or _PRODUCT_ACTION_RE.search(text)):
        return ClaimType.GOVERNMENT_ANNOUNCEMENT
    if _PRODUCT_ACTION_RE.search(text) and (_PRODUCT_CUE_RE.search(text) or _has_capitalized_product_after_action(text)):
        return ClaimType.PRODUCT_ANNOUNCEMENT
    if _COMPANY_RE.search(text) and (_ANNOUNCEMENT_RE.search(text) or _PRODUCT_ACTION_RE.search(text)):
        return ClaimType.COMPANY_ANNOUNCEMENT
    if _ORGANIZATION_RE.search(text) and (_ANNOUNCEMENT_RE.search(text) or _PRODUCT_ACTION_RE.search(text)):
        return ClaimType.ORGANIZATION_ANNOUNCEMENT
    if _PERSON_STATEMENT_RE.search(text):
        return ClaimType.PERSON_STATEMENT
    if _EVENT_RE.search(text):
        return ClaimType.EVENT
    if _FACTUAL_RE.search(text):
        return ClaimType.GENERAL_FACT
    if _OTHER_ACTION_RE.search(text):
        return ClaimType.OTHER
    return ClaimType.UNKNOWN


def _clean_subject(subject: str) -> str | None:
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", subject.strip(), flags=re.IGNORECASE)
    if not cleaned or cleaned.casefold() in {"today", "yesterday", "currently", "latest"}:
        return None
    return cleaned


def _extract_subject(text: str, claim_type: ClaimType) -> str | None:
    if claim_type == ClaimType.FINANCIAL_PRICE:
        for commodity in _COMMODITIES:
            match = re.search(rf"\b{re.escape(commodity)}\b", text, re.IGNORECASE)
            if match:
                return match.group(0).casefold()
        price_subject = re.search(r"\b(?P<subject>[A-Za-z][A-Za-z0-9'’\-]*(?:\s+[A-Za-z][A-Za-z0-9'’\-]*){0,3})\s+(?:price|rate|cost)\b", text, re.IGNORECASE)
        if price_subject:
            return _clean_subject(price_subject.group("subject"))

    subject_text = _LEADING_TIME_RE.sub("", text)
    match = _SUBJECT_VERB_RE.search(subject_text)
    if match:
        return _clean_subject(match.group("subject"))
    return None


def _extract_event(text: str, claim_type: ClaimType) -> str | None:
    if claim_type == ClaimType.FINANCIAL_PRICE:
        return "price"
    if claim_type == ClaimType.PRODUCT_ANNOUNCEMENT:
        if re.search(r"\bunveil", text, re.IGNORECASE):
            return "product unveiling"
        if re.search(r"\breleas", text, re.IGNORECASE):
            return "product release"
        return "product launch"
    if claim_type in {
        ClaimType.GOVERNMENT_ANNOUNCEMENT,
        ClaimType.COMPANY_ANNOUNCEMENT,
        ClaimType.ORGANIZATION_ANNOUNCEMENT,
    }:
        return "announcement"
    if claim_type == ClaimType.PERSON_STATEMENT:
        return "statement"
    if claim_type == ClaimType.LEGAL_COURT:
        return "court/legal outcome"
    if claim_type == ClaimType.SPORTS_RESULT:
        return "sports result"
    if claim_type == ClaimType.EVENT:
        return "event"
    return None


def _looks_like_independent_clause(text: str) -> bool:
    """Return true only for conjunction suffixes with an explicit new predicate."""
    predicate = r"(?:is|are|was|were|has|have|had|will|would|can|could|costs?|priced|announced|"
    predicate += r"launched|unveiled|introduced|released|confirmed|said|stated|claimed|won|defeated|"
    predicate += r"began|started|ended|occurred|happened)\b"
    lower = text.casefold()
    if re.match(rf"(?:the|this|that|it|they|its)\s+.+?\b{predicate}", lower):
        return True
    return bool(re.match(rf"[A-Z][A-Za-z0-9'’\-]*(?:\s+[A-Z][A-Za-z0-9'’\-]*){{0,3}}\s+{predicate}", text))


def _split_obvious_clauses(sentence: str) -> list[str]:
    terminal = sentence[-1] if sentence and sentence[-1] in ".!?" else ""
    core = sentence[:-1].strip() if terminal else sentence.strip()
    raw_clauses = [part.strip(" ;") for part in core.split(";") if part.strip(" ;")]
    clauses: list[str] = []
    for raw_clause in raw_clauses:
        remaining = raw_clause
        while True:
            split_match = re.search(r"\s+(?:and|but)\s+", remaining, re.IGNORECASE)
            if not split_match:
                clauses.append(remaining)
                break
            suffix = remaining[split_match.end():]
            if not _looks_like_independent_clause(suffix):
                clauses.append(remaining)
                break
            clauses.append(remaining[:split_match.start()].strip())
            remaining = suffix.strip()
    return [f"{clause}{terminal}" if terminal else clause for clause in clauses if clause]


def split_claim_text(text: object) -> list[str]:
    """Split sentences and only clearly independent coordinated statements."""
    normalized = _normalise_text(text)
    if not normalized:
        return []
    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence]
    return [claim for sentence in sentences for claim in _split_obvious_clauses(sentence)]


def extract_claims(text: object) -> list[Claim]:
    """Create stable claim IDs for deterministic sentence/clause candidates."""
    return [
        Claim(
            claim_id=f"claim-{index}",
            text=claim_text,
            extraction_method="deterministic_claim_understanding",
        )
        for index, claim_text in enumerate(split_claim_text(text), 1)
    ]


def understand_claim(claim: Claim | str, *, claim_id: str | None = None) -> ClaimContext:
    """Return a conservative structured context for one already-identified claim."""
    if isinstance(claim, Claim):
        resolved_claim_id = claim.claim_id
        text = _normalise_text(claim.text)
    else:
        resolved_claim_id = claim_id or "claim-1"
        text = _normalise_text(claim)

    if not text:
        return ClaimContext(
            claim_id=resolved_claim_id,
            factual_statement="",
            extraction_notes=["No claim text was available for deterministic parsing."],
        )

    numbers, attributes = _extract_numbers(text)
    claim_type = _classify_claim(text)
    notes: list[str] = []
    if claim_type == ClaimType.UNKNOWN:
        notes.append("Claim type could not be determined from explicit text cues.")
    return ClaimContext(
        claim_id=resolved_claim_id,
        factual_statement=text,
        subject=_extract_subject(text, claim_type),
        claim_type=claim_type,
        event=_extract_event(text, claim_type),
        location=_extract_location(text),
        time_reference=_extract_time_reference(text),
        numerical_values=numbers,
        relevant_attributes=attributes,
        extraction_notes=notes,
    )


def understand_claims(claims: Sequence[Claim] | str | object) -> list[ClaimContext]:
    """Understand a sequence of existing claims, preserving every provided ID."""
    if isinstance(claims, str):
        return [understand_claim(claim) for claim in extract_claims(claims)]
    if not isinstance(claims, Sequence):
        return []

    contexts: list[ClaimContext] = []
    for index, raw_claim in enumerate(claims, 1):
        try:
            claim = raw_claim if isinstance(raw_claim, Claim) else Claim.model_validate(raw_claim)
        except Exception:
            continue
        contexts.append(understand_claim(claim, claim_id=f"claim-{index}"))
    return contexts


class ClaimUnderstandingService:
    """Small service boundary used by the news pipeline and focused unit tests."""

    def extract_claims(self, text: object) -> list[Claim]:
        return extract_claims(text)

    def understand_claim(self, claim: Claim | str, *, claim_id: str | None = None) -> ClaimContext:
        return understand_claim(claim, claim_id=claim_id)

    def understand_claims(self, claims: Sequence[Claim] | str | object) -> list[ClaimContext]:
        return understand_claims(claims)


claim_understanding = ClaimUnderstandingService()
