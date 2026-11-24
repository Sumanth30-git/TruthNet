"""Conservative deterministic comparison of explicit claim/source values."""

from __future__ import annotations

import re
from datetime import date

from backend.schemas import Claim, ClaimContext, ClaimNumber, NumericalCheck, NumericalStatus
from backend.services.claim_understanding import understand_claim


_APPROXIMATE_RE = re.compile(r"\b(?:about|around|approximately|approx\.?|roughly|nearly)\b|~", re.I)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ISO_DATE_RE = re.compile(r"\b(?P<year>(?:19|20)\d{2})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b")
_MONTHS = (
    "january february march april may june july august september october november december"
).split()
_MONTH_DAY_RE = re.compile(
    rf"\b(?P<month>{'|'.join(_MONTHS)})\s+(?P<day>\d{{1,2}}),?\s+(?P<year>(?:19|20)\d{{2}})\b",
    re.I,
)
_DAY_MONTH_RE = re.compile(
    rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{'|'.join(_MONTHS)})\s+(?P<year>(?:19|20)\d{{2}})\b",
    re.I,
)


def _explicit_dates(text: str) -> list[tuple[str, str]]:
    """Return validated explicit dates and years without interpreting relative dates."""
    values: list[tuple[str, str]] = []
    covered: list[tuple[int, int]] = []
    for pattern in (_ISO_DATE_RE, _MONTH_DAY_RE, _DAY_MONTH_RE):
        for match in pattern.finditer(text):
            try:
                month_value = match.group("month")
                month = _MONTHS.index(month_value.casefold()) + 1 if not month_value.isdigit() else int(month_value)
                normalized = date(int(match.group("year")), month, int(match.group("day"))).isoformat()
            except (ValueError, TypeError):
                continue
            values.append((normalized, match.group(0)))
            covered.append(match.span())
    for match in _YEAR_RE.finditer(text):
        if any(start <= match.start() < end for start, end in covered):
            continue
        values.append((match.group(0), match.group(0)))
    return values


def _date_check(context: ClaimContext, source_text: str) -> NumericalCheck | None:
    reference = (context.time_reference or "").strip()
    if not reference:
        return None
    claim_dates = _explicit_dates(reference)
    if not claim_dates:
        return None
    source_dates = _explicit_dates(source_text)
    claim_value, claim_raw = claim_dates[0]
    if not source_dates:
        return NumericalCheck(
            status=NumericalStatus.NOT_ASSESSED,
            claim_raw=claim_raw,
            unit="date",
            reason="The claim has an explicit date, but the source text has no comparable date.",
        )

    claim_is_full_date = len(claim_value) > 4
    exact = next((raw for value, raw in source_dates if value == claim_value), None)
    if exact is not None:
        return NumericalCheck(
            status=NumericalStatus.EXACT_MATCH,
            claim_raw=claim_raw,
            source_raw=exact,
            unit="date" if claim_is_full_date else "year",
            reason="The explicit claim date is present in the source text.",
        )

    comparable = [(value, raw) for value, raw in source_dates if len(value) == len(claim_value)]
    if len(comparable) == 1:
        source_value, source_raw = comparable[0]
        return NumericalCheck(
            status=NumericalStatus.MISMATCH,
            claim_raw=claim_raw,
            source_raw=source_raw,
            claim_value=float(claim_value[:4]) if not claim_is_full_date else None,
            source_value=float(source_value[:4]) if not claim_is_full_date else None,
            unit="year" if not claim_is_full_date else "date",
            reason="The source has one explicit date at the same level of precision, and it differs from the claim.",
        )
    return NumericalCheck(
        status=NumericalStatus.NOT_ASSESSED,
        claim_raw=claim_raw,
        unit="date" if claim_is_full_date else "year",
        reason="The source contains multiple dates, so the claim date cannot be matched unambiguously.",
    )


def _number_key(number: ClaimNumber, currency: str | None) -> tuple[str | None, str | None, str | None]:
    return number.attribute, number.unit, currency


def _not_assessed(claim_number: ClaimNumber, reason: str) -> NumericalCheck:
    return NumericalCheck(
        status=NumericalStatus.NOT_ASSESSED,
        claim_raw=claim_number.raw_text,
        claim_value=claim_number.value,
        unit=claim_number.unit,
        reason=reason,
    )


class NumericalComparisonService:
    """Compare only unambiguous, explicitly extracted values; never infer truth."""

    def compare(self, context: ClaimContext, source_text: str | None) -> NumericalCheck | None:
        claim_numbers = context.numerical_values
        text = source_text if isinstance(source_text, str) else ""
        if not claim_numbers:
            return _date_check(context, text)
        if not text.strip():
            return _not_assessed(claim_numbers[0], "No usable source text was available for numerical comparison.")

        source_context = understand_claim(Claim(claim_id=context.claim_id, text=text))
        source_numbers = source_context.numerical_values
        if not source_numbers:
            return _not_assessed(claim_numbers[0], "The source text contains no comparable extracted numerical value.")

        claim_currency = context.relevant_attributes.get("currency")
        source_currency = source_context.relevant_attributes.get("currency")
        unused_source_indexes = set(range(len(source_numbers)))
        exact_pairs: list[tuple[ClaimNumber, ClaimNumber]] = []
        mismatches: list[tuple[ClaimNumber, ClaimNumber, NumericalStatus]] = []
        ambiguous: ClaimNumber | None = None
        unmatched_claim_numbers: list[ClaimNumber] = []

        for claim_number in claim_numbers:
            claim_key = _number_key(claim_number, claim_currency)
            matching_indexes = [
                index for index in sorted(unused_source_indexes)
                if _number_key(source_numbers[index], source_currency) == claim_key
            ]
            exact_index = next(
                (index for index in matching_indexes if source_numbers[index].value == claim_number.value),
                None,
            )
            if exact_index is not None:
                exact_pairs.append((claim_number, source_numbers[exact_index]))
                unused_source_indexes.remove(exact_index)
                continue
            unmatched_claim_numbers.append(claim_number)

        for claim_number in unmatched_claim_numbers:
            claim_key = _number_key(claim_number, claim_currency)
            matching_indexes = [
                index for index in sorted(unused_source_indexes)
                if _number_key(source_numbers[index], source_currency) == claim_key
            ]
            if len(matching_indexes) == 1:
                source_number = source_numbers[matching_indexes[0]]
                if (
                    _APPROXIMATE_RE.search(context.factual_statement)
                    or _APPROXIMATE_RE.search(text)
                ):
                    ambiguous = claim_number
                else:
                    mismatches.append((claim_number, source_number, NumericalStatus.MISMATCH))
                continue

            related_indexes = [
                index for index in sorted(unused_source_indexes)
                if source_numbers[index].attribute == claim_number.attribute
            ]
            if not matching_indexes and len(related_indexes) == 1:
                mismatches.append((claim_number, source_numbers[related_indexes[0]], NumericalStatus.UNIT_MISMATCH))
            elif not matching_indexes:
                ambiguous = claim_number
            else:
                ambiguous = claim_number

        if mismatches:
            claim_number, source_number, status = mismatches[0]
            difference = abs(claim_number.value - source_number.value)
            relative = difference / abs(claim_number.value) if claim_number.value else None
            return NumericalCheck(
                status=status,
                claim_raw=claim_number.raw_text,
                source_raw=source_number.raw_text,
                claim_value=claim_number.value,
                source_value=source_number.value,
                unit=claim_number.unit,
                absolute_difference=difference,
                relative_difference=relative,
                tolerance=0,
                reason=(
                    "A uniquely comparable source value differs from the claim."
                    if status == NumericalStatus.MISMATCH
                    else "A uniquely related source value uses a different or incompatible unit/currency."
                ),
            )
        if ambiguous is not None:
            return _not_assessed(ambiguous, "The available values are missing or ambiguous; no numerical conclusion was assigned.")
        if exact_pairs:
            claim_number, source_number = exact_pairs[0]
            return NumericalCheck(
                status=NumericalStatus.EXACT_MATCH,
                claim_raw=claim_number.raw_text,
                source_raw=source_number.raw_text,
                claim_value=claim_number.value,
                source_value=source_number.value,
                unit=claim_number.unit,
                absolute_difference=0,
                relative_difference=0,
                tolerance=0,
                reason="Explicit claim and source values match exactly after the existing unit-alias normalization.",
            )
        return None


numerical_comparison = NumericalComparisonService()
