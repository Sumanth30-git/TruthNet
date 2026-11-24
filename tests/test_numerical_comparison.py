from backend.schemas import Claim, NumericalStatus
from backend.services.numerical_comparison import NumericalComparisonService
from backend.services.claim_understanding import understand_claim


def _context(text: str):
    return understand_claim(Claim(claim_id="claim-1", text=text))


def test_exact_percent_count_currency_and_unit_aliases_are_recorded():
    service = NumericalComparisonService()
    cases = [
        ("Growth reached 7.1%.", "Growth reached 7.1 percent."),
        ("The program has 1,200 employees.", "The program has 1,200 employees."),
        ("The gold price is $1,500.", "The gold price is USD 1,500."),
        ("The shipment weighs 5 kg.", "The shipment weighs 5 kilograms."),
    ]

    for claim, source in cases:
        check = service.compare(_context(claim), source)
        assert check is not None
        assert check.status == NumericalStatus.EXACT_MATCH
        assert check.absolute_difference == 0


def test_percentage_and_currency_differences_are_explicit_mismatches():
    service = NumericalComparisonService()
    percentage = service.compare(_context("Growth reached 7%."), "Growth reached 8%.")
    currency = service.compare(_context("The gold price is $1,500."), "The gold price is EUR 1,500.")

    assert percentage.status == NumericalStatus.MISMATCH
    assert (percentage.claim_value, percentage.source_value, percentage.absolute_difference) == (7, 8, 1)
    assert currency.status == NumericalStatus.UNIT_MISMATCH


def test_ranges_compare_each_explicit_value_without_collapsing_the_range():
    service = NumericalComparisonService()
    context = _context("Obesity prevalence rose from 2% to 8%.")

    matching = service.compare(context, "Obesity prevalence rose from 2 percent to 8 percent.")
    differing = service.compare(context, "Obesity prevalence rose from 3 percent to 8 percent.")

    assert matching.status == NumericalStatus.EXACT_MATCH
    assert differing.status == NumericalStatus.MISMATCH
    assert differing.claim_raw == "2%" and differing.source_raw == "3 percent"


def test_explicit_dates_and_years_are_compared_conservatively():
    service = NumericalComparisonService()
    exact = service.compare(_context("NASA launched the mission on March 5, 2025."), "The mission launched on 2025-03-05.")
    mismatch = service.compare(_context("NASA launched the mission on March 5, 2025."), "The mission launched on March 6, 2025.")
    year_mismatch = service.compare(_context("The economy grew in 2026."), "The economy grew in 2025.")

    assert exact.status == NumericalStatus.EXACT_MATCH
    assert mismatch.status == NumericalStatus.MISMATCH
    assert year_mismatch.status == NumericalStatus.MISMATCH


def test_approximate_ambiguous_missing_and_malformed_values_are_not_assessed():
    service = NumericalComparisonService()
    approximate = service.compare(_context("About 100 employees joined."), "99 employees joined.")
    ambiguous = service.compare(_context("Revenue rose 10%."), "Revenue rose 8% and 12%.")
    missing = service.compare(_context("Revenue rose 10%."), "Revenue rose significantly.")
    malformed = service.compare(_context("Revenue rose 10%."), None)

    assert approximate.status == NumericalStatus.NOT_ASSESSED
    assert ambiguous.status == NumericalStatus.NOT_ASSESSED
    assert missing.status == NumericalStatus.NOT_ASSESSED
    assert malformed.status == NumericalStatus.NOT_ASSESSED


def test_claims_without_numeric_or_explicit_date_content_do_not_get_a_check():
    service = NumericalComparisonService()

    assert service.compare(_context("NASA launched a mission."), "NASA launched a mission.") is None
    assert service.compare(_context("The mission is expected to launch soon."), "The mission is expected to launch soon.") is None
