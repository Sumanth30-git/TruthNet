from backend.schemas import (
    Claim,
    ClaimContext,
    ClaimRelation,
    ClaimType,
    ClaimUnderstandingMetadata,
    EntityMention,
    EntityType,
)
from backend.services.claim_understanding import (
    ClaimUnderstandingService,
    extract_claims,
    split_claim_text,
    understand_claim,
    understand_claims,
)


def test_understands_a_single_product_announcement_claim_conservatively():
    context = understand_claim(Claim(claim_id="claim-7", text="Acme launched Product Y today."))

    assert context.claim_id == "claim-7"
    assert context.factual_statement == "Acme launched Product Y today."
    assert context.subject == "Acme"
    assert context.claim_type == ClaimType.PRODUCT_ANNOUNCEMENT
    assert context.event == "product launch"
    assert context.time_reference == "today"
    assert context.location is None


def test_extracts_multiple_sentence_and_independent_clause_claims_with_stable_ids():
    claims = extract_claims("Acme launched Product Y today and the product costs ₹50,000. The ministry announced a policy.")
    contexts = understand_claims(claims)

    assert [(claim.claim_id, claim.text) for claim in claims] == [
        ("claim-1", "Acme launched Product Y today."),
        ("claim-2", "the product costs ₹50,000."),
        ("claim-3", "The ministry announced a policy."),
    ]
    assert [context.claim_id for context in contexts] == ["claim-1", "claim-2", "claim-3"]
    assert contexts[0].claim_type == ClaimType.PRODUCT_ANNOUNCEMENT
    assert contexts[1].claim_type == ClaimType.GENERAL_FACT
    assert contexts[2].claim_type == ClaimType.GOVERNMENT_ANNOUNCEMENT


def test_unknown_claim_type_does_not_invent_structure():
    context = understand_claim(Claim(claim_id="claim-1", text="Hello there."))

    assert context.claim_type == ClaimType.UNKNOWN
    assert context.subject is None
    assert context.event is None
    assert context.location is None
    assert context.time_reference is None
    assert context.numerical_values == []
    assert context.extraction_notes


def test_extracts_specific_and_relative_time_references():
    specific = understand_claim(Claim(claim_id="claim-1", text="On March 5, 2025, Acme announced an update."))
    relative = understand_claim(Claim(claim_id="claim-2", text="The company will publish results this week."))

    assert specific.time_reference == "2025-03-05"
    assert relative.time_reference == "this week"


def test_extracts_location_from_explicit_location_phrase():
    context = understand_claim(Claim(claim_id="claim-1", text="Today's gold price in Bengaluru is ₹15,268/g for 24K gold."))

    assert context.claim_type == ClaimType.FINANCIAL_PRICE
    assert context.subject == "gold"
    assert context.location == "Bengaluru"
    assert context.time_reference == "today"


def test_extracts_price_number_unit_currency_and_purity_without_guessing_tolerance():
    context = understand_claim(Claim(claim_id="claim-1", text="Today's gold price in Bengaluru is ₹15,268/g for 24K gold."))

    assert [(number.raw_text, number.value, number.unit, number.attribute) for number in context.numerical_values] == [
        ("₹15,268/g", 15268.0, "gram", "price"),
    ]
    assert context.relevant_attributes == {"currency": "INR", "purity": "24K"}


def test_normalizes_common_measurement_abbreviations():
    cases = [
        ("Gold price is ₹15,268/g.", "gram"),
        ("Weight is 5 kg.", "kilogram"),
        ("Medicine contains 500 mg.", "milligram"),
        ("Container holds 250 ml.", "milliliter"),
        ("The distance is 10 km.", "kilometer"),
        ("The table is 2 m long.", "meter"),
        ("Package weighs 10 lbs.", "pound"),
    ]

    for text, expected_unit in cases:
        context = understand_claim(Claim(claim_id="claim-1", text=text))
        assert [number.unit for number in context.numerical_values] == [expected_unit]


def test_text_api_and_service_boundary_are_safe_for_empty_input():
    service = ClaimUnderstandingService()

    assert split_claim_text(None) == []
    assert service.extract_claims(" ") == []
    assert service.understand_claims(None) == []
    empty = service.understand_claim(" ", claim_id="claim-empty")
    assert empty.claim_id == "claim-empty"
    assert empty.claim_type == ClaimType.UNKNOWN


def test_additive_nlp_contract_defaults_are_empty_or_unassessed():
    context = ClaimContext(claim_id="claim-1", factual_statement="A claim.")

    assert context.entity_mentions == []
    assert context.relations == []
    assert context.understanding_metadata is None
    assert context.is_atomic_claim is None
    assert context.is_verifiable_claim is None


def test_additive_entity_relation_and_metadata_contracts_are_conservative():
    entity = EntityMention(
        text="Acme",
        entity_type=EntityType.ORGANIZATION,
        start=0,
        end=4,
        normalized_text="acme",
        confidence=0.8,
        status="complete",
    )
    relation = ClaimRelation(subject_text="Acme", predicate=None, qualifiers=["reported"])
    metadata = ClaimUnderstandingMetadata(
        method="deterministic_claim_understanding",
        status="complete",
    )

    context = ClaimContext(
        claim_id="claim-1",
        factual_statement="Acme reported an update.",
        entity_mentions=[entity],
        relations=[relation],
        understanding_metadata=metadata,
    )

    assert context.entity_mentions == [entity]
    assert context.relations == [relation]
    assert context.relations[0].predicate is None
    assert context.understanding_metadata == metadata
    assert context.understanding_metadata.model_id is None
    assert context.understanding_metadata.model_version is None


def test_deterministic_context_has_metadata_without_invented_nlp_structure():
    context = understand_claim(Claim(claim_id="claim-1", text="Acme launched Product Y today."))

    assert context.understanding_metadata is not None
    assert context.understanding_metadata.method == "deterministic_claim_understanding"
    assert context.understanding_metadata.status == "complete"
    assert context.understanding_metadata.model_id is None
    assert context.understanding_metadata.model_version is None
    assert context.entity_mentions == []
    assert context.relations == []
    assert context.is_atomic_claim is None
    assert context.is_verifiable_claim is None


def test_understanding_preserves_original_claim_text():
    claim = Claim(claim_id="claim-1", text="Acme   launched Product Y today.")

    understand_claim(claim)

    assert claim.text == "Acme   launched Product Y today."
