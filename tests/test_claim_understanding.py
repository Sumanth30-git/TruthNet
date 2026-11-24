import sys
from types import SimpleNamespace

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
    DeterministicClaimUnderstandingAdapter,
    extract_claims,
    split_claim_text,
    understand_claim,
    understand_claims,
)


class FakeEntity:
    def __init__(self, text: str, label: str, source: str) -> None:
        self.label_ = label
        self.start_char = source.index(text)
        self.end_char = self.start_char + len(text)


class FakeNLP:
    def __init__(self, entities: list[FakeEntity] | None = None) -> None:
        self.ents = entities or []
        self.meta = {"version": "3.8.0-test"}
        self.inputs: list[str] = []

    def __call__(self, text: str):
        self.inputs.append(text)
        return SimpleNamespace(ents=self.ents)


def _install_fake_spacy(monkeypatch, *, entities: list[FakeEntity] | None = None):
    load_calls: list[str] = []
    pipeline = FakeNLP(entities)

    def load(model_name: str):
        load_calls.append(model_name)
        return pipeline

    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=load))
    return load_calls, pipeline


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


def test_deterministic_location_extraction_rejects_possessives_without_changing_other_scalars():
    cases = [
        ("Moon's surface contains water ice.", "Moon's surface"),
        ("Moon\u2019s surface contains water ice.", "Moon\u2019s surface"),
        ("India's space programme has expanded significantly.", "India's space programme"),
        ("India\u2019s space programme has expanded significantly.", "India\u2019s space programme"),
    ]

    for text, expected_subject in cases:
        claim = Claim(claim_id="claim-1", text=text)
        context = understand_claim(claim)

        assert claim.text == text
        assert context.factual_statement == text
        assert context.subject == expected_subject
        assert context.claim_type == ClaimType.GENERAL_FACT
        assert context.event is None
        assert context.time_reference is None
        assert context.numerical_values == []
        assert context.relevant_attributes == {}
        assert context.location is None


def test_deterministic_location_extraction_keeps_explicit_from_location():
    claim = Claim(claim_id="claim-1", text="Chandrayaan-3 was launched from India.")

    context = understand_claim(claim)

    assert claim.text == "Chandrayaan-3 was launched from India."
    assert context.location == "India"
    assert context.subject == "Chandrayaan-3"


def test_from_preposition_does_not_turn_a_possessive_name_into_a_location():
    for text in (
        "The mission launched from India's spaceport.",
        "The mission launched from India\u2019s spaceport.",
    ):
        context = understand_claim(Claim(claim_id="claim-1", text=text))
        assert context.location is None


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


def test_spacy_is_disabled_by_default_and_keeps_deterministic_behavior(monkeypatch):
    monkeypatch.delenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", raising=False)
    load_calls, _ = _install_fake_spacy(monkeypatch)
    service = ClaimUnderstandingService()

    context = service.understand_claim(Claim(claim_id="claim-1", text="Acme launched Product Y today."))

    assert context.understanding_metadata.method == "deterministic_claim_understanding"
    assert context.entity_mentions == []
    assert load_calls == []


def test_spacy_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    text = "NASA launched a mission."
    load_calls, _ = _install_fake_spacy(monkeypatch, entities=[FakeEntity("NASA", "ORG", text)])
    service = ClaimUnderstandingService()

    context = service.understand_claim(Claim(claim_id="claim-1", text=text))

    assert load_calls == ["en_core_web_sm"]
    assert context.entity_mentions[0].entity_type == EntityType.ORGANIZATION
    assert context.understanding_metadata.method == "spacy_claim_understanding"


def test_spacy_load_is_lazy_and_reuses_the_model(monkeypatch):
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    load_calls, _ = _install_fake_spacy(monkeypatch)
    service = ClaimUnderstandingService()

    claims = service.extract_claims("First claim. Second claim.")
    assert len(claims) == 2
    assert load_calls == []

    service.understand_claim(claims[0])
    service.understand_claim(claims[1])

    assert load_calls == ["en_core_web_sm"]


def test_missing_spacy_fails_safely_with_deterministic_scalars(monkeypatch):
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    monkeypatch.setitem(sys.modules, "spacy", None)
    claim = Claim(claim_id="claim-1", text="Acme launched Product Y in Bengaluru today.")

    context = ClaimUnderstandingService().understand_claim(claim)
    deterministic = DeterministicClaimUnderstandingAdapter().understand_claim(claim)

    assert context.subject == deterministic.subject
    assert context.location == deterministic.location
    assert context.event == deterministic.event
    assert context.time_reference == deterministic.time_reference
    assert context.entity_mentions == []
    assert context.relations == []
    assert context.understanding_metadata.status == "unavailable"
    assert context.understanding_metadata.model_id == "en_core_web_sm"


def test_missing_spacy_model_fails_safely_without_downloading(monkeypatch):
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    download_calls: list[str] = []

    def missing_model(model_name: str):
        raise OSError(f"{model_name} is not installed")

    fake_spacy = SimpleNamespace(
        load=missing_model,
        cli=SimpleNamespace(download=lambda name: download_calls.append(name)),
    )
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    claim = Claim(claim_id="claim-1", text="Acme launched Product Y today.")

    context = ClaimUnderstandingService().understand_claim(claim)

    assert context.claim_type == ClaimType.PRODUCT_ANNOUNCEMENT
    assert context.entity_mentions == []
    assert context.relations == []
    assert context.understanding_metadata.status == "unavailable"
    assert download_calls == []


def test_spacy_maps_only_supported_entity_labels(monkeypatch):
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    text = "Alice NASA India Bengaluru airport Product X launch today noon"
    labels = [
        ("Alice", "PERSON"),
        ("NASA", "ORG"),
        ("India", "GPE"),
        ("Bengaluru", "LOC"),
        ("airport", "FAC"),
        ("Product X", "PRODUCT"),
        ("launch", "EVENT"),
        ("today", "DATE"),
        ("noon", "TIME"),
        ("Alice", "NORP"),
    ]
    entities = [FakeEntity(mention, label, text) for mention, label in labels]
    _install_fake_spacy(monkeypatch, entities=entities)

    context = ClaimUnderstandingService().understand_claim(Claim(claim_id="claim-1", text=text))

    assert [(entity.text, entity.entity_type) for entity in context.entity_mentions] == [
        ("Alice", EntityType.PERSON),
        ("NASA", EntityType.ORGANIZATION),
        ("India", EntityType.LOCATION),
        ("Bengaluru", EntityType.LOCATION),
        ("airport", EntityType.LOCATION),
        ("Product X", EntityType.PRODUCT),
        ("launch", EntityType.EVENT),
    ]
    assert all(entity.confidence is None for entity in context.entity_mentions)
    assert context.understanding_metadata.model_id == "en_core_web_sm"
    assert context.understanding_metadata.model_version == "3.8.0-test"
    assert context.understanding_metadata.inference_time_ms is not None
    assert context.relations == []


def test_unsupported_spacy_labels_are_dropped(monkeypatch):
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    text = "Acme published today at noon."
    entities = [
        FakeEntity("today", "DATE", text),
        FakeEntity("noon", "TIME", text),
    ]
    _install_fake_spacy(monkeypatch, entities=entities)

    context = ClaimUnderstandingService().understand_claim(Claim(claim_id="claim-1", text=text))

    assert context.entity_mentions == []


def test_spacy_preserves_exact_claim_text_and_original_character_offsets(monkeypatch):
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    text = "  Acme\tlaunched Product Y  "
    claim = Claim(claim_id="claim-1", text=text)
    entities = [FakeEntity("Acme", "ORG", text), FakeEntity("Product Y", "PRODUCT", text)]
    _, pipeline = _install_fake_spacy(monkeypatch, entities=entities)

    context = ClaimUnderstandingService().understand_claim(claim)

    assert claim.text == text
    assert pipeline.inputs == [text]
    assert [(entity.start, entity.end, entity.text) for entity in context.entity_mentions] == [
        (text.index("Acme"), text.index("Acme") + len("Acme"), "Acme"),
        (text.index("Product Y"), text.index("Product Y") + len("Product Y"), "Product Y"),
    ]
    assert all(text[entity.start:entity.end] == entity.text for entity in context.entity_mentions)


def test_possessive_location_spans_are_dropped(monkeypatch):
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    text = "Moon's surface changed. India's probe arrived."
    entities = [FakeEntity("Moon", "GPE", text), FakeEntity("India", "GPE", text)]
    _install_fake_spacy(monkeypatch, entities=entities)

    context = ClaimUnderstandingService().understand_claim(Claim(claim_id="claim-1", text=text))

    assert context.entity_mentions == []


def test_person_prediction_for_moon_remains_person_in_possessive_text(monkeypatch):
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    text = "Moon's surface changed."
    _install_fake_spacy(monkeypatch, entities=[FakeEntity("Moon", "PERSON", text)])

    context = ClaimUnderstandingService().understand_claim(Claim(claim_id="claim-1", text=text))

    assert [(entity.text, entity.entity_type) for entity in context.entity_mentions] == [
        ("Moon", EntityType.PERSON),
    ]


def test_spacy_does_not_replace_deterministic_scalars_and_keeps_relations_empty(monkeypatch):
    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    text = "Acme launched Product Y in Bengaluru on March 5, 2025."
    claim = Claim(claim_id="claim-1", text=text)
    entities = [
        FakeEntity("Acme", "ORG", text),
        FakeEntity("Product Y", "PRODUCT", text),
        FakeEntity("Bengaluru", "GPE", text),
    ]
    _install_fake_spacy(monkeypatch, entities=entities)

    deterministic = DeterministicClaimUnderstandingAdapter().understand_claim(claim)
    context = ClaimUnderstandingService().understand_claim(claim)

    assert context.subject == deterministic.subject
    assert context.claim_type == deterministic.claim_type
    assert context.event == deterministic.event
    assert context.location == deterministic.location
    assert context.time_reference == deterministic.time_reference
    assert context.numerical_values == deterministic.numerical_values
    assert context.relevant_attributes == deterministic.relevant_attributes
    assert context.factual_statement == deterministic.factual_statement
    assert context.relations == []


def test_explicitly_injected_adapter_remains_authoritative(monkeypatch):
    class InjectedAdapter(DeterministicClaimUnderstandingAdapter):
        def understand_claim(self, claim, *, claim_id=None):
            context = super().understand_claim(claim, claim_id=claim_id)
            context.extraction_notes.append("injected adapter")
            return context

    monkeypatch.setenv("CLAIM_UNDERSTANDING_SPACY_ENABLED", "true")
    monkeypatch.setitem(sys.modules, "spacy", None)

    context = ClaimUnderstandingService(adapter=InjectedAdapter()).understand_claim(
        Claim(claim_id="claim-1", text="Acme launched Product Y today.")
    )

    assert "injected adapter" in context.extraction_notes
    assert context.understanding_metadata.method == "deterministic_claim_understanding"
