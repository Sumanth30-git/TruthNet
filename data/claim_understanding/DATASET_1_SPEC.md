# Dataset 1 — Claim Understanding Evaluation Dataset

## Scope and target

Dataset 1 evaluates Claim Understanding only. It is not a fake-news dataset,
does not determine a TruthNet verdict, and is not imported by production code.

Target corpus: 230 English news-style documents and approximately 460 claims:

- development: 80 documents;
- validation: 50 documents;
- frozen test: 100 documents.

Frozen test must contain no synthetic documents. Every document is independently
annotated twice, followed by expert adjudication and quality control before
freeze. This repository currently contains infrastructure only; its JSONL files
are deliberately empty.

## Canonical text and offsets

`original_text` is preserved verbatim. `normalized_text` is UTF-8 NFC text with
CRLF/CR line endings converted to LF. Offsets are zero-based, half-open Python
Unicode-code-point offsets into `normalized_text`; every span stores its exact
substring in `text`. `normalized_to_original_offsets` maps each normalized text
boundary back to an original-text boundary. The normalizer performs no trimming,
paraphrasing, correction, or semantic rewriting.

## Gold annotations

The adjudicated gold annotation records claim segmentation, source span,
original claim text, ClaimType, atomicity, verifiability, entities, relations,
temporal annotations, quantities, locations, modifiers, negation, modality,
non-claim spans, and explicit abstentions where applicable. Annotation uncertainty
is represented with nullable labels and `abstentions`; it must not be fabricated
as a positive assertion.

`abstentions` is the sole abstention representation:
`[{"scope": "document"|"claim", "reason": ..., "claim_id": ...}]`.
The scorer sets `scorer_abstained` (and contributes to corpus abstention rate)
only for a valid document-scope abstention. A claim-scope record is scoped to
that claim and does not make the whole document abstained. An empty prediction
without a document-scope record is not silently recast as abstention.

## USR and structure matching

**Unsupported Structure Rate (USR)** is the primary metric:

`USR = unsupported predicted structures / all asserted predicted structures`.

The denominator includes every predicted claim segment and every non-null
atomicity, verifiability, ClaimType, entity, relation, temporal, quantity,
location, modifier, negation, modality, and non-claim structure. A prediction is
supported only when it exactly matches an adjudicated gold structure of the same
kind. Exact matching requires all specified semantic values and source support
spans to match; source spans match only when start, end, and text all match.

- Entity: equal `EntityType` and source span.
- Relation: equal relation type, predicate, subject/object spans, all support
  spans, negation, modality, and qualifier set.
- Date/temporal, quantity, location, modifier, negation, modality: equal source
  span and all stated values.
- A partially overlapping but non-exact prediction is recorded as `partial`; it
  remains a false positive and unsupported assertion until a future approved
  policy changes this rule.

Results explicitly expose `EXACT_MATCH`, `PARTIAL_BOUNDARY_MATCH`, and
`UNSUPPORTED`. Strict precision/recall/F1 count every non-exact assertion as a
false positive; partial-boundary outcomes are also reported separately so they
are never hidden as ordinary unsupported structures.

The scorer also reports structure precision, recall, and F1. A model that asserts
no structures receives `USR = null` and precision/recall/F1 of zero. It counts
as abstained only when it emits the canonical document-scope abstention record;
it is therefore never rewarded with USR zero.

## Split and freeze contract

Splits are document- and story-cluster-disjoint. Each split row includes the
cluster ID, split algorithm version, and assignment seed. Exact target counts are
enforced only after the corpus is populated; an empty corpus reports `dataset not
populated`. Freeze requires successful schema, provenance, duplicate, split,
double-annotation, adjudication, and QC checks, then immutable content hashes in
the freeze manifest.

An empty artifact layout is a valid `empty_scaffold` validation state, but is
not a populated corpus and cannot satisfy population or freeze requirements.
