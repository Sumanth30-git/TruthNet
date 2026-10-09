# Dataset 1 Stage 3A — Sourcing and Collection Workflow

## Status and boundary

This is the reproducible workflow for selecting Dataset 1 source documents. It
does not authorize collection, downloading, scraping, annotation, or changes to
the approved Dataset 1 design. The repository contains an empty candidate ledger
at `provenance/candidates.jsonl`; no candidate or corpus document has been
collected.

Before collection begins, a human must approve the source eligibility plan,
licensing/retention approach, date coverage, diversity targets, and any
permissible synthetic development/validation material as required by
`SOURCE_POLICY.md`. Frozen-test material must never be synthetic.

## Records and identifiers

`provenance/candidates.jsonl` is the pre-normalization **collection ledger**.
It is not a second document-provenance model: it records discovery and the
collection decision while a document is only a candidate. A retained candidate
then has exactly one normalized document and exactly one existing row in
`provenance/sources.jsonl`. The three records are joined by the retained
candidate's `retained_document_id` and the existing `source_id`; its collected
content checksum must agree with the provenance checksum. `source_id` retains
its existing role as the source identifier shared by a normalized document and
its provenance record.

Each candidate uses the `collection_candidate.schema.json` contract. Its
`source_identity` names the publisher or official organization responsible for
the presented document, not merely the host URL. `identity_id` is the stable,
documented identifier used consistently for that source identity. A URL, a
mirror, and a wire-service copy do not by themselves establish independent
source identities.

## Discovery and diversity

Discover candidates only through documented routes: curated source review,
publisher or official archives, licensed archives, or a recorded manual-search
route. Record the discovery method, route/query or archive path, discovery time,
and discoverer in every ledger row. Acceptable categories are the schema's
controlled values: publisher, broadcaster, wire service, government, official
organization, licensed archive, and other documented sources.

The human-approved collection plan must set category, publisher/source-identity,
topic, and time-coverage targets before selection. Track progress against those
targets outside the corpus artifacts; do not satisfy diversity by silently
counting multiple URLs, mirrors, or syndicated copies as independent sources.

## Deterministic eligibility gate

A candidate is `eligible` or `retained` only when every recorded eligibility
check is true: English language, meaningful text, attributable source, stable
source identity, accessible content, licensing review, and source-independence
review. It must have accessible collected content with a SHA-256 checksum and a
positive content length. Retention additionally requires a documented permitted
usage assessment and a verified or not-applicable license status.

Candidate language is `en` only; another language is represented as an excluded
ledger record with `unsupported_language` rather than being normalized. Meaningful
text means sufficient article/document body for claim-understanding annotation;
headlines are retained with the source text but headline-only, navigation-only,
or materially inaccessible pages fail this check. Attribution requires a named
publisher/official source or documented archive record. A stable identity requires
a documented identity basis and stable `identity_id`; do not infer it from a URL.

Every non-retained candidate is `excluded` with one explicit controlled
`exclusion_reason`. An eligible candidate may remain unretained while selection
is pending; it must not be normalized or assigned to a split until retained.

## Collection metadata and temporal handling

Before normalization, record canonical URI, source category, source identity,
independence assessment, access status, UTC collection timestamp, collector,
content SHA-256, content length, content format, licensing assessment, discovery
metadata, and eligibility results. Preserve the acquired original text only in
the later normalized document workflow; normalization must still run exclusively
through `scripts/dataset1_normalize.py`.

Record source publication information exactly as presented. `publication.value`
is either an ISO date (`precision: date`) or a timezone-bearing ISO timestamp
(`precision: timestamp`). For unavailable or uncertain information, use the
corresponding precision and a null value; retain the displayed wording in
`publication.verbatim` when present. Never infer a date or timezone from URL,
collection date, page layout, or surrounding stories.

## Independence and duplicate review

Record `primary_or_official`, `independent_reporting`,
`syndicated_or_republished`, `copy_or_mirror`, or `unknown_needs_review` for
each candidate. Syndicated/reprinted and copied/mirrored material must identify
the known origin URI and origin source identity; it is not independent reporting.
Primary/official material and secondary reporting remain distinct even where they
concern the same event.

The ledger records a human duplicate-review outcome. Exact duplicates, known
syndicated text, and reviewed near duplicates point to one canonical candidate;
such candidate rows cannot be retained. A near-duplicate or independence concern
is never auto-resolved by fuzzy matching. After normalization, existing duplicate
QC and human story clustering remain mandatory before split assignment.

## Reproducibility and audit trail

Another researcher must be able to reconstruct, from the candidate ledger and
the existing provenance record, where a retained document came from, when and by
whom it was discovered/collected, why it met every eligibility check, which
source identity and independence relationship it has, its licensing assessment,
and its publication-date precision. Excluded candidates remain in the ledger
with their explicit reason. The ledger and its schema are part of the Dataset 1
freeze inventory once data exists.
