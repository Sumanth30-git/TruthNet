# Dataset 1 Source and Provenance Policy

No source has been collected by this repository.

Before collection, a human must approve source eligibility, licensing/retention,
date coverage, diversity targets, and the treatment of permissible synthetic
development/validation material. Frozen-test documents must never be synthetic.

Each accepted document requires a provenance record containing its document and
source IDs, source category, acquisition date, license, verification status, URI,
SHA-256 checksum, collector ID, and notes. Source records describe provenance;
they are not evidence of claim truth or source authority.

Document text must be normalized only through `scripts/dataset1_normalize.py`.
Exact and normalized-text duplicates must be reviewed, and near-duplicate
candidates must be story-clustered by a human before split assignment. The
assignment tool never invents or automatically assigns a real story cluster.

## Stage 3A sourcing and collection records

`SOURCING_WORKFLOW.md` is the binding operational workflow for candidate
discovery, eligibility, collection, source identity, independence,
deduplication, temporal metadata, exclusion, and reproducibility. Every
discovered candidate must be recorded in `provenance/candidates.jsonl` using
`schemas/collection_candidate.schema.json`, including candidates that are
excluded. This candidate ledger is a pre-normalization collection-decision
record, not a replacement or competing model for `provenance/sources.jsonl`.

A retained candidate must be joined to exactly one normalized document and the
existing provenance record through its `retained_document_id` and `source_id`.
Its collected-content checksum must agree with the provenance checksum. The
ledger records a documented source identity and independence relationship;
different URLs must not be treated as independent evidence merely because their
hosts differ. Exact and near duplicate decisions remain human-reviewed and are
represented in the ledger; no fuzzy deduplication automatically excludes or
retains material.
