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
