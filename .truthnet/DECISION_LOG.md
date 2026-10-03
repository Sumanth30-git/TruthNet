# TruthNet V2 Decision Log

## Active project is TruthNet V2

- **Decision:** TruthNet V2 is the active project and source of truth.
- **Rationale:** The current repository, APIs, pipelines, and tests are maintained in V2.
- **Current status:** Active.
- **Reversible:** No for normal development scope; any project migration requires explicit human direction.

## V1 remains untouched

- **Decision:** Preserve V1 and the old backup; development targets only TruthNet V2.
- **Rationale:** Protect historical work and prevent accidental divergence or overwrite.
- **Current status:** Active boundary.
- **Reversible:** Only by explicit human-approved migration work.

## News is the primary advanced branch

- **Decision:** News verification is the primary advanced branch before Image or Video expansion.
- **Rationale:** The current advanced evidence, search, NLI, matching, aggregation, and decision work is concentrated in News.
- **Current status:** Active project direction.
- **Reversible:** Yes, with human approval and revised priorities.

## Tavily is the current search provider

- **Decision:** Tavily is the implemented current-web search provider.
- **Rationale:** `SearchProvider` is provider-independent, and the concrete configured adapter is Tavily.
- **Current status:** Implemented but disabled by default unless explicitly configured.
- **Reversible:** Yes; the provider boundary permits an evaluated replacement.

## NLI model selection

- **Decision:** NLI uses `cross-encoder/nli-deberta-v3-base`.
- **Rationale:** The current adapter and fixed three-way label mapping are built around this model.
- **Current status:** Implemented as an opt-in lazy adapter.
- **Reversible:** Yes, but only with explicit mapping, evaluation, and contract review.

## Evidence aggregation precedes verdicts

- **Decision:** Claim-level evidence aggregation occurs before the final News verdict policy.
- **Rationale:** The policy consumes evidence summaries, retains support and contradiction separately, and requires a summary for every claim.
- **Current status:** Implemented in the production News pipeline when search completes.
- **Reversible:** Yes, with approved policy redesign and regression coverage.

## BERT is not a truth authority

- **Decision:** The News BERT baseline must not determine factual truth or the final News verdict.
- **Rationale:** It is an uncalibrated text-classifier signal; `decide_news_verdict()` deliberately ignores model signals.
- **Current status:** Implemented and tested.
- **Reversible:** Only with explicit evaluation, calibration, policy approval, and contract review.

## Inconclusive is intentional

- **Decision:** `inconclusive` is a valid final outcome.
- **Rationale:** Insufficient, unavailable, partial, weak, or conflicting evidence must not be forced into an authentic/fake answer.
- **Current status:** Implemented for News and Image.
- **Reversible:** No as a safety principle; decision thresholds may be revised only with human approval.

## Claim understanding is deterministic and a redesign target

- **Decision:** Current claim understanding is deterministic; production sentence segmentation and richer claim-understanding extraction are recognized as a redesign point.
- **Rationale:** The production News pipeline currently segments sentences, while `claim_understanding.py` contains richer deterministic clause/context parsing that is not the production segmentation source.
- **Current status:** Active known design boundary.
- **Reversible:** Yes, through an approved NLP Claim Understanding redesign with compatibility planning and tests.

## Heavy research may use Google Colab

- **Decision:** Heavy NLP/model research may use Google Colab.
- **Rationale:** Research and evaluation workloads can require resources beyond normal local development.
- **Current status:** Available research practice; not a TruthNet runtime dependency.
- **Reversible:** Yes.

## Laptop limits do not define model quality

- **Decision:** Laptop limitations must not dictate model-quality decisions.
- **Rationale:** Model selection and evaluation should be driven by evidence and suitability, with external research compute available when necessary.
- **Current status:** Active development principle.
- **Reversible:** Yes, with human direction.

## Current Git checkpoint

- **Decision:** Use `0c30910` — `TruthNet V2 Phase 3F integrate news verdict policy` — as the current audited Git checkpoint.
- **Rationale:** It is the repository HEAD at the time the development-control layer was created.
- **Current status:** Current baseline checkpoint.
- **Checkpoint meaning:** Historical baseline metadata only; it is not a requirement that `HEAD` remain at this commit.
- **Reversible:** Yes; future human-approved commits will supersede it.
