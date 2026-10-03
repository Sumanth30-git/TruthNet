# TruthNet V2 Project Rules

## Source of truth and scope

- `TruthNet-V2` is the active project and source of truth.
- The old `TruthNet-V2 - Copy` backup must never be inspected, modified, renamed, deleted, committed, or used as an implementation target.
- When repository artifacts conflict, current production code, schemas, and tests govern implemented behavior.
- Documentation is subordinate to current production code, schemas, and tests when it conflicts with them; conflicting or stale documentation must be corrected through approved work.
- Preserve existing API contracts unless a contract change is explicitly designed, reviewed, approved, and tested.
- Do not make unrelated refactors during focused feature work.

## Safety and product semantics

- Preserve the existing structured error and request-ID semantics.
- `inconclusive` is a valid, intentional result. Never force an authentic or fake verdict when evidence is insufficient, unavailable, conflicting, or incomplete.
- The BERT fake-news classifier is an auxiliary text signal. It is not the final authority for News fact verification and must not silently influence the final News verdict policy.
- Keep source authority, claim-source relevance, textual stance, freshness, source independence, and factual truth as separate concepts.
- Do not turn a source identity, NLI output, raw score, or a single source into proof of truth.
- Do not hard-code individual examples, entities, headlines, or expected answers as a substitute for a general NLP solution.

## Models, data, and external services

- Keep inference opt-in through the existing model-inference safety gate unless an approved design changes that contract.
- Do not silently change model IDs, label mappings, score meaning, model versions, or calibration semantics.
- New NLP/model components require defined evaluation and verification before production integration.
- Never expose API keys, credentials, private request data, or provider diagnostics in code, logs, responses, tests, or documentation.
- Treat source-provider results as candidates, not findings.

## Quality and delivery

- Behavior changes require focused tests; contract changes require regression coverage.
- Do not remove or weaken existing tests merely to make a change pass.
- Do not claim a feature is complete without appropriate verification.
- Inspect the resulting diff and Git status before handoff.
- Do not commit or push automatically. A human decides whether and when to create a Git checkpoint.
