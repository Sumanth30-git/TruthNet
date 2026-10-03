# TruthNet V2 Development Workflow

This workflow is inspired by the disciplined specification, test-first, review, and verification principles associated with obra/superpowers. It is TruthNet's own process; Superpowers is not part of the runtime and must not be added as an application dependency.

## Phase 1 — Understand

- Inspect the current code and tests before changing behavior.
- Identify affected public contracts, including schemas, endpoints, configuration, model mappings, and response semantics.
- Identify safety, regression, privacy, model, and operational risks.
- Establish what is implemented, what is a stub, and what the production path actually uses.

## Phase 2 — Brainstorm and specification

- Define the problem in observable terms.
- Record feasible alternatives and their trade-offs.
- State constraints, non-goals, safety requirements, and acceptance criteria.
- Obtain human approval before major architecture, model, evidence-policy, or API-contract changes.
- Codex, Cursor, Antigravity, and OpenClaw may act only within the user-approved task and file scope. They must stop and obtain human approval before materially expanding that scope.
- Agents must not independently change architecture, model selection or policy, or verdict policy; must stop before external coordination or other external state changes; and must not commit or push unless explicitly authorized.
- Destructive actions require explicit authorization. Agents must preserve unrelated pre-existing worktree changes.

## Phase 3 — Plan

- Break approved work into small, reviewable tasks.
- Identify the exact files allowed to change.
- Identify affected and new tests.
- Define the verification command(s), expected behavior, and any required manual/API/model smoke tests.

## Phase 4 — Test-first implementation

- RED: add or adjust a focused test that demonstrates the intended behavior.
- GREEN: write the smallest implementation that makes the test pass.
- REFACTOR: improve clarity only after behavior is protected.
- Avoid speculative abstractions, unrelated cleanup, and unnecessary code.

## Phase 5 — Review

Review the resulting work for:

- specification compliance;
- architecture and contract compliance;
- regression risk;
- safety, security, and credential handling;
- test quality, including meaningful failure coverage;
- unsupported completeness or accuracy claims.

## Phase 6 — Verification

Run verification appropriate to the change:

- `python -m pytest -q`;
- focused tests for the touched behavior;
- integration tests where layers interact;
- manual API and model smoke tests when applicable and explicitly safe to run;
- `git diff` and `git status --short` inspection.

When verification is intended to leave a clean worktree, use no-cache/no-bytecode test modes. Obtain explicit human authorization before networked smoke tests, model smoke tests that may download weights, or use of non-public or sensitive input data.

Do not report completion when required verification has failed, been skipped, or is inapplicable without being disclosed.

## Phase 7 — Checkpoint

- Reach this phase only after verification.
- Present the changed files, behavior, verification results, and remaining risks.
- A human decides whether to commit or push. Agents must not do either automatically.

## Roles

### Human

- Final architectural authority.
- Approves major design, API-contract, model, and evidence-policy decisions.

### Codex

- Primary major engineering implementation agent.
- Applies the workflow, maintains scoped changes, verifies work, and reports evidence.
- Does not independently expand scope or make reserved human decisions.

### Cursor

- Focused/local edit and debugging agent.
- Works within approved scope and preserves shared contracts.
- Does not independently expand scope or make reserved human decisions.

### Antigravity

- Optional independent agent for larger workflow, review, or research tasks.
- Provides findings without becoming an unreviewed production authority.
- Does not independently expand scope or make reserved human decisions.

### OpenClaw

- Optional automation/orchestration agent.
- Must not silently alter production code.
- Does not independently expand scope or make reserved human decisions.

### Superpowers

- Methodology inspiration for this workflow only.
- Not part of TruthNet runtime; do not add it as an application dependency.
