# TruthNet NLI evaluation (Stage 14.1)

This directory is **offline NLI textual-relation evaluation**. It does not measure whether a news claim is true, and it is not a Stage 13 production path.

Gold labels are `entailment`, `contradiction`, or `neutral` between a source-text **premise** and a claim-text **hypothesis**. `unresolved` and `abstained` are annotation states, not extra NLI classes.

## Canonical evaluator inputs

Use only these files with `scripts/evaluate_nli_relations.py`:

| File | Role |
| --- | --- |
| `pilot/batch2_evaluation_gold.jsonl` | Canonical Batch 2 gold (38 labeled, 1 abstained, 1 unresolved) |
| `pilot/batch2_predictions.jsonl` | Stored predictions for the 38 labeled Batch 2 rows |
| `pilot/batch1.jsonl` | Schema-valid smoke gold (10 labeled; **no contradiction class**) |
| `pilot/batch1_predictions.jsonl` | Stored predictions for Batch 1 |

Do **not** pass these as `--input` (keep them on disk; do not overwrite):

| File | Why |
| --- | --- |
| `pilot/batch2.jsonl` | Unannotated template; evaluator rejects `unannotated` |
| `pilot/batch2_annotator_a.jsonl` / `batch2_annotator_b.jsonl` | Label fragments, not the contract |
| `pilot/batch2_adjudicated.jsonl` | Invalid provenance (`annotated`; abstain/unresolved stored in `gold_relation`) |
| `pilot/batch2_prediction_input.jsonl` | Extra provenance fields; schema-invalid |

Contract: `contracts/nli_relation.schema.json`. Protocol: `pilot/ANNOTATION_GUIDE.md`.

## Annotator A/B identity

The Batch 2 annotator files contain the **same 40 labels in the same order**. After ignoring the annotator name, the records are identical. Canonical gold uses `annotator_id` `annotator_a+annotator_b`.

The integrity audit reports this and computes raw agreement plus Cohen’s kappa **on the files as they are**. Kappa is 1.0 because the labels match. **That does not prove independent double annotation.** Stage 14.1 does not invent a second annotation stream or change gold labels.

## Splits

Batch 2 has **10 stories**, one source/article/URL per story, and four pairs per story. Grouping by `story_or_event_cluster_id` would also keep source/article together **in this corpus**.

**10 groups is too small for train/dev/test.** `scripts/assign_nli_evaluation_splits.py` groups by story, never randomly splits rows, and **refuses to write** `development.jsonl` / `test.jsonl` / `frozen_test.jsonl` unless there are at least 30 story groups **and** all three labeled classes in every requested split. For Batch 2 it emits `status: insufficient_groups`.

`duplicate_or_syndication_group_id` is unique per Batch 2 row and does **not** encode copies. Splits (when a larger corpus qualifies) still key off story and fail if the same `source_id`, `article_id`, `canonical_url`, or duplicate-group id would land in two splits.

## Commands

From the repository root. `--output -` prints JSON to stdout and does not write a report file. Do not overwrite existing gold or prediction files.

```powershell
python -m pytest -q tests/test_nli_evaluation_integrity.py tests/test_nli_evaluation_splits.py tests/test_nli_relation_evaluation.py tests/test_nli_relation_evaluation_contract.py

python scripts/audit_nli_evaluation.py --gold data/news_evaluation/pilot/batch2_evaluation_gold.jsonl --annotator-a data/news_evaluation/pilot/batch2_annotator_a.jsonl --annotator-b data/news_evaluation/pilot/batch2_annotator_b.jsonl --predictions data/news_evaluation/pilot/batch2_predictions.jsonl --output -

python scripts/assign_nli_evaluation_splits.py --gold data/news_evaluation/pilot/batch2_evaluation_gold.jsonl --output -

python scripts/evaluate_nli_relations.py --input data/news_evaluation/pilot/batch2_evaluation_gold.jsonl --predictions data/news_evaluation/pilot/batch2_predictions.jsonl --output -
```

Optional Batch 1 smoke scoring (not a three-class validation set):

```powershell
python scripts/evaluate_nli_relations.py --input data/news_evaluation/pilot/batch1.jsonl --predictions data/news_evaluation/pilot/batch1_predictions.jsonl --output -
```

## Limitations

- NLI relation accuracy is **not** news-verdict accuracy.
- Batch 2 pairs from the same article are not independent tests.
- Batch 1 has no contradiction class; 100% accuracy there is uninformative.
- Stored predictions do not pin model revision; fresh inference is a separate, optional step and must write a **new** path.
- Do not train or tune on Batch 2 and then treat the same 38 rows as a frozen test set.
