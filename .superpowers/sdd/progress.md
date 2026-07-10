# SDD Progress — daily_v5 preprocessing overhaul

Plan: docs/superpowers/plans/2026-07-10-daily-v5-preprocessing.md
Branch: feat/sharpe-above-1
Implementer agent: superpowers-implementer-mercury
Reviewer agent: superpowers-reviewer-mercury

## Tasks
Task 1: complete (commits 79fe62d..c1c048a, review clean)
Task 2: complete (commits c1c048a..555a66a, review clean; dropped vol_z_20 -> FEATURE_COLS now 29, not 30)
Task 3: complete (commits 555a66a..835d9cc, review clean after 1 fix loop — vacuous-test hardening)
Task 4: complete (commits 835d9cc..694160e, review clean)
Task 5: complete (commits 694160e..6c7018d, review clean; 1 minor: redundant in-test import)
Task 6: complete (commits 6c7018d..b8ad02a, review clean; minors logged)
Task 7: complete (commits b8ad02a..6654a7a, review clean; minors logged)
Task 8: complete (commits 6654a7a..c4f083f, review clean after 1 fix loop — dead code deletion)
Task 9: complete (commits c4f083f..dfd56c0, review clean)
Task 10: complete (commits dfd56c0..6e2094b, review clean; minors: redundant _preprocess in ml_strategies fallback line ~250, non-discriminative test)
Task 11: complete (commits 6e2094b..54bbed8, review clean after 1 fix loop — copy + discriminative test)
(pending) Task 12: full suite, retrain, re-tune, recommit models

## Minor findings (for final review triage)
- Task 2 (test_features.py:176): guard test uses `spy = df.copy()`, so ret_*_vs_spy cols are zero-variance and filtered out (not correlation-tested). Brief comment wrong. Could use independent random SPY to actually cover those cols.
- Task 2 / Task 5 / Task 6: redundant in-function/in-test imports (brief-prescribed, harmless).
- Task 6 (predictors/logistic.py:15): docstring still says "_preprocess pass" — now uses _scale. Stale; fix in final cleanup wave.
- Task 6 (test_predictors.py test_ridge_predictor_uses_scale): not discriminative (finite under old path too); consider an abs(scores) bound if a real clip test is wanted.
- Task 7 (test_predict.py:902-903): stale docstring "_preprocess must clip" — no longer clips. Fix in final wave.
- Task 7 (predict_next_day_lite.py:203): stale "±5-std-clip" comment — Task 10 edits this file; ensure fixed there.
- Task 7 (test_model_accuracy.py:37): test name says is_robust_and_clipped but only asserts RobustScaler type.
## Contract note
- FEATURE_COLS is 29 after Task 2 (design spec still says 30 — reconcile spec doc at end).
