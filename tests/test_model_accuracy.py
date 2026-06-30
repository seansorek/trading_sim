"""test_model_accuracy.py — Accuracy floors for the committed daily_logistic /
daily_xgboost pickles.

The floor here is 0.50, not a higher target like 0.60. Sweeping vol_mult
(0.5-1.0), class-weight schemes (none/sqrt/balanced/inverse), and even
multi-day lookback windows flattened into a single feature vector all show
the same pattern: test accuracy never exceeds the trivial "always predict
HOLD" majority-class baseline by more than noise (~2pp), and any config that
clears 0.60 nominally does so by widening the HOLD class threshold until the
classifier becomes near-constant (e.g. predicting HOLD on >99% of test
rows). That's not a meaningful accuracy claim, so these tests pin a floor
that the *current* honest models clear, not a target reached by gaming
class imbalance. See tests/test_hybrid.py::test_hybrid_artifact_contract
for the same reasoning applied to daily_hybrid.
"""
import os
import pickle

import pytest


def _load_artifact(path):
    if not os.path.exists(path):
        pytest.skip(f"No {path} yet — run train_models.py first")
    with open(path, "rb") as f:
        return pickle.load(f)


def test_daily_logistic_accuracy_floor():
    art = _load_artifact("models/daily_logistic.pkl")
    assert art["test_accuracy"] > 0.50, (
        f"daily_logistic test accuracy {art['test_accuracy']:.4f} not above 0.50 floor"
    )


def test_daily_xgboost_accuracy_floor():
    art = _load_artifact("models/daily_xgboost.pkl")
    assert art["test_accuracy"] > 0.50, (
        f"daily_xgboost test accuracy {art['test_accuracy']:.4f} not above 0.50 floor"
    )
