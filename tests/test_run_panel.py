import numpy as np
import pandas as pd

from panel_backtester import PanelConfig
from run_panel import run_grid


class _FakePanel:
    def __init__(self, pred, ret):
        self.pred = pred
        self.ret = ret
        self.close = pred
        self.symbols = list(pred.columns)
        self.dropped = {}


def test_run_grid_runs_every_config_and_returns_a_verdict():
    idx = pd.bdate_range("2020-01-01", periods=400)
    cols = [f"S{i}" for i in range(30)]
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(rng.normal(0, 0.02, (len(idx), 30)), index=idx, columns=cols)
    pred = pd.DataFrame(rng.normal(0, 1, (len(idx), 30)), index=idx, columns=cols)
    spy_ret = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)

    out = run_grid(_FakePanel(pred, ret), PanelConfig(min_names=5), spy_ret)

    assert len(out["per_config"]) == 4
    assert "verdict" in out
    assert isinstance(out["passed"], bool)
