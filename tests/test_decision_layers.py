"""Tests for decision layer implementations."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from decision_layers.base import BaseDecisionLayer, DecisionContext


class TestBaseDecisionLayer:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseDecisionLayer()

    def test_decision_context_stores_index_and_symbol(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        ctx = DecisionContext(index=idx, symbol="AAPL")
        assert ctx.symbol == "AAPL"
        assert len(ctx.index) == 5
