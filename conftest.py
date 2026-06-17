"""Root conftest.py — install torch mock if torch is unavailable.

This runs before pytest collects test modules, so any test file that
imports simulation_pipeline (which imports torch at module level)
will find the mock already in sys.modules.
"""
import sys
import types
from unittest.mock import MagicMock


def _install_torch_mock():
    try:
        import torch  # noqa: F401
        return
    except (ImportError, OSError):
        pass

    mock_torch = types.ModuleType("torch")

    mock_nn = types.ModuleType("torch.nn")
    mock_nn.Module = type("Module", (), {"__init__": lambda self, *a, **kw: None})
    mock_nn.Linear = MagicMock()
    mock_nn.ReLU = MagicMock()
    mock_nn.Sequential = MagicMock()
    mock_nn.functional = MagicMock()

    mock_optim = types.ModuleType("torch.optim")
    mock_optim.Adam = MagicMock()

    mock_torch.nn = mock_nn
    mock_torch.optim = mock_optim
    mock_torch.Tensor = MagicMock
    mock_torch.FloatTensor = MagicMock()
    mock_torch.no_grad = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(), __exit__=MagicMock()
    ))
    mock_torch.tensor = MagicMock()
    mock_torch.load = MagicMock()
    mock_torch.save = MagicMock()
    mock_torch.device = MagicMock()
    mock_torch.float32 = "float32"
    mock_torch.zeros = MagicMock()
    mock_torch.from_numpy = MagicMock()

    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = mock_nn
    sys.modules["torch.nn.functional"] = mock_nn.functional
    sys.modules["torch.optim"] = mock_optim


_install_torch_mock()
