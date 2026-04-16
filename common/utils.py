import os
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for python random, numpy, and torch (CPU + MPS if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Return torch.device for the requested backend. 'auto' → mps > cpu."""
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "mps":
        return torch.device("mps")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_output_dir(root, task: str) -> Path:
    """Create outputs/<task>/<timestamp>/ and return its Path."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(root) / task / ts
    out.mkdir(parents=True, exist_ok=False)
    return out
