import random
import re
import numpy as np
import torch
from common.utils import set_seed, get_device, make_output_dir


def test_set_seed_python_numpy_torch():
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_get_device_is_mps_or_cpu():
    dev = get_device()
    assert dev.type in ("mps", "cpu")


def test_get_device_forced_cpu():
    dev = get_device("cpu")
    assert dev.type == "cpu"


def test_make_output_dir_creates_timestamped_dir(tmp_path):
    out = make_output_dir(root=tmp_path, task="classification")
    assert out.exists() and out.is_dir()
    assert out.parent.name == "classification"
    assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", out.name)
