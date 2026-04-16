import pytest
from common.config import load_config, apply_overrides


def test_load_config_parses_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "data:\n"
        "  batch_size: 32\n"
        "  image_size: 224\n"
        "train:\n"
        "  lr: 0.0001\n"
        "  epochs: 30\n"
        "seed: 42\n"
    )
    cfg = load_config(str(p))
    assert cfg["data"]["batch_size"] == 32
    assert cfg["train"]["lr"] == 0.0001
    assert cfg["seed"] == 42


def test_apply_overrides_dotted_path():
    cfg = {"train": {"lr": 0.0001, "epochs": 30}, "seed": 42}
    out = apply_overrides(cfg, ["train.lr=0.0005", "seed=7"])
    assert out["train"]["lr"] == 0.0005
    assert out["seed"] == 7
    assert out["train"]["epochs"] == 30


def test_apply_overrides_type_coercion():
    cfg = {"train": {"epochs": 30, "use_sampler": True}}
    out = apply_overrides(cfg, ["train.epochs=50", "train.use_sampler=false"])
    assert out["train"]["epochs"] == 50
    assert out["train"]["use_sampler"] is False


def test_apply_overrides_unknown_key_raises():
    cfg = {"train": {"lr": 0.0001}}
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["train.nonexistent=1"])
