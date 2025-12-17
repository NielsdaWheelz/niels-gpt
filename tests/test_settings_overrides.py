import json

from niels_gpt.settings import resolve_settings


def test_resolve_settings_override_applied(tmp_path):
    overrides = {
        "model": {"V": 128},
        "training": {
            "pretrain": {
                "base_lr": 1e-5,
                "micro_B": 2,
            }
        },
    }
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(overrides), encoding="utf-8")

    resolved = resolve_settings(phase="pretrain", overrides_path=str(path))

    assert resolved.train_cfg.base_lr == 1e-5
    assert resolved.train_cfg.B == 2
    assert resolved.model_cfg.V == 128

