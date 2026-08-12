from __future__ import annotations

from pathlib import Path

import yaml


required = {
    "data": ["dataset", "data_path", "adjacency_path", "community_path", "history", "split"],
    "model": ["communities", "top_k", "d_model", "d_gcn", "k_high", "k_low"],
    "training": ["epochs", "batch_size", "learning_rate", "lambda_spec", "lambda_sep", "lambda_orth"]
}

for path in sorted(Path("configs").glob("*.yaml")):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section, keys in required.items():
        assert section in config, f"{path}: missing {section}"
        for key in keys:
            assert key in config[section], f"{path}: missing {section}.{key}"
    assert abs(sum(config["data"]["split"]) - 1) < 1e-9
    assert config["data"]["horizon"] == 1, f"{path}: executable path is one-step"
    print(f"valid: {path}")
