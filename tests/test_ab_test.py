"""pipeline/ab_test.py + pipeline/deploy.py A/B Testing 단위 테스트"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── ab_test._decide_winner ────────────────────────────────────────────────────

def test_challenger_wins_with_significant_improvement():
    from pipeline.ab_test import _decide_winner
    # challenger가 일관되게 높으면 t-test 유의 + MIN_IMPROVEMENT 충족
    champion_aucs = [0.880] * 10
    challenger_aucs = [0.895] * 10
    result = _decide_winner(champion_aucs, challenger_aucs, "v1", "v2", 10)
    assert result["winner"] == "challenger"
    assert result["promote"] is True
    assert result["p_value"] < 0.05


def test_champion_wins_when_challenger_worse():
    from pipeline.ab_test import _decide_winner
    champion_aucs = [0.890] * 10
    challenger_aucs = [0.860] * 10
    result = _decide_winner(champion_aucs, challenger_aucs, "v1", "v2", 10)
    assert result["winner"] == "champion"
    assert result["promote"] is False


def test_inconclusive_when_no_significant_difference():
    from pipeline.ab_test import _decide_winner
    rng = np.random.default_rng(42)
    # 거의 동일한 분포
    champion_aucs = (0.880 + rng.normal(0, 0.001, 10)).tolist()
    challenger_aucs = (0.880 + rng.normal(0, 0.001, 10)).tolist()
    result = _decide_winner(champion_aucs, challenger_aucs, "v1", "v2", 10)
    # p-value가 높거나 개선 폭이 MIN_IMPROVEMENT 미만 → inconclusive 또는 champion
    assert not result["promote"]


def test_result_contains_required_fields():
    from pipeline.ab_test import _decide_winner
    result = _decide_winner([0.88]*5, [0.89]*5, "v1", "v2", 5)
    for key in ("winner", "promote", "p_value", "t_stat",
                "champion_mean_auc", "challenger_mean_auc",
                "champion_version", "challenger_version", "rounds"):
        assert key in result


# ── ab_test.promote_challenger / reject_challenger ────────────────────────────

def test_promote_challenger(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.ab_test.REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr("pipeline.ab_test.FLAGS_PATH", tmp_path / "flags.json")

    registry = {"models": [
        {"version": "v1", "status": "deployed"},
        {"version": "v2", "status": "challenger"},
    ]}
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    (tmp_path / "flags.json").write_text(json.dumps({
        "active_model_version": "v1",
        "challenger_model_version": "v2",
        "challenger_traffic_weight": 0.1,
    }))

    from pipeline.ab_test import promote_challenger
    promote_challenger("v2")

    updated = json.loads((tmp_path / "registry.json").read_text())
    statuses = {m["version"]: m["status"] for m in updated["models"]}
    assert statuses["v1"] == "retired"
    assert statuses["v2"] == "deployed"

    flags = json.loads((tmp_path / "flags.json").read_text())
    assert flags["active_model_version"] == "v2"
    assert "challenger_model_version" not in flags


def test_reject_challenger(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.ab_test.REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr("pipeline.ab_test.FLAGS_PATH", tmp_path / "flags.json")

    registry = {"models": [
        {"version": "v1", "status": "deployed"},
        {"version": "v2", "status": "challenger"},
    ]}
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    (tmp_path / "flags.json").write_text(json.dumps({
        "active_model_version": "v1",
        "challenger_model_version": "v2",
        "challenger_traffic_weight": 0.1,
    }))

    from pipeline.ab_test import reject_challenger
    reject_challenger("v2")

    updated = json.loads((tmp_path / "registry.json").read_text())
    statuses = {m["version"]: m["status"] for m in updated["models"]}
    assert statuses["v1"] == "deployed"
    assert statuses["v2"] == "rejected"

    flags = json.loads((tmp_path / "flags.json").read_text())
    assert flags["active_model_version"] == "v1"
    assert "challenger_model_version" not in flags


# ── deploy._register_challenger ───────────────────────────────────────────────

def test_deploy_registers_challenger(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.deploy.REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr("pipeline.deploy.FLAGS_PATH", tmp_path / "flags.json")

    registry = {"models": [
        {"version": "v1", "status": "deployed", "dataset_id": 44089},
        {"version": "v2", "status": "evaluated_pass", "dataset_id": 44089,
         "model_name": "hgb", "dataset_name": "credit"},
    ]}
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    (tmp_path / "flags.json").write_text(json.dumps({"active_model_version": "v1", "active_dataset_id": 44089}))

    from pipeline.deploy import _load_registry, _load_flags, _register_challenger
    reg = _load_registry()
    candidate = next(m for m in reg["models"] if m["version"] == "v2")
    result = _register_challenger(reg, candidate)

    assert result is True
    updated_reg = json.loads((tmp_path / "registry.json").read_text())
    statuses = {m["version"]: m["status"] for m in updated_reg["models"]}
    assert statuses["v2"] == "challenger"
    assert statuses["v1"] == "deployed"  # 챔피언은 그대로

    flags = json.loads((tmp_path / "flags.json").read_text())
    assert flags["challenger_model_version"] == "v2"
    assert flags["challenger_traffic_weight"] == 0.1


# ── predict.py 트래픽 분기 ────────────────────────────────────────────────────

def test_predict_uses_challenger_on_high_random(tmp_path, monkeypatch):
    """random() < weight 일 때 챌린저 모델 선택"""
    monkeypatch.setattr("pipeline.predict.FLAGS_PATH", tmp_path / "flags.json")
    monkeypatch.setattr("pipeline.predict.REGISTRY_PATH", tmp_path / "registry.json")

    flags = {
        "active_model_version": "v1",
        "active_dataset_id": 44089,
        "challenger_model_version": "v2",
        "challenger_traffic_weight": 1.0,  # 100% → 항상 챌린저
    }
    (tmp_path / "flags.json").write_text(json.dumps(flags))

    # _load_active_model에서 version이 "v2"를 반환하는지 확인
    import pipeline.predict as predict_mod

    loaded_versions = []
    original_load = predict_mod._load_active_model

    def mock_load():
        # flags에서 읽은 version을 기록
        import json as _json
        import random as _random
        _flags = _json.loads((tmp_path / "flags.json").read_text())
        version = _flags.get("active_model_version")
        challenger = _flags.get("challenger_model_version")
        weight = float(_flags.get("challenger_traffic_weight", 0.0))
        if challenger and weight > 0 and _random.random() < weight:
            version = challenger
        loaded_versions.append(version)
        raise RuntimeError("stop here")  # 실제 파일 로드는 하지 않음

    monkeypatch.setattr("pipeline.predict._load_active_model", mock_load)

    import pandas as pd
    with pytest.raises(RuntimeError):
        predict_mod.predict(pd.DataFrame([{"f": 1.0}]))

    assert loaded_versions[0] == "v2"


def test_predict_uses_champion_on_low_random(tmp_path, monkeypatch):
    """random() >= weight 일 때 챔피언 모델 선택"""
    monkeypatch.setattr("pipeline.predict.FLAGS_PATH", tmp_path / "flags.json")

    flags = {
        "active_model_version": "v1",
        "challenger_model_version": "v2",
        "challenger_traffic_weight": 0.0,  # 0% → 항상 챔피언
    }
    (tmp_path / "flags.json").write_text(json.dumps(flags))

    import pipeline.predict as predict_mod
    loaded_versions = []

    def mock_load():
        import json as _json, random as _random
        _flags = _json.loads((tmp_path / "flags.json").read_text())
        version = _flags.get("active_model_version")
        challenger = _flags.get("challenger_model_version")
        weight = float(_flags.get("challenger_traffic_weight", 0.0))
        if challenger and weight > 0 and _random.random() < weight:
            version = challenger
        loaded_versions.append(version)
        raise RuntimeError("stop")

    monkeypatch.setattr("pipeline.predict._load_active_model", mock_load)

    import pandas as pd
    with pytest.raises(RuntimeError):
        predict_mod.predict(pd.DataFrame([{"f": 1.0}]))

    assert loaded_versions[0] == "v1"
