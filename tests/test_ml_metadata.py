"""ml_metadata/store.py 단위 테스트"""

import pytest

import ml_metadata.store as mds


def test_log_and_get_data_engineering(tmp_path, monkeypatch):
    """Data Engineering 실행을 기록하고 조회할 수 있어야 한다."""
    monkeypatch.setattr("ml_metadata.store.METADATA_PATH", tmp_path / "runs.json")
    mds.log_data_engineering({"dataset_id": 1, "n_features_raw": 10})
    runs = mds.get_runs("data_engineering")
    assert len(runs) == 1
    assert runs[0]["stage"] == "data_engineering"
    assert runs[0]["dataset_id"] == 1


def test_log_and_get_training(tmp_path, monkeypatch):
    """Training 실행을 기록하고 조회할 수 있어야 한다."""
    monkeypatch.setattr("ml_metadata.store.METADATA_PATH", tmp_path / "runs.json")
    mds.log_training({"model_version": "v1", "roc_auc": 0.85})
    runs = mds.get_runs("training")
    assert len(runs) == 1
    assert runs[0]["roc_auc"] == 0.85


def test_get_runs_all_stages(tmp_path, monkeypatch):
    """stage=None이면 전체 실행 기록을 반환해야 한다."""
    monkeypatch.setattr("ml_metadata.store.METADATA_PATH", tmp_path / "runs.json")
    mds.log_data_engineering({"dataset_id": 1})
    mds.log_training({"model_version": "v1"})
    runs = mds.get_runs()
    assert len(runs) == 2


def test_get_runs_filter_by_stage(tmp_path, monkeypatch):
    """stage 필터가 정확히 동작해야 한다."""
    monkeypatch.setattr("ml_metadata.store.METADATA_PATH", tmp_path / "runs.json")
    mds.log_data_engineering({"dataset_id": 1})
    mds.log_training({"model_version": "v1"})
    assert len(mds.get_runs("data_engineering")) == 1
    assert len(mds.get_runs("training")) == 1


def test_get_latest_run(tmp_path, monkeypatch):
    """가장 최근 실행을 반환해야 한다."""
    monkeypatch.setattr("ml_metadata.store.METADATA_PATH", tmp_path / "runs.json")
    mds.log_training({"model_version": "v1", "roc_auc": 0.80})
    mds.log_training({"model_version": "v2", "roc_auc": 0.85})
    latest = mds.get_latest_run("training")
    assert latest["model_version"] == "v2"


def test_get_latest_run_empty(tmp_path, monkeypatch):
    """실행 기록이 없으면 None을 반환해야 한다."""
    monkeypatch.setattr("ml_metadata.store.METADATA_PATH", tmp_path / "runs.json")
    assert mds.get_latest_run("training") is None


def test_runs_persist_across_calls(tmp_path, monkeypatch):
    """여러 번 기록해도 누적되어야 한다."""
    monkeypatch.setattr("ml_metadata.store.METADATA_PATH", tmp_path / "runs.json")
    mds.log_training({"model_version": "v1"})
    mds.log_training({"model_version": "v2"})
    mds.log_training({"model_version": "v3"})
    assert len(mds.get_runs("training")) == 3
