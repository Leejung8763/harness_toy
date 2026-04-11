"""feature_store/store.py + pipeline/data_engineering.py 단위 테스트"""

import json

import numpy as np
import pandas as pd
import pytest

from data.loader import DatasetSplit
from feature_store.store import _store_path, exists, list_features, load, save


def _make_split(dataset_id: int = 999) -> DatasetSplit:
    """테스트용 DatasetSplit을 생성합니다."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}).astype("float32")
    y = pd.Series([0, 1, 0], dtype="int32", name="target")
    return DatasetSplit(
        dataset_id=dataset_id,
        name="test_dataset",
        X_train=X,
        X_test=X.copy(),
        y_train=y,
        y_test=y.copy(),
        feature_names=["a", "b"],
        n_classes=2,
    )


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    """저장 후 로드하면 원본과 동일한 DatasetSplit이 반환되어야 한다."""
    monkeypatch.setattr("feature_store.store.STORE_DIR", tmp_path)
    split = _make_split()

    save(split, version="v1")
    loaded = load(999, version="v1")

    assert loaded.dataset_id == split.dataset_id
    assert loaded.name == split.name
    assert loaded.feature_names == split.feature_names
    assert loaded.n_classes == split.n_classes
    pd.testing.assert_frame_equal(loaded.X_train, split.X_train)
    pd.testing.assert_series_equal(loaded.y_train.reset_index(drop=True),
                                   split.y_train.reset_index(drop=True))


def test_exists_false_before_save(tmp_path, monkeypatch):
    """저장 전에는 exists가 False를 반환해야 한다."""
    monkeypatch.setattr("feature_store.store.STORE_DIR", tmp_path)
    assert exists(999, "v1") is False


def test_exists_true_after_save(tmp_path, monkeypatch):
    """저장 후에는 exists가 True를 반환해야 한다."""
    monkeypatch.setattr("feature_store.store.STORE_DIR", tmp_path)
    save(_make_split(), version="v1")
    assert exists(999, "v1") is True


def test_exists_version_specific(tmp_path, monkeypatch):
    """다른 버전은 exists가 False를 반환해야 한다."""
    monkeypatch.setattr("feature_store.store.STORE_DIR", tmp_path)
    save(_make_split(), version="v1")
    assert exists(999, "v2") is False


def test_load_raises_when_not_found(tmp_path, monkeypatch):
    """Feature Store에 없는 데이터 로드 시 FileNotFoundError가 발생해야 한다."""
    monkeypatch.setattr("feature_store.store.STORE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="data_engineering"):
        load(999, "v1")


def test_list_features_empty(tmp_path, monkeypatch):
    """저장된 피처가 없으면 빈 리스트를 반환해야 한다."""
    monkeypatch.setattr("feature_store.store.STORE_DIR", tmp_path)
    assert list_features() == []


def test_list_features_returns_metadata(tmp_path, monkeypatch):
    """저장된 피처가 있으면 메타데이터 목록을 반환해야 한다."""
    monkeypatch.setattr("feature_store.store.STORE_DIR", tmp_path)
    save(_make_split(dataset_id=1), version="v1")
    save(_make_split(dataset_id=2), version="v1")
    result = list_features()
    assert len(result) == 2
    assert all("dataset_id" in r for r in result)
