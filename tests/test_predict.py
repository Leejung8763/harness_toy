"""pipeline/predict.py — 입력 검증 및 모델 정보 조회 단위 테스트"""

import numpy as np
import pandas as pd
import pytest

from pipeline.predict import _validate_input, get_active_model_info


def test_validate_input_converts_to_float32():
    """입력 DataFrame을 float32로 변환"""
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = _validate_input(X)
    assert all(result.dtypes == np.float32)


def test_validate_input_fills_nan_with_median():
    """NaN을 중앙값으로 대체"""
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    result = _validate_input(X)
    assert not result.isnull().any().any()
    # 중앙값 = 2.0 (1, 3 의 중앙값)
    assert result["a"].iloc[1] == pytest.approx(2.0, abs=1e-3)


def test_validate_input_does_not_modify_original():
    """원본 DataFrame을 수정하지 않음"""
    X = pd.DataFrame({"a": [1.0, np.nan]})
    _validate_input(X)
    assert X["a"].isna().any()


def test_get_active_model_info_no_flags(tmp_path, monkeypatch):
    """flags.json이 없을 때 no model deployed 반환"""
    monkeypatch.setattr("pipeline.predict.FLAGS_PATH", tmp_path / "flags.json")
    info = get_active_model_info()
    assert info["status"] == "no model deployed"
