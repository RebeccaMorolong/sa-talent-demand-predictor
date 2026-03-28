"""
Tests for model training pipelines — verifies they fit and predict
without errors using small synthetic datasets.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from src.models.unemployment_risk import _synthetic_data, build_pipeline


def test_synthetic_data_shape():
    df = _synthetic_data(n=100)
    assert len(df) == 100
    assert "employed" in df.columns
    assert "skill_count" in df.columns


def test_unemployment_risk_pipeline_fits():
    df = _synthetic_data(n=200)
    X = df.drop(columns=["employed"])
    y = df["employed"]
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_unemployment_risk_predict_proba():
    df = _synthetic_data(n=200)
    X = df.drop(columns=["employed"])
    y = df["employed"]
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
