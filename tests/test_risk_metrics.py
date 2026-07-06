"""
Unit tests for risk_metrics.py
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from risk_metrics import (
    calculate_daily_returns,
    rolling_volatility,
    detect_outliers,
    run_adf_test,
    calculate_var,
    calculate_sharpe_ratio,
)


@pytest.fixture
def sample_prices():
    return pd.Series([100, 102, 101, 105, 103, 108, 107, 110, 109, 112])


@pytest.fixture
def sample_returns(sample_prices):
    return calculate_daily_returns(sample_prices)


def test_calculate_daily_returns_basic(sample_prices):
    returns = calculate_daily_returns(sample_prices)
    assert len(returns) == len(sample_prices)
    assert pd.isna(returns.iloc[0])
    assert np.isclose(returns.iloc[1], 0.02)


def test_calculate_daily_returns_raises_on_short_series():
    with pytest.raises(ValueError):
        calculate_daily_returns(pd.Series([100]))


def test_rolling_volatility_output_length(sample_returns):
    vol = rolling_volatility(sample_returns, window=3)
    assert len(vol) == len(sample_returns)


def test_rolling_volatility_invalid_window(sample_returns):
    with pytest.raises(ValueError):
        rolling_volatility(sample_returns, window=0)


def test_detect_outliers_flags_extreme_values():
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.5, -0.6, 0.005])
    outliers = detect_outliers(returns, n_std=1)
    assert 0.5 in outliers.values or -0.6 in outliers.values


def test_calculate_var_within_range(sample_returns):
    var = calculate_var(sample_returns, confidence_level=0.95)
    assert var <= 0


def test_calculate_var_invalid_confidence(sample_returns):
    with pytest.raises(ValueError):
        calculate_var(sample_returns, confidence_level=1.5)


def test_calculate_sharpe_ratio_basic(sample_returns):
    sharpe = calculate_sharpe_ratio(sample_returns)
    assert isinstance(sharpe, float)


def test_calculate_sharpe_ratio_zero_variance():
    constant_returns = pd.Series([0.01] * 10)
    with pytest.raises(ValueError):
        calculate_sharpe_ratio(constant_returns)


def test_run_adf_test_returns_expected_keys(sample_prices):
    long_series = pd.Series(np.cumsum(np.random.normal(0, 1, 50)) + 100)
    result = run_adf_test(long_series, name="test_series")
    assert "adf_statistic" in result
    assert "p_value" in result
    assert "is_stationary" in result


def test_run_adf_test_raises_on_short_series():
    with pytest.raises(ValueError):
        run_adf_test(pd.Series([1, 2, 3]), name="too_short")