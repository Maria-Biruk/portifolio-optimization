"""
risk_metrics.py

Functions for computing returns, volatility, stationarity tests,
and risk metrics (VaR, Sharpe Ratio) on asset price series.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def calculate_daily_returns(price_series):
    """
    Compute daily percentage returns from a price series.

    Parameters
    ----------
    price_series : pd.Series

    Returns
    -------
    pd.Series

    Raises
    ------
    ValueError
        If the series is empty or has fewer than 2 observations.
    """
    if price_series is None or len(price_series) < 2:
        raise ValueError("price_series must contain at least 2 observations.")

    return price_series.pct_change()


def rolling_volatility(returns, window=30):
    """
    Compute rolling standard deviation of returns.

    Parameters
    ----------
    returns : pd.Series
    window : int, default 30

    Returns
    -------
    pd.Series
    """
    if window < 1:
        raise ValueError("window must be a positive integer.")
    return returns.rolling(window=window).std()


def detect_outliers(returns, n_std=3):
    """
    Flag return observations beyond n_std standard deviations from the mean.

    Parameters
    ----------
    returns : pd.Series
    n_std : float, default 3

    Returns
    -------
    pd.Series
        Subset of `returns` flagged as outliers.
    """
    clean_returns = returns.dropna()
    mean, std = clean_returns.mean(), clean_returns.std()
    threshold = n_std * std

    return clean_returns[(clean_returns - mean).abs() > threshold]


def run_adf_test(series, name="series"):
    """
    Run the Augmented Dickey-Fuller stationarity test.

    Parameters
    ----------
    series : pd.Series
    name : str
        Label for print output.

    Returns
    -------
    dict
        ADF statistic, p-value, critical values, and stationarity verdict.

    Raises
    ------
    ValueError
        If the series has fewer than 10 non-null observations
        (ADF is unreliable on very short series).
    """
    clean_series = series.dropna()
    if len(clean_series) < 10:
        raise ValueError(f"'{name}' has too few observations ({len(clean_series)}) for a reliable ADF test.")

    result = adfuller(clean_series)
    is_stationary = result[1] <= 0.05

    return {
        "name": name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "critical_values": result[4],
        "is_stationary": is_stationary,
    }


def calculate_var(returns, confidence_level=0.95):
    """
    Historical Value at Risk (VaR) at a given confidence level.

    Parameters
    ----------
    returns : pd.Series
    confidence_level : float, default 0.95

    Returns
    -------
    float
        VaR as a negative decimal (e.g. -0.05 = -5%).

    Raises
    ------
    ValueError
        If confidence_level is not between 0 and 1.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    clean_returns = returns.dropna()
    return clean_returns.quantile(1 - confidence_level)


def calculate_sharpe_ratio(returns, risk_free_rate_annual=0.02, trading_days=252):
    """
    Annualized Sharpe Ratio.

    Parameters
    ----------
    returns : pd.Series
    risk_free_rate_annual : float, default 0.02
    trading_days : int, default 252

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If returns have effectively zero variance (undefined Sharpe ratio).
    """
    clean_returns = returns.dropna()
    std = clean_returns.std()

    if np.isclose(std, 0, atol=1e-10):
        raise ValueError("Cannot compute Sharpe Ratio: returns have zero (or near-zero) variance.")

    daily_rf = risk_free_rate_annual / trading_days
    excess_return = clean_returns.mean() - daily_rf

    return (excess_return / std) * np.sqrt(trading_days)