"""
data_loader.py

Handles fetching, validating, and cleaning historical price data
for the GMF portfolio optimization project.
"""

import os
import pandas as pd
import yfinance as yf


def fetch_asset_data(tickers, start_date, end_date, auto_adjust=False):
    """
    Fetch historical OHLCV data for one or more tickers via yfinance.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols, e.g. ["TSLA", "BND", "SPY"].
    start_date : str
        Start date in "YYYY-MM-DD" format.
    end_date : str
        End date in "YYYY-MM-DD" format.
    auto_adjust : bool, default False
        If False, retains both raw 'Close' and dividend/split-adjusted
        'Adj Close' as separate columns.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker -> cleaned DataFrame with flat column names.

    Raises
    ------
    ValueError
        If tickers is empty, or if no data is returned for a ticker.
    """
    if not tickers:
        raise ValueError("`tickers` must be a non-empty list of ticker symbols.")

    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=auto_adjust)

        if df.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}'. "
                f"Check the symbol and date range ({start_date} to {end_date})."
            )

        # yfinance may return a MultiIndex (Price, Ticker) even for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        data[ticker] = df

    return data


def validate_data(df, ticker="asset"):
    """
    Run basic data-quality checks on a price DataFrame and report issues.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with at least an 'Adj Close' column.
    ticker : str
        Label used in printed messages for context.

    Returns
    -------
    dict
        Summary of missing values per column and any dtype issues found.
    """
    if df is None or df.empty:
        raise ValueError(f"Data for '{ticker}' is empty or None.")

    missing = df.isnull().sum()
    non_numeric_cols = [
        col for col in df.select_dtypes(exclude="number").columns
    ]

    summary = {
        "ticker": ticker,
        "total_missing": int(missing.sum()),
        "missing_by_column": missing.to_dict(),
        "non_numeric_columns": non_numeric_cols,
    }

    if summary["total_missing"] > 0:
        print(f"[WARNING] {ticker}: {summary['total_missing']} missing values found.")
    if non_numeric_cols:
        print(f"[WARNING] {ticker}: non-numeric columns detected: {non_numeric_cols}")

    return summary


def clean_data(df, method="ffill"):
    """
    Handle missing values in a price DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    method : str, default "ffill"
        One of "ffill" (forward-fill), "interpolate", or "drop".

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the input DataFrame.

    Raises
    ------
    ValueError
        If an unsupported method is passed.
    """
    if method not in {"ffill", "interpolate", "drop"}:
        raise ValueError(f"Unsupported method '{method}'. Use 'ffill', 'interpolate', or 'drop'.")

    cleaned = df.copy()

    if method == "ffill":
        cleaned = cleaned.ffill()
    elif method == "interpolate":
        cleaned = cleaned.interpolate()
    elif method == "drop":
        cleaned = cleaned.dropna()

    return cleaned


def save_processed_data(data, output_dir="data/processed"):
    """
    Save each ticker's DataFrame to a CSV file.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
    output_dir : str
        Directory to save files into; created if it doesn't exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    for ticker, df in data.items():
        path = os.path.join(output_dir, f"{ticker}_processed.csv")
        df.to_csv(path)
        print(f"Saved {path}")


def load_processed_data(tickers, input_dir="data/processed"):
    """
    Load previously saved processed CSVs back into DataFrames.

    Parameters
    ----------
    tickers : list[str]
    input_dir : str

    Returns
    -------
    dict[str, pd.DataFrame]

    Raises
    ------
    FileNotFoundError
        If an expected CSV file is missing.
    """
    data = {}
    for ticker in tickers:
        path = os.path.join(input_dir, f"{ticker}_processed.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected processed file not found: {path}")
        data[ticker] = pd.read_csv(path, index_col="Date", parse_dates=True)
    return data