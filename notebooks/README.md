## Data Quality Summary

- Fetched daily OHLCV data for TSLA, BND, and SPY from Jan 1, 2015 to Jun 30, 2026 via yfinance (auto_adjust=False to preserve both raw Close and dividend/split-adjusted Close).
- No missing values found across any of the three assets (0 nulls in every column).
- All columns hold appropriate data types (float64 for prices, int64 for volume) — no type coercion was necessary.
- For TSLA, Adj Close == Close throughout the period, consistent with TSLA never having paid a dividend.
- For BND, Adj Close is meaningfully lower than Close, reflecting BND's regular dividend distributions — a reminder that Adj Close is the more accurate series to use for return calculations on income-paying assets.

## Volatility Insight

All three assets exhibit volatility clustering — extended calm periods punctuated by sharp spikes in volatility during market stress events (COVID crash in 2020, 2022 rate-hike cycle). TSLA maintains persistently elevated volatility throughout, while BND and SPY show mostly low volatility with pronounced, temporary spikes.
