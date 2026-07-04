## Data Quality Summary

- Fetched daily OHLCV data for TSLA, BND, and SPY from Jan 1, 2015 to Jun 30, 2026 via yfinance (auto_adjust=False to preserve both raw Close and dividend/split-adjusted Close).
- No missing values found across any of the three assets (0 nulls in every column).
- All columns hold appropriate data types (float64 for prices, int64 for volume) — no type coercion was necessary.
- For TSLA, Adj Close == Close throughout the period, consistent with TSLA never having paid a dividend.
- For BND, Adj Close is meaningfully lower than Close, reflecting BND's regular dividend distributions — a reminder that Adj Close is the more accurate series to use for return calculations on income-paying assets.

## Volatility Insight

All three assets exhibit volatility clustering — extended calm periods punctuated by sharp spikes in volatility during market stress events (COVID crash in 2020, 2022 rate-hike cycle). TSLA maintains persistently elevated volatility throughout, while BND and SPY show mostly low volatility with pronounced, temporary spikes.

## Discussion: Model Selection Rationale

**LSTM decisively outperformed both ARIMA and SARIMA** on every metric, driven primarily by its use of a rolling 60-day context window rather than a single long-horizon static projection — an inherently easier forecasting task, not necessarily proof of superior pattern-learning alone.

**Simple vs. complex LSTM**: The smaller LSTM (1 layer, 32 units) performed only marginally worse than the larger one (2 layers, 50 units) — MAE of 11.57 vs. 9.65. This ~20% relative difference is fairly small given the larger model has substantially more parameters and training cost. This suggests the added architectural complexity provided limited additional value for this dataset and horizon, and a simpler, faster, more interpretable model may be preferable in practice — especially in a production setting where training cost and overfitting risk matter.

**ARIMA vs. SARIMA**: Adding seasonal terms improved AIC modestly (16368 → 16355), but the core structure remained a random walk (0,1,0), reinforcing that TSLA price changes carry little exploitable autocorrelation at any horizon tested — consistent with the Efficient Market Hypothesis.

**Overall recommendation**: For short-horizon, context-rich forecasting tasks, LSTM (even a simple configuration) is the stronger choice. For long-horizon forecasting, none of the models tested provide strong predictive power beyond a naive random-walk baseline — a finding that should temper confidence in any single model's long-term price forecasts and supports GMF's stated philosophy of using these models as one input among many, not standalone predictors.
