# XAUUSD AI Trading Bot

An AI-powered Gold (XAU/USD) Forex trading bot combining machine learning, Smart Money Concepts (SMC), and macro fundamental analysis.

## Architecture

```
Market Data → Feature Engineering → AI Model → Signal → Risk Filter → MT5 Execution
```

## Strategy

- **Instrument:** XAUUSD (Gold vs USD)
- **Sessions:** London Open + London/NY Overlap (07:00–17:00 UTC)
- **Timeframes:** D1/H4 (bias) → H1 (confirmation) → M15 (entry)
- **Core approach:** Trend-following + breakout during active sessions, mean reversion during Asian session
- **AI layer:** XGBoost classifier trained on SMC features + macro regime filter

## Fundamental Drivers Tracked

- Real interest rates (10Y TIPS yield)
- DXY (US Dollar Index)
- Central bank policy (FOMC, Fed Funds Rate)
- Inflation expectations (CPI, PCE)
- Geopolitical risk index

## Technical Indicators

- EMA 9/20/50/200
- RSI (21-period, 80/20 levels)
- ATR (volatility-based stop sizing)
- Bollinger Bands
- MACD
- Order Blocks & Fair Value Gaps (SMC)

## Risk Management

- Max risk per trade: 1–2% of account equity
- Stop loss: 1.5x–2.0x ATR from entry
- Max daily drawdown: 3%
- Max total drawdown: 10%
- Circuit breaker: auto-pause after 3 consecutive losses

## Project Structure

```
tradingbot/
├── config/          # Settings and parameters
├── data/            # Data fetching and processing
├── features/        # Feature engineering
├── models/          # ML models
├── strategy/        # Trading strategy logic
├── risk/            # Risk management engine
├── execution/       # Broker API / MT5 bridge
├── backtest/        # Backtesting framework
├── monitoring/      # Alerts and performance tracking
└── tests/           # Unit and integration tests
```

## Setup

```bash
pip install -r requirements.txt
cp config/config.example.py config/config.py
# Fill in your broker credentials in config/config.py
python backtest/run_backtest.py
```

## Status

Currently in research and development phase.
