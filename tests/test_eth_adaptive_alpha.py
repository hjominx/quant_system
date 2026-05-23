"""ETH adaptive alpha strategy tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, BacktestEngine
from strategies.base import Signal, StrategyContext
from strategies.eth_adaptive_alpha import ETHAdaptiveAlphaStrategy


def make_quality_uptrend(n_bars: int = 420) -> pd.DataFrame:
    dates = pd.date_range(start="2025-01-01", periods=n_bars, freq="4h")
    t = np.arange(n_bars)
    close = 100_000 * np.exp(0.0011 * t + 0.010 * np.sin(2 * np.pi * t / 36))
    close[-80:] *= np.exp(np.linspace(0, 0.24, 80))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = 2000 + 100 * np.sin(2 * np.pi * t / 24)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


def make_breakdown(n_bars: int = 420) -> pd.DataFrame:
    df = make_quality_uptrend(n_bars)
    df.iloc[-40:, df.columns.get_loc("close")] *= np.exp(np.linspace(0, -0.30, 40))
    df["open"] = np.concatenate([[df["close"].iloc[0]], df["close"].iloc[:-1]])
    df["high"] = np.maximum(df["open"], df["close"]) * 1.006
    df["low"] = np.minimum(df["open"], df["close"]) * 0.994
    return df


def make_strategy(params=None):
    merged = {
        "entry_score": 0.55,
        "min_momentum": 0.015,
        "min_adx": 10,
        "rsi_max": 95,
        "max_realized_volatility": 3.0,
        "max_atr_pct": 0.20,
    }
    if params:
        merged.update(params)
    return ETHAdaptiveAlphaStrategy(params=merged)


def test_adaptive_alpha_generates_quality_entry():
    print("\n[테스트 1] ETH adaptive quality entry")
    print("-" * 60)

    df = make_quality_uptrend()
    strategy = make_strategy()
    decision = strategy.generate_signal(StrategyContext(
        timestamp=df.index[-1],
        ticker="TEST-ETH",
        ohlcv=df,
        current_position=0.0,
        cash=10_000_000,
        portfolio_value=10_000_000,
    ))

    assert decision.signal == Signal.BUY
    assert 0 < decision.target_weight <= 1

    print(f"  ✓ target weight: {decision.target_weight:.3f}")


def test_adaptive_alpha_exits_breakdown():
    print("\n[테스트 2] ETH adaptive breakdown exit")
    print("-" * 60)

    df = make_breakdown()
    strategy = make_strategy()
    strategy.peak_close = df["close"].iloc[-45]
    decision = strategy.generate_signal(StrategyContext(
        timestamp=df.index[-1],
        ticker="TEST-ETH",
        ohlcv=df,
        current_position=1.0,
        avg_buy_price=df["close"].iloc[-100],
        cash=0,
        portfolio_value=10_000_000,
    ))

    assert decision.signal == Signal.SELL

    print("  ✓ exit signal generated")


def test_adaptive_alpha_backtest_runs():
    print("\n[테스트 3] ETH adaptive backtest runs")
    print("-" * 60)

    result = BacktestEngine(BacktestConfig(
        initial_capital=10_000_000,
        commission_rate=0.0004,
        slippage_rate=0.0003,
    )).run(
        make_strategy(),
        make_quality_uptrend(n_bars=520),
        ticker="TEST-ETH",
    )

    assert not result.equity_curve.empty
    assert result.metrics.days > 0

    print(f"  ✓ return: {result.metrics.total_return*100:.2f}%")
    print(f"  ✓ trades: {result.metrics.total_trades}")


def main():
    print("=" * 60)
    print("ETH Adaptive Alpha 검증 테스트")
    print("=" * 60)

    tests = [
        test_adaptive_alpha_generates_quality_entry,
        test_adaptive_alpha_exits_breakdown,
        test_adaptive_alpha_backtest_runs,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as exc:
            print(f"  ✗ 실패: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ✗ 예외: {type(exc).__name__}: {exc}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"결과: {passed}개 통과 / {failed}개 실패 (총 {len(tests)}개)")
    print("=" * 60)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
