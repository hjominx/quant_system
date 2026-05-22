"""
ETH/Cash Regime 전략 검증 테스트

검증 시나리오:
1. 장기 상승 + 돌파 구간에서 매수 신호 생성
2. 보유 중 추세 붕괴 시 청산 신호 생성
3. 변동성이 높아질수록 목표 비중 축소
4. BacktestEngine에서 look-ahead 없이 실행
5. 공격형 저점 탐색 진입
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, BacktestEngine
from strategies.base import Signal, StrategyContext
from strategies.eth_cash_regime import ETHCashRegimeStrategy


def make_breakout_ohlcv(n_days: int = 520) -> pd.DataFrame:
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    close = 100_000 * np.exp(0.0008 * t + 0.025 * np.sin(2 * np.pi * t / 55))
    close[-45:] *= np.exp(np.linspace(0, 0.32, 45))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.006
    low = np.minimum(open_, close) * 0.994
    volume = 2000 + 250 * np.sin(2 * np.pi * t / 30)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


def make_breakdown_ohlcv(n_days: int = 520) -> pd.DataFrame:
    df = make_breakout_ohlcv(n_days)
    decline = np.exp(np.linspace(0, -0.55, 70))
    df.iloc[-70:, df.columns.get_loc("close")] *= decline
    df["open"] = np.concatenate([[df["close"].iloc[0]], df["close"].iloc[:-1]])
    df["high"] = np.maximum(df["open"], df["close"]) * 1.010
    df["low"] = np.minimum(df["open"], df["close"]) * 0.990
    return df


def make_dip_probe_ohlcv(n_days: int = 180) -> pd.DataFrame:
    dates = pd.date_range(start="2025-01-01", periods=n_days, freq="4h")
    t = np.arange(n_days)
    close = 100_000 * np.exp(0.0002 * t)
    close[-36:] *= np.exp(np.linspace(0, -0.08, 36))
    close[-10:] *= np.array([1.000, 0.995, 1.004, 0.991, 1.003, 0.990, 1.004, 0.992, 1.002, 0.996])
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    volume = 2000 + 100 * np.sin(2 * np.pi * t / 24)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


def make_strategy(params=None):
    params = params or {}
    params.setdefault("momentum_min", 0.02)
    params.setdefault("rsi_max", 101)
    params.setdefault("max_entry_volatility", 2.0)
    params.setdefault("target_volatility", 0.38)
    return ETHCashRegimeStrategy(params=params)


def test_regime_generates_breakout_buy():
    print("\n[테스트 1] ETH/Cash 돌파 매수")
    print("-" * 60)

    df = make_breakout_ohlcv()
    strategy = make_strategy()
    ctx = StrategyContext(
        timestamp=df.index[-1],
        ticker="TEST-ETH",
        ohlcv=df,
        current_position=0.0,
        cash=10_000_000,
        portfolio_value=10_000_000,
    )
    decision = strategy.generate_signal(ctx)

    assert decision.signal == Signal.BUY
    assert 0 < decision.target_weight <= 1

    print(f"  ✓ 매수 비중: {decision.target_weight:.3f}")


def test_regime_exits_on_breakdown():
    print("\n[테스트 2] ETH/Cash 추세 붕괴 청산")
    print("-" * 60)

    df = make_breakdown_ohlcv()
    strategy = make_strategy()
    strategy.peak_close = df["close"].iloc[-90]
    ctx = StrategyContext(
        timestamp=df.index[-1],
        ticker="TEST-ETH",
        ohlcv=df,
        current_position=1.0,
        avg_buy_price=df["close"].iloc[-120],
        cash=0,
        portfolio_value=10_000_000,
    )
    decision = strategy.generate_signal(ctx)

    assert decision.signal == Signal.SELL

    print("  ✓ 청산 신호 생성")


def test_regime_target_weight_reduces_with_volatility():
    print("\n[테스트 3] 변동성 상승 시 비중 축소")
    print("-" * 60)

    strategy = make_strategy({
        "min_position": 0.10,
        "max_position": 1.00,
        "target_volatility": 0.40,
    })
    calm_weight = strategy._target_weight(0.35)
    volatile_weight = strategy._target_weight(1.20)

    assert calm_weight > volatile_weight
    assert volatile_weight >= 0.10

    print(f"  ✓ 평온 구간 비중: {calm_weight:.3f}")
    print(f"  ✓ 고변동 구간 비중: {volatile_weight:.3f}")


def test_regime_backtest_runs():
    print("\n[테스트 4] ETH/Cash 백테스트 실행")
    print("-" * 60)

    result = BacktestEngine(BacktestConfig(
        initial_capital=10_000_000,
        commission_rate=0.001,
        slippage_rate=0.001,
    )).run(
        make_strategy(),
        make_breakout_ohlcv(n_days=650),
        ticker="TEST-ETH",
    )

    assert not result.equity_curve.empty
    assert result.metrics.days > 0

    print(f"  ✓ 수익률: {result.metrics.total_return*100:.2f}%")
    print(f"  ✓ 거래수: {result.metrics.total_trades}")


def test_aggressive_dip_probe_entry():
    print("\n[테스트 5] 공격형 저점 탐색 진입")
    print("-" * 60)

    df = make_dip_probe_ohlcv()
    strategy = ETHCashRegimeStrategy(params={
        "fast_window": 6,
        "slow_window": 36,
        "exit_window": 12,
        "momentum_window": 9,
        "rsi_window": 14,
        "vol_window": 12,
        "periods_per_year": 365 * 6,
        "max_entry_volatility": 2.20,
        "target_volatility": 0.50,
        "max_position": 0.70,
        "min_position": 0.25,
        "use_exit_ma": False,
        "dip_probe_enabled": True,
        "dip_rsi_min": 0,
        "dip_rsi_max": 45,
        "dip_momentum_floor": -0.10,
        "dip_max_position": 0.55,
    })
    ctx = StrategyContext(
        timestamp=df.index[-1],
        ticker="TEST-ETH",
        ohlcv=df,
        current_position=0.0,
        cash=10_000_000,
        portfolio_value=10_000_000,
    )
    decision = strategy.generate_signal(ctx)

    assert decision.signal == Signal.BUY
    assert 0 < decision.target_weight <= 0.55
    assert "aggressive dip entry" in decision.reason

    print(f"  ✓ 공격형 탐색 매수 비중: {decision.target_weight:.3f}")


def main():
    print("=" * 60)
    print("ETH/Cash Regime 검증 테스트")
    print("=" * 60)

    tests = [
        test_regime_generates_breakout_buy,
        test_regime_exits_on_breakdown,
        test_regime_target_weight_reduces_with_volatility,
        test_regime_backtest_runs,
        test_aggressive_dip_probe_entry,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ 실패: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ 예외: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"결과: {passed}개 통과 / {failed}개 실패 (총 {len(tests)}개)")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
