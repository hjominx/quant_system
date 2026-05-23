"""ETH micro scalper strategy tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import Signal, StrategyContext
from strategies.eth_micro_scalper import ETHMicroScalperStrategy


def make_scalp_ohlcv(n_bars: int = 90, *, rising: bool = True) -> pd.DataFrame:
    """Create synthetic 5m ETH data with enough texture for indicators."""
    dates = pd.date_range(start="2026-01-01", periods=n_bars, freq="5min")
    drift = np.linspace(0, 0.045 if rising else -0.025, n_bars)
    wave = np.sin(np.linspace(0, 10, n_bars)) * 0.004
    close = 3_000_000 * (1 + drift + wave)
    close[-1] *= 1.004 if rising else 0.993
    open_ = close * 0.999
    high = close * 1.003
    low = close * 0.997
    volume = np.linspace(900, 1300, n_bars)
    volume[-1] = 1500
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def test_scalper_buys_textured_uptrend():
    print("\n[테스트 1] 스캘퍼 상승 추세 진입")
    print("-" * 60)

    strategy = ETHMicroScalperStrategy(params={"rsi_entry_max": 95})
    ohlcv = make_scalp_ohlcv()
    decision = strategy.generate_signal(StrategyContext(
        timestamp=ohlcv.index[-1],
        ticker="KRW-ETH",
        ohlcv=ohlcv,
        current_position=0.0,
        cash=200000,
        portfolio_value=200000,
    ))

    assert decision.signal == Signal.BUY
    assert decision.target_weight > 0

    print(f"  ✓ BUY target={decision.target_weight:.3f}")


def test_scalper_exits_take_profit():
    print("\n[테스트 2] 스캘퍼 익절")
    print("-" * 60)

    strategy = ETHMicroScalperStrategy()
    ohlcv = make_scalp_ohlcv()
    close = float(ohlcv["close"].iloc[-1])
    decision = strategy.generate_signal(StrategyContext(
        timestamp=ohlcv.index[-1],
        ticker="KRW-ETH",
        ohlcv=ohlcv,
        current_position=0.05,
        avg_buy_price=close * 0.99,
        cash=50000,
        portfolio_value=200000,
    ))

    assert decision.signal == Signal.SELL

    print("  ✓ take-profit SELL")


def test_scalper_exits_stop_loss():
    print("\n[테스트 3] 스캘퍼 손절")
    print("-" * 60)

    strategy = ETHMicroScalperStrategy()
    ohlcv = make_scalp_ohlcv(rising=False)
    close = float(ohlcv["close"].iloc[-1])
    decision = strategy.generate_signal(StrategyContext(
        timestamp=ohlcv.index[-1],
        ticker="KRW-ETH",
        ohlcv=ohlcv,
        current_position=0.05,
        avg_buy_price=close * 1.01,
        cash=50000,
        portfolio_value=200000,
    ))

    assert decision.signal == Signal.SELL

    print("  ✓ stop-loss SELL")


def main():
    print("=" * 60)
    print("ETH Micro Scalper 테스트")
    print("=" * 60)
    tests = [
        test_scalper_buys_textured_uptrend,
        test_scalper_exits_take_profit,
        test_scalper_exits_stop_loss,
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
