"""
QPE Trend Overlay 검증 테스트

검증 시나리오 4가지:
1. 상승 추세에서 공격적 매수 신호를 생성하는지 확인
2. 장기 추세 이탈 시 청산 신호를 생성하는지 확인
3. 급락 위험 구간에서 목표 비중이 줄어드는지 확인
4. Walk-Forward 엔진에서 복합 objective로 실행되는지 확인
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig
from backtest.walk_forward import WalkForwardConfig, WalkForwardEngine
from strategies.base import Signal, StrategyContext
from strategies.qpe_trend_overlay import QPETrendOverlayStrategy


def make_trending_ohlcv(n_days: int = 900) -> pd.DataFrame:
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    t = np.arange(n_days)
    close = 100_000 * np.exp(0.0014 * t + 0.035 * np.sin(2 * np.pi * t / 90))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.006
    low = np.minimum(open_, close) * 0.994
    volume = 2500 + 800 * np.sin(2 * np.pi * t / 45) + np.linspace(0, 1200, n_days)
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)


def make_breakdown_ohlcv(n_days: int = 900) -> pd.DataFrame:
    df = make_trending_ohlcv(n_days)
    break_start = n_days - 120
    decline = np.exp(np.linspace(0, -0.65, n_days - break_start))
    df.iloc[break_start:, df.columns.get_loc('close')] *= decline
    df['open'] = np.concatenate([[df['close'].iloc[0]], df['close'].iloc[:-1]])
    df['high'] = np.maximum(df['open'], df['close']) * 1.012
    df['low'] = np.minimum(df['open'], df['close']) * 0.988
    return df


def make_strategy(params=None):
    return QPETrendOverlayStrategy(params=params)


def test_overlay_generates_aggressive_buy():
    print("\n[테스트 1] 상승 추세 매수 신호")
    print("-" * 60)

    df = make_trending_ohlcv()
    strategy = make_strategy({
        'entry_threshold': 0.35,
        'exit_threshold': 0.20,
        'trend_entry_threshold': 0.35,
        'crash_entry_threshold': 0.80,
        'crash_exit_threshold': 0.90,
        'max_position': 0.95,
    })
    ctx = StrategyContext(
        timestamp=df.index[-1],
        ticker='TEST',
        ohlcv=df,
        current_position=0.0,
        cash=10_000_000,
        portfolio_value=10_000_000,
    )
    decision = strategy.generate_signal(ctx)

    assert decision.signal == Signal.BUY
    assert decision.target_weight >= 0.40

    print(f"  ✓ 매수 비중: {decision.target_weight:.3f}")


def test_overlay_exits_on_trend_breakdown():
    print("\n[테스트 2] 추세 이탈 청산 신호")
    print("-" * 60)

    df = make_breakdown_ohlcv()
    strategy = make_strategy({
        'entry_threshold': 0.45,
        'exit_threshold': 0.34,
        'crash_entry_threshold': 0.80,
        'crash_exit_threshold': 0.90,
    })
    ctx = StrategyContext(
        timestamp=df.index[-1],
        ticker='TEST',
        ohlcv=df,
        current_position=1.0,
        cash=0,
        portfolio_value=10_000_000,
    )
    decision = strategy.generate_signal(ctx)

    assert decision.signal == Signal.SELL

    print("  ✓ 청산 신호 생성")


def test_overlay_reduces_crash_exposure():
    print("\n[테스트 3] 급락 위험 비중 축소")
    print("-" * 60)

    strategy = make_strategy()
    calm = strategy.evaluate(make_trending_ohlcv()).target_weight
    breakdown = strategy.evaluate(make_breakdown_ohlcv()).target_weight

    assert calm > breakdown

    print(f"  ✓ 상승 추세 비중: {calm:.3f}")
    print(f"  ✓ 붕괴 구간 비중: {breakdown:.3f}")


def test_overlay_walk_forward_runs():
    print("\n[테스트 4] QPE Trend Overlay Walk-Forward 실행")
    print("-" * 60)

    engine = WalkForwardEngine(
        strategy_factory=make_strategy,
        backtest_config=BacktestConfig(
            initial_capital=10_000_000,
            commission_rate=0.001,
            slippage_rate=0.001,
        ),
        config=WalkForwardConfig(
            initial_train_years=1,
            test_years=1,
            step_years=1,
            mode='expanding',
            objective='qpe_composite_score',
            min_train_bars=300,
            min_test_bars=120,
        ),
    )
    result = engine.run(
        make_trending_ohlcv(n_days=1100),
        ticker='TEST',
        parameter_grid={
            'short_window': [10, 20],
            'long_window': [60, 100],
            'entry_threshold': [0.40],
            'exit_threshold': [0.25],
            'trend_entry_threshold': [0.40],
            'crash_entry_threshold': [0.80],
            'crash_exit_threshold': [0.90],
            'max_position': [0.90],
        },
    )

    assert len(result.windows) >= 2
    assert not result.optimization_log.empty
    assert 'objective_sharpe_component' in result.optimization_log.columns

    print(f"  ✓ 윈도우 수: {len(result.windows)}")
    print(f"  ✓ OOS Sharpe: {result.oos_metrics.sharpe_ratio:.2f}")


def main():
    print("=" * 60)
    print("QPE Trend Overlay 검증 테스트")
    print("=" * 60)

    tests = [
        test_overlay_generates_aggressive_buy,
        test_overlay_exits_on_trend_breakdown,
        test_overlay_reduces_crash_exposure,
        test_overlay_walk_forward_runs,
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


if __name__ == '__main__':
    main()
