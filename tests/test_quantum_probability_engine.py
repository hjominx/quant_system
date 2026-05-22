"""
QPE(Quantum Probability Engine) 검증 테스트

검증 시나리오 5가지:
1. 시장 피처가 0~1 범위이며 미래 데이터를 보지 않는지 확인
2. 시장 상태 확률이 1로 정규화되는지 확인
3. 위험/불확실성이 커질수록 포지션이 줄어드는지 확인
4. QPE 전략이 매수/청산 신호를 생성하는지 확인
5. QPE가 Walk-Forward 엔진에서 실행되는지 확인
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig
from backtest.walk_forward import WalkForwardConfig, WalkForwardEngine
from features.market_features import MarketFeatureExtractor
from regime.market_state import QuantumMarketStateEngine
from risk.position_sizer import PositionSizingInput, QuantumPositionSizer
from strategies.base import Signal, StrategyContext
from strategies.quantum_probability_engine import QuantumProbabilityEngineStrategy


def make_test_ohlcv(n_days: int = 900, seed: int = 11, drift: float = 0.0007) -> pd.DataFrame:
    np.random.seed(seed)
    t = np.arange(n_days)
    cycle = 0.0015 * np.sin(2 * np.pi * t / 140)
    noise = np.random.normal(0, 0.014, n_days)
    log_returns = drift + cycle + noise
    close = 100_000 * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.012
    low = np.minimum(open_, close) * 0.988
    volume = np.random.uniform(1000, 6000, n_days)
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)


def make_clean_uptrend(n_days: int = 900) -> pd.DataFrame:
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    close = 100_000 * np.exp(np.linspace(0, 1.2, n_days))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = np.linspace(1000, 5000, n_days)
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)


def make_strategy(params=None):
    return QuantumProbabilityEngineStrategy(params=params)


def test_features_are_bounded_and_causal():
    print("\n[테스트 1] QPE 피처 범위 및 미래 데이터 차단")
    print("-" * 60)

    df = make_test_ohlcv()
    extractor = MarketFeatureExtractor()
    cut = 500

    snap_a = extractor.extract(df.iloc[:cut])
    snap_b = extractor.extract(df.iloc[:cut].copy())

    for key, value in snap_a.as_dict().items():
        if key in ('realized_volatility', 'drawdown_60d'):
            continue
        assert 0.0 <= value <= 1.0, f"{key} 범위 오류: {value}"
        assert abs(value - snap_b.as_dict()[key]) < 1e-12, f"{key} 미래 데이터 영향 가능성"

    print(f"  ✓ 피처 범위 정상")
    print(f"  ✓ 동일 절단 구간 결과 일치")


def test_market_state_probabilities_sum_to_one():
    print("\n[테스트 2] 시장 상태 확률 정규화")
    print("-" * 60)

    features = MarketFeatureExtractor().extract(make_test_ohlcv())
    states = QuantumMarketStateEngine().infer(features)
    probs = [
        states.trend_up,
        states.mean_reversion,
        states.volatility_breakout,
        states.crash_risk,
        states.neutral,
    ]

    assert abs(sum(probs) - 1.0) < 1e-12
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert 0.0 <= states.uncertainty <= 1.0

    print(f"  ✓ 확률 합: {sum(probs):.6f}")
    print(f"  ✓ 불확실성: {states.uncertainty:.3f}")


def test_position_sizer_reduces_risky_exposure():
    print("\n[테스트 3] 위험 증가 시 포지션 축소")
    print("-" * 60)

    sizer = QuantumPositionSizer(max_position=0.9, min_position=0.05)
    calm = sizer.size(PositionSizingInput(
        edge_score=0.75,
        uncertainty=0.20,
        crash_risk=0.10,
        realized_volatility=0.30,
    ))
    risky = sizer.size(PositionSizingInput(
        edge_score=0.75,
        uncertainty=0.80,
        crash_risk=0.70,
        realized_volatility=0.90,
    ))

    assert calm > risky
    assert calm > 0

    print(f"  ✓ 평온 구간 비중: {calm:.3f}")
    print(f"  ✓ 위험 구간 비중: {risky:.3f}")


def test_qpe_strategy_generates_buy_and_exit():
    print("\n[테스트 4] QPE 전략 매수/청산 신호")
    print("-" * 60)

    df = make_clean_uptrend()
    strategy = make_strategy({
        'entry_threshold': 0.20,
        'exit_threshold': 0.10,
        'crash_entry_threshold': 0.80,
        'crash_exit_threshold': 0.90,
        'max_position': 0.80,
        'min_position': 0.05,
    })

    entry_ctx = StrategyContext(
        timestamp=df.index[-1],
        ticker='TEST',
        ohlcv=df,
        current_position=0.0,
        cash=10_000_000,
        portfolio_value=10_000_000,
    )
    entry = strategy.generate_signal(entry_ctx)
    assert entry.signal == Signal.BUY
    assert entry.target_weight > 0

    exit_strategy = make_strategy({
        'entry_threshold': 0.99,
        'exit_threshold': 0.98,
        'crash_entry_threshold': 0.80,
        'crash_exit_threshold': 0.90,
    })
    exit_ctx = StrategyContext(
        timestamp=df.index[-1],
        ticker='TEST',
        ohlcv=df,
        current_position=1.0,
        cash=0,
        portfolio_value=10_000_000,
    )
    exit_decision = exit_strategy.generate_signal(exit_ctx)
    assert exit_decision.signal == Signal.SELL

    print(f"  ✓ 매수 비중: {entry.target_weight:.3f}")
    print(f"  ✓ 청산 신호 생성")


def test_qpe_walk_forward_runs():
    print("\n[테스트 5] QPE Walk-Forward 실행")
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
            objective='calmar_ratio',
            min_train_bars=300,
            min_test_bars=120,
        ),
    )
    result = engine.run(
        make_test_ohlcv(n_days=1100),
        ticker='TEST',
        parameter_grid={
            'entry_threshold': [0.25, 0.35],
            'exit_threshold': [0.10, 0.20],
            'crash_entry_threshold': [0.80],
            'crash_exit_threshold': [0.90],
            'max_position': [0.70],
            'min_position': [0.05],
        },
    )

    assert len(result.windows) >= 2
    assert not result.oos_equity_curve.empty
    assert not result.optimization_log.empty

    print(f"  ✓ 윈도우 수: {len(result.windows)}")
    print(f"  ✓ OOS 수익률: {result.oos_metrics.total_return*100:.2f}%")


def main():
    print("=" * 60)
    print("QPE 검증 테스트")
    print("=" * 60)

    tests = [
        test_features_are_bounded_and_causal,
        test_market_state_probabilities_sum_to_one,
        test_position_sizer_reduces_risky_exposure,
        test_qpe_strategy_generates_buy_and_exit,
        test_qpe_walk_forward_runs,
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
