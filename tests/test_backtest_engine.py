"""
백테스트 엔진 검증 테스트

여기서 통과해야만 실제 데이터로 검증 가능.
3가지 핵심 테스트:
1. 알려진 답이 있는 합성 데이터로 정확성 확인
2. Look-ahead bias 검출 (미래 데이터 사용 시도 시 결과 안 바뀌어야 함)
3. 거래비용이 결과에 제대로 반영되는지 확인
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from strategies.sma_crossover import SMACrossoverStrategy
from strategies.base import BaseStrategy, Signal, StrategyContext, TradeDecision
from backtest.engine import BacktestEngine, BacktestConfig


class BuyAndHoldForeverStrategy(BaseStrategy):
    """강제청산 회귀 테스트용 전략"""
    
    def __init__(self):
        super().__init__(
            name="BuyAndHoldForever",
            params={'required_history_days': 1},
        )
    
    def _validate_params(self) -> None:
        pass
    
    def generate_signal(self, ctx: StrategyContext) -> TradeDecision:
        if ctx.current_position == 0:
            return TradeDecision(
                signal=Signal.BUY,
                target_weight=1.0,
                reason="테스트용 최초 매수",
            )
        return TradeDecision(
            signal=Signal.HOLD,
            target_weight=0.0,
            reason="보유",
        )


def generate_synthetic_ohlcv(n_days: int = 1000, seed: int = 42) -> pd.DataFrame:
    """재현 가능한 합성 OHLCV 생성 (검증용)
    
    상승추세 + 변동성 섞인 가격 시리즈 생성.
    """
    np.random.seed(seed)
    
    # 사이클 + 노이즈 (교차가 자주 발생하도록 사이클 강하게)
    t = np.arange(n_days)
    # 두 개의 사이클을 겹쳐서 다양한 추세 전환 만들기
    cycle1 = 0.0015 * np.cos(2 * np.pi * t / 120)  # 120일 사이클
    cycle2 = 0.0008 * np.cos(2 * np.pi * t / 60)   # 60일 사이클
    drift = 0.0003  # 약한 상승 드리프트
    noise = np.random.normal(0, 0.02, n_days)  # 일일 변동성 2%
    
    log_returns = drift + cycle1 + cycle2 + noise
    close = 10_000_000 * np.exp(np.cumsum(log_returns))
    
    # OHLCV 생성 (고가/저가는 종가 기준 ±1% 범위에서 랜덤)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_ = np.concatenate([[close[0]], close[:-1]])  # 시가는 전일 종가
    volume = np.random.uniform(1e9, 5e9, n_days)
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    df = pd.DataFrame({
        'open': open_,
        'high': np.maximum(high, np.maximum(open_, close)),
        'low': np.minimum(low, np.minimum(open_, close)),
        'close': close,
        'volume': volume,
    }, index=dates)
    
    return df


def test_basic_backtest():
    """테스트 1: 기본 백테스트가 에러 없이 실행되는가"""
    print("\n[테스트 1] 기본 백테스트 실행")
    print("-" * 60)
    
    ohlcv = generate_synthetic_ohlcv(1000)
    strategy = SMACrossoverStrategy()
    config = BacktestConfig(initial_capital=10_000_000)
    
    engine = BacktestEngine(config)
    result = engine.run(strategy, ohlcv, ticker='SYNTH-TEST')
    
    assert result.equity_curve is not None
    assert len(result.equity_curve) > 0
    assert result.metrics is not None
    
    print(f"  ✓ 백테스트 실행 완료")
    print(f"  ✓ 자본 곡선 길이: {len(result.equity_curve)}")
    print(f"  ✓ 총 거래수: {result.metrics.total_trades}")
    return result


def test_no_lookahead_bias():
    """테스트 2: Look-ahead bias 부재 확인
    
    동일한 데이터의 앞부분만 사용해서 백테스트를 돌리고,
    전체 데이터로 돌린 결과의 해당 구간이 정확히 일치하는지 확인.
    만약 미래 데이터를 몰래 보고 있다면 결과가 달라짐.
    """
    print("\n[테스트 2] Look-ahead bias 부재 검증")
    print("-" * 60)
    
    full_ohlcv = generate_synthetic_ohlcv(1000)
    half_ohlcv = full_ohlcv.iloc[:500].copy()
    
    strategy_full = SMACrossoverStrategy()
    strategy_half = SMACrossoverStrategy()
    config = BacktestConfig(initial_capital=10_000_000)
    engine = BacktestEngine(config)
    
    result_full = engine.run(strategy_full, full_ohlcv, ticker='TEST')
    result_half = engine.run(strategy_half, half_ohlcv, ticker='TEST')
    
    # 두 백테스트의 공통 구간 비교
    common_dates = result_full.equity_curve.index.intersection(
        result_half.equity_curve.index
    )
    # 절단된 백테스트의 마지막 날은 종료 강제청산이 반영되므로
    # 전체 백테스트의 같은 날짜와 비교 대상이 아님.
    common_dates = common_dates[common_dates < result_half.equity_curve.index[-1]]
    
    full_values = result_full.equity_curve.loc[common_dates]
    half_values = result_half.equity_curve.loc[common_dates]
    
    # 절대 오차 < 1원 (부동소수점 오차 허용)
    max_diff = (full_values - half_values).abs().max()
    
    print(f"  공통 구간 길이: {len(common_dates)}일")
    print(f"  자본 곡선 최대 차이: {max_diff:.4f}원")
    
    if max_diff < 1.0:
        print(f"  ✓ Look-ahead bias 없음 (전체/부분 백테스트 결과 일치)")
    else:
        print(f"  ✗ 경고: 결과 불일치 → look-ahead bias 가능성!")
        raise AssertionError(f"Look-ahead bias 검출: 최대 차이 {max_diff}")
    
    return result_full, result_half


def test_transaction_costs():
    """테스트 3: 거래비용이 결과에 제대로 반영되는가
    
    수수료를 0으로 한 백테스트와 정상 백테스트 비교.
    수수료가 있는 쪽 수익률이 더 낮아야 함.
    """
    print("\n[테스트 3] 거래비용 반영 검증")
    print("-" * 60)
    
    ohlcv = generate_synthetic_ohlcv(1000)
    
    config_no_cost = BacktestConfig(
        initial_capital=10_000_000,
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    config_with_cost = BacktestConfig(
        initial_capital=10_000_000,
        commission_rate=0.0025,
        slippage_rate=0.002,
    )
    
    strategy1 = SMACrossoverStrategy()
    strategy2 = SMACrossoverStrategy()
    
    result_no_cost = BacktestEngine(config_no_cost).run(strategy1, ohlcv, 'TEST')
    result_with_cost = BacktestEngine(config_with_cost).run(strategy2, ohlcv, 'TEST')
    
    no_cost_return = result_no_cost.metrics.total_return
    with_cost_return = result_with_cost.metrics.total_return
    diff = no_cost_return - with_cost_return
    
    print(f"  수수료 0%   누적수익률: {no_cost_return*100:.2f}%")
    print(f"  수수료 0.45% 누적수익률: {with_cost_return*100:.2f}%")
    print(f"  차이: {diff*100:.2f}%p")
    
    if no_cost_return > with_cost_return:
        print(f"  ✓ 거래비용이 결과에 반영됨")
    else:
        raise AssertionError("거래비용이 반영되지 않음")
    
    n_trades = result_with_cost.metrics.total_trades
    expected_min_cost = n_trades * 0.0045  # 왕복 0.45%
    print(f"  거래수: {n_trades}회, 예상 최소 비용: {expected_min_cost*100:.2f}%p")
    
    return result_no_cost, result_with_cost


def test_buy_and_hold_comparison():
    """테스트 4: 단순보유(B&H)와 비교
    
    SMA 전략과 단순보유 비교. 합성 데이터에서 단순보유 수익률이 얼마인지 보고
    SMA 전략과의 차이가 합리적인지 확인.
    """
    print("\n[테스트 4] 단순보유 vs SMA 전략 비교")
    print("-" * 60)
    
    ohlcv = generate_synthetic_ohlcv(1000)
    bh_return = (ohlcv['close'].iloc[-1] / ohlcv['close'].iloc[0]) - 1
    
    strategy = SMACrossoverStrategy()
    config = BacktestConfig()
    result = BacktestEngine(config).run(strategy, ohlcv, 'TEST')
    
    print(f"  단순보유 수익률: {bh_return*100:.2f}%")
    print(f"  SMA 전략 수익률: {result.metrics.total_return*100:.2f}%")
    print(f"  단순보유 MDD: (별도 계산 필요)")
    print(f"  SMA 전략 MDD: {result.metrics.max_drawdown*100:.2f}%")
    
    return result


def test_final_liquidation_costs_reflected():
    """테스트 5: 종료 강제청산 비용이 최종 자산곡선과 거래기록에 반영되는가"""
    print("\n[테스트 5] 종료 강제청산 비용 반영 검증")
    print("-" * 60)
    
    dates = pd.date_range(start='2024-01-01', periods=12, freq='D')
    ohlcv = pd.DataFrame({
        'open': [100.0] * len(dates),
        'high': [100.0] * len(dates),
        'low': [100.0] * len(dates),
        'close': [100.0] * len(dates),
        'volume': [1_000_000.0] * len(dates),
    }, index=dates)
    
    config = BacktestConfig(
        initial_capital=10_000_000,
        commission_rate=0.01,
        slippage_rate=0.0,
    )
    result = BacktestEngine(config).run(
        BuyAndHoldForeverStrategy(),
        ohlcv,
        ticker='TEST',
    )
    
    expected_entry_price = 101.0
    expected_exit_price = 99.0
    expected_final_equity = config.initial_capital * (expected_exit_price / expected_entry_price)
    
    actual_final_equity = result.equity_curve.iloc[-1]
    final_diff = abs(actual_final_equity - expected_final_equity)
    
    assert len(result.trades) == 1
    assert abs(result.trades.iloc[0]['entry_price'] - expected_entry_price) < 1e-9
    assert abs(result.trades.iloc[0]['exit_price'] - expected_exit_price) < 1e-9
    assert final_diff < 1.0, f"최종 자산 불일치: {actual_final_equity} vs {expected_final_equity}"
    
    print(f"  ✓ 진입가 비용 반영: {result.trades.iloc[0]['entry_price']:.2f}")
    print(f"  ✓ 종료 청산가 비용 반영: {result.trades.iloc[0]['exit_price']:.2f}")
    print(f"  ✓ 최종 자산곡선 비용 반영: {actual_final_equity:,.0f}원")
    
    return result


def main():
    print("=" * 60)
    print("백테스트 엔진 검증 테스트 시작")
    print("=" * 60)
    
    try:
        result1 = test_basic_backtest()
        result2_full, result2_half = test_no_lookahead_bias()
        result3_no, result3_with = test_transaction_costs()
        result4 = test_buy_and_hold_comparison()
        result5 = test_final_liquidation_costs_reflected()
        
        print("\n" + "=" * 60)
        print("모든 테스트 통과 ✓")
        print("=" * 60)
        
        # 최종 성과 리포트
        print("\n[최종 성과 리포트 - SMA 교차 전략 (합성 데이터 1000일)]")
        result1.print_summary()
        
    except AssertionError as e:
        print(f"\n✗ 테스트 실패: {e}")
        raise


if __name__ == '__main__':
    main()
