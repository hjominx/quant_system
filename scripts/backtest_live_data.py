"""
실데이터 백테스트 실행

목적:
- 캐시에 다운받은 5년치 데이터로 SMA 교차 전략 백테스트
- 결과를 합성 데이터와 비교 (시스템 sanity check)
- 단순보유 BTC와 SMA 전략 객관적 비교
- 백테스트 결과 시각화·저장

대상:
- KRW-BTC 1d
- KRW-ETH 1d

실행:
    python scripts/backtest_live_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from utils.logger import setup_logging
from data.loader import load_for_backtest, get_cache_summary
from strategies.sma_crossover import SMACrossoverStrategy
from backtest.engine import BacktestEngine, BacktestConfig
from backtest.metrics import calculate_metrics


def calculate_buy_and_hold_metrics(df: pd.DataFrame, initial_capital: float):
    """단순보유 비교 기준선 계산"""
    # 첫 봉 종가에 매수해서 마지막 봉 종가까지 그대로 보유
    initial_price = df['close'].iloc[0]
    final_price = df['close'].iloc[-1]
    
    # 자본곡선: 보유 코인 수 × 일별 가격
    coins = initial_capital / initial_price
    equity_curve = df['close'] * coins
    equity_curve.name = 'equity'
    
    metrics = calculate_metrics(equity_curve)
    return metrics, equity_curve


def run_single_backtest(market: str, timeframe: str = '1d'):
    """단일 종목 백테스트 실행"""
    print(f"\n{'='*70}")
    print(f"  {market} {timeframe} — SMA 교차 전략 vs 단순보유")
    print(f"{'='*70}\n")
    
    # 1. 캐시에서 데이터 로드
    df = load_for_backtest(market, timeframe)
    print(f"데이터 로드: {len(df)}봉 ({df.index[0].date()} ~ {df.index[-1].date()})")
    
    # 2. 단순보유 기준선
    initial_capital = 10_000_000
    bh_metrics, bh_curve = calculate_buy_and_hold_metrics(df, initial_capital)
    
    # 3. SMA 교차 전략 백테스트
    strategy = SMACrossoverStrategy(params={
        'short_window': 20,
        'long_window': 60,
        'target_weight': 0.95,
    })
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=0.0025,  # 빗썸 일반 수수료
        slippage_rate=0.002,     # 보수적 슬리피지 가정
    )
    engine = BacktestEngine(config)
    result = engine.run(strategy, df, ticker=market)
    
    # 4. 결과 비교 출력
    print(f"\n{'─'*70}")
    print(f"[수익성 비교]")
    print(f"  단순보유:  {bh_metrics.total_return*100:>+8.2f}% (연환산 {bh_metrics.annualized_return*100:>+7.2f}%)")
    print(f"  SMA 전략:  {result.metrics.total_return*100:>+8.2f}% (연환산 {result.metrics.annualized_return*100:>+7.2f}%)")
    
    excess_return = result.metrics.total_return - bh_metrics.total_return
    print(f"  차이:      {excess_return*100:>+8.2f}%p ({'SMA 우세' if excess_return > 0 else 'B&H 우세'})")
    
    print(f"\n[위험 비교]")
    print(f"  단순보유 MDD: {bh_metrics.max_drawdown*100:>+8.2f}%   변동성: {bh_metrics.volatility*100:>6.2f}%")
    print(f"  SMA 전략 MDD: {result.metrics.max_drawdown*100:>+8.2f}%   변동성: {result.metrics.volatility*100:>6.2f}%")
    
    print(f"\n[위험조정수익] ← 가장 중요")
    print(f"                          단순보유    SMA전략")
    print(f"  샤프지수:               {bh_metrics.sharpe_ratio:>7.2f}    {result.metrics.sharpe_ratio:>7.2f}")
    print(f"  소르티노지수:           {bh_metrics.sortino_ratio:>7.2f}    {result.metrics.sortino_ratio:>7.2f}")
    print(f"  칼마지수:               {bh_metrics.calmar_ratio:>7.2f}    {result.metrics.calmar_ratio:>7.2f}")
    
    print(f"\n[매매 통계 (SMA 전략)]")
    print(f"  총 거래수:     {result.metrics.total_trades:>5d}회")
    print(f"  승률:          {result.metrics.win_rate*100:>5.2f}%")
    print(f"  손익비:        {result.metrics.profit_factor:>5.2f}")
    print(f"  평균 수익거래: {result.metrics.avg_win*100:>+5.2f}%")
    print(f"  평균 손실거래: {result.metrics.avg_loss*100:>+5.2f}%")
    
    # 5. 합격 여부
    print(f"\n[합격 여부]")
    is_passed = result.metrics.is_acceptable(min_sharpe=1.5, min_calmar=2.0)
    oos_passed = result.metrics.is_acceptable(min_sharpe=1.0, min_calmar=1.5)
    
    print(f"  In-Sample 기준 (Sharpe≥1.5, Calmar≥2.0):  {'✓ 통과' if is_passed else '✗ 실패'}")
    print(f"  Out-of-Sample 기준 (Sharpe≥1.0, Calmar≥1.5):  {'✓ 통과' if oos_passed else '✗ 실패'}")
    
    return {
        'market': market,
        'data_points': len(df),
        'bh_metrics': bh_metrics,
        'sma_result': result,
    }


def main():
    setup_logging(log_level='WARNING')  # 백테스트 중 로그 줄임
    
    print("="*70)
    print(" 실데이터 백테스트 — SMA 교차 전략의 진짜 성과 확인")
    print("="*70)
    
    # 캐시 상태 확인
    summary = get_cache_summary()
    if summary.empty:
        print("✗ 캐시 없음. 먼저 데이터 다운로드 필요.")
        sys.exit(1)
    
    print(f"\n캐시된 종목: {len(summary)}개")
    print(summary[['market', 'timeframe', 'rows', 'start', 'end']].to_string(index=False))
    
    # BTC, ETH 일봉 백테스트
    targets = [
        ('KRW-BTC', '1d'),
        ('KRW-ETH', '1d'),
    ]
    
    results = []
    for market, timeframe in targets:
        result = run_single_backtest(market, timeframe)
        results.append(result)
    
    # 종합 비교
    print(f"\n{'='*70}")
    print(" 종합 비교")
    print(f"{'='*70}\n")
    
    print(f"{'종목':<12} {'B&H 수익':>10} {'SMA 수익':>10} {'B&H MDD':>10} {'SMA MDD':>10} {'SMA 샤프':>10}")
    print("─" * 70)
    for r in results:
        print(f"{r['market']:<12} "
              f"{r['bh_metrics'].total_return*100:>+9.2f}% "
              f"{r['sma_result'].metrics.total_return*100:>+9.2f}% "
              f"{r['bh_metrics'].max_drawdown*100:>+9.2f}% "
              f"{r['sma_result'].metrics.max_drawdown*100:>+9.2f}% "
              f"{r['sma_result'].metrics.sharpe_ratio:>10.2f}")
    
    # 결과 해석
    print(f"\n{'='*70}")
    print(" 결과 해석")
    print(f"{'='*70}\n")
    
    print("1. SMA 교차 전략은 단순 검증용입니다 (실전 전략 아님)")
    print("   - 만약 SMA가 단순보유를 크게 이기면 백테스트 엔진 버그 의심")
    print("   - 단순보유가 더 좋게 나오는 게 정상 (5년이 강세장 포함)")
    print()
    print("2. 합격 기준 통과 못 했다면 정상입니다")
    print("   - 진짜 전략은 Phase 5에서 만들 추세추종 + 평균회귀 + 변동성타게팅")
    print("   - 그 전략들은 합격 기준 통과를 목표로 설계됨")
    print()
    print("3. MDD 비교가 핵심:")
    print("   - 단순보유 MDD는 보통 -70% 수준 (2022년 폭락장 포함)")
    print("   - SMA가 MDD를 절반 정도로 줄이는 게 추세추종의 핵심 가치")
    
    print("\n" + "="*70)
    print(" 백테스트 완료")
    print("="*70)


if __name__ == '__main__':
    main()
