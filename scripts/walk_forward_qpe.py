"""
QPE(Quantum Probability Engine) Walk-Forward 검증

목적:
- SMA 기준선을 넘어서는 시장상태 적응형 복합 알파 엔진 검증
- 각 학습 구간에서 QPE 임계값/위험 파라미터를 고르고 다음 OOS 구간만 평가

실행:
    python scripts/walk_forward_qpe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logging
from data.loader import load_for_backtest, get_cache_summary
from strategies.quantum_probability_engine import QuantumProbabilityEngineStrategy
from backtest.engine import BacktestConfig
from backtest.walk_forward import WalkForwardConfig, WalkForwardEngine


PARAMETER_GRID = {
    'entry_threshold': [0.34, 0.42, 0.50],
    'exit_threshold': [0.20, 0.28],
    'crash_entry_threshold': [0.36, 0.48],
    'crash_exit_threshold': [0.56],
    'max_position': [0.70, 0.90],
    'min_position': [0.03],
    'target_volatility': [0.35],
    'uncertainty_penalty': [0.45],
    'crash_penalty': [0.80],
}


def make_qpe_strategy(params):
    return QuantumProbabilityEngineStrategy(params=params)


def run_single_walk_forward(market: str, timeframe: str = '1d'):
    print(f"\n{'='*90}")
    print(f"  {market} {timeframe} — QPE Walk-Forward 검증")
    print(f"{'='*90}\n")

    df = load_for_backtest(market, timeframe)
    print(f"데이터 로드: {len(df)}봉 ({df.index[0].date()} ~ {df.index[-1].date()})")

    wf_config = WalkForwardConfig(
        initial_train_years=1,
        test_years=1,
        step_years=1,
        mode='expanding',
        objective='qpe_composite_score',
        min_train_bars=300,
        min_test_bars=180,
        allow_partial_final_window=True,
    )
    backtest_config = BacktestConfig(
        initial_capital=10_000_000,
        commission_rate=0.0025,
        slippage_rate=0.002,
    )

    engine = WalkForwardEngine(
        strategy_factory=make_qpe_strategy,
        backtest_config=backtest_config,
        config=wf_config,
    )
    result = engine.run(df, ticker=market, parameter_grid=PARAMETER_GRID)

    print(result.summary())

    print("\n[윈도우별 OOS 결과]")
    print(
        f"{'ID':>2} {'Train':<23} {'Test':<23} "
        f"{'Best Params':<58} {'Score':>7} {'OOS Ret':>9} {'MDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Trades':>6}"
    )
    print("─" * 160)

    for _, row in result.windows_dataframe().iterrows():
        params = row['best_params']
        param_text = (
            f"E{params['entry_threshold']:.2f}/X{params['exit_threshold']:.2f}/"
            f"CE{params['crash_entry_threshold']:.2f}/CX{params['crash_exit_threshold']:.2f}/"
            f"W{params['max_position']:.2f}/U{params['uncertainty_penalty']:.2f}"
        )
        train_range = f"{row['train_start'].date()}~{row['train_end'].date()}"
        test_range = f"{row['test_start'].date()}~{row['test_end'].date()}"
        print(
            f"{int(row['window_id']):>2} "
            f"{train_range:<23} {test_range:<23} "
            f"{param_text:<58} "
            f"{row['objective_value']:>7.3f} "
            f"{row['oos_return']*100:>+8.2f}% "
            f"{row['oos_mdd']*100:>+7.2f}% "
            f"{row['oos_sharpe']:>7.2f} "
            f"{row['oos_calmar']:>7.2f} "
            f"{int(row['oos_trades']):>6d}"
        )

    return result


def main():
    setup_logging(log_level='WARNING')

    print("=" * 90)
    print(" QPE Walk-Forward 검증 — 시장상태 적응형 복합 알파 엔진")
    print("=" * 90)

    summary = get_cache_summary()
    if summary.empty:
        print("✗ 캐시 없음. 먼저 scripts/download_historical.py 실행 필요.")
        sys.exit(1)

    targets = [
        ('KRW-BTC', '1d'),
        ('KRW-ETH', '1d'),
    ]

    results = []
    for market, timeframe in targets:
        results.append(run_single_walk_forward(market, timeframe))

    print(f"\n{'='*90}")
    print(" QPE 종합 OOS 비교")
    print(f"{'='*90}\n")
    print(f"{'종목':<12} {'윈도우':>6} {'OOS 수익':>10} {'OOS MDD':>10} {'Sharpe':>8} {'Calmar':>8} {'거래수':>8} {'통과율':>8}")
    print("─" * 90)

    for (market, _), result in zip(targets, results):
        m = result.oos_metrics
        print(
            f"{market:<12} "
            f"{len(result.windows):>6d} "
            f"{m.total_return*100:>+9.2f}% "
            f"{m.max_drawdown*100:>+9.2f}% "
            f"{m.sharpe_ratio:>8.2f} "
            f"{m.calmar_ratio:>8.2f} "
            f"{m.total_trades:>8d} "
            f"{result.pass_rate*100:>7.2f}%"
        )

    print("\n해석:")
    print("  - QPE는 연구용 첫 버전입니다. 통과율과 OOS MDD가 핵심입니다.")
    print("  - 좋은 구간만 고르는 것이 아니라, 나쁜 구간에서 덜 깨지는지가 중요합니다.")
    print("  - 다음 단계는 다종목 포트폴리오와 EntanglementRiskManager입니다.")


if __name__ == '__main__':
    main()
