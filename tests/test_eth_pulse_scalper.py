"""ETH pulse scalper strategy tests."""
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import Signal, StrategyContext
from strategies.eth_pulse_scalper import PulseScalperStrategy


def make_context(
    *,
    now: datetime,
    trade_price: float = 3_000_000.0,
    current_position: float = 0.0,
    avg_buy_price: float = 0.0,
    last_pulse_buy_at: datetime | None = None,
    last_order_at: datetime | None = None,
    last_action: str | None = None,
    closes: list[float] | None = None,
    spread_pct: float = 0.00005,
    cash: float = 200_000.0,
    averaging_count: int = 0,
    min_order_krw: float = 5_000.0,
    tick_size_krw: float = 0.01,
) -> StrategyContext:
    """Create a tiny live-like context using ticker price metadata."""
    values = closes or [trade_price]
    index = [now - timedelta(minutes=len(values) - idx - 1) for idx in range(len(values))]
    ohlcv = pd.DataFrame(
        {
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": [1.0 for _ in values],
        },
        index=index,
    )
    return StrategyContext(
        timestamp=now,
        ticker="KRW-XRP",
        ohlcv=ohlcv,
        current_position=current_position,
        avg_buy_price=avg_buy_price,
        cash=cash,
        portfolio_value=200_000.0,
        metadata={
            "now": now.isoformat(timespec="seconds"),
            "trade_price": trade_price,
            "last_pulse_buy_at": (
                last_pulse_buy_at.isoformat(timespec="seconds")
                if last_pulse_buy_at
                else None
            ),
            "last_order_at": (
                last_order_at.isoformat(timespec="seconds")
                if last_order_at
                else None
            ),
            "last_action": last_action,
            "spread_pct": spread_pct,
            "tick_size_krw": tick_size_krw,
            "averaging_count": averaging_count,
            "min_order_krw": min_order_krw,
        },
    )


def night_strategy() -> PulseScalperStrategy:
    return PulseScalperStrategy(
        params={
            "label": "night micro",
            "buy_interval_seconds": 0,
            "min_exit_hold_seconds": 0,
            "take_profit_pct": 0.0010,
            "take_profit_min_pct": 0.0010,
            "take_profit_max_pct": 0.0010,
            "take_profit_vol_multiplier": 1.0,
            "stop_loss_pct": 0.0070,
            "max_position": 1.0,
            "add_while_holding": False,
            "averaging_enabled": True,
            "averaging_trigger_pct": 0.0020,
            "averaging_second_trigger_pct": 0.0035,
            "averaging_order_krw": 20000,
            "averaging_max_adds": 2,
            "tick_mode_enabled": True,
            "take_profit_min_ticks": 10,
            "averaging_min_ticks": 10,
            "averaging_second_min_ticks": 18,
            "stop_loss_min_ticks": 30,
            "tick_band_min_10tick_pct": 0.0010,
            "tick_band_max_10tick_pct": 0.0060,
            "entry_signal_required": True,
            "min_micro_momentum": 0.00002,
            "min_micro_volatility": 0.00002,
            "momentum_lookback": 3,
            "volatility_window": 8,
            "reversal_entry_enabled": True,
            "reversal_lookback": 8,
            "reversal_min_rebound": 0.00020,
            "breakout_tolerance": 0.0060,
            "min_reentry_seconds": 3,
            "positive_ev_required": True,
            "max_entry_spread_pct": 0.00045,
            "fee_round_trip_pct": 0.0008,
            "rebate_round_trip_pct": 0.0008,
            "slippage_buffer_pct": 0.0002,
            "min_expected_net_pct": 0.0,
            "min_take_profit_after_cost_pct": 0.0005,
            "min_reward_to_risk": 0.15,
        }
    )


def test_pulse_buys_when_no_prior_buy():
    print("\n[테스트 1] 첫 10분 pulse 매수")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    decision = strategy.generate_signal(make_context(now=datetime(2026, 5, 23, 9, 0, 0)))

    assert decision.signal == Signal.BUY
    assert decision.target_weight == 1.0
    assert "pulse entry" in decision.reason
    print("  ✓ 이전 매수 기록이 없으면 즉시 10만원 pulse 시작")


def test_pulse_waits_until_ten_minutes():
    print("\n[테스트 2] 10분 간격 강제")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    now = datetime(2026, 5, 23, 9, 5, 0)
    decision = strategy.generate_signal(
        make_context(
            now=now,
            last_pulse_buy_at=now - timedelta(minutes=5),
        )
    )

    assert decision.signal == Signal.HOLD
    assert "cash wait" in decision.reason
    print("  ✓ 마지막 pulse 이후 10분 전에는 추가 매수 없음")


def test_pulse_adds_every_ten_minutes_even_while_holding():
    print("\n[테스트 3] 보유 중 10분마다 추가 pulse")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    now = datetime(2026, 5, 23, 9, 10, 0)
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=3_000_500.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(minutes=10),
        )
    )

    assert decision.signal == Signal.BUY
    assert "pulse add" in decision.reason
    print("  ✓ 손익이 TP/SL 사이면 10분마다 10만원 추가 pulse")


def test_pulse_takes_profit_from_live_ticker():
    print("\n[테스트 4] 3초 이후 +0.3% 익절")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    now = datetime(2026, 5, 23, 9, 0, 3)
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=3_009_100.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(seconds=3),
        )
    )

    assert decision.signal == Signal.SELL
    assert "take-profit" in decision.reason
    print("  ✓ 실시간 ticker 기준 +0.3% 도달 시 전량 매도")


def test_pulse_stops_loss_from_live_ticker():
    print("\n[테스트 5] -0.04% 손절")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    now = datetime(2026, 5, 23, 9, 0, 3)
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=2_998_700.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(seconds=3),
        )
    )

    assert decision.signal == Signal.SELL
    assert "stop-loss" in decision.reason
    print("  ✓ 실시간 ticker 기준 -0.04% 도달 시 전량 매도")


def test_pulse_does_not_exit_before_three_seconds():
    print("\n[테스트 6] 첫 3초 보호 구간")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    now = datetime(2026, 5, 23, 9, 0, 2)
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=3_009_000.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(seconds=2),
        )
    )

    assert decision.signal == Signal.HOLD
    assert "monitor" in decision.reason
    print("  ✓ 매수 후 3초 전에는 TP/SL 청산을 시작하지 않음")


def test_night_micro_requires_micro_edge_before_entry():
    print("\n[테스트 7] Night Micro 신호 기반 진입")
    print("-" * 60)

    strategy = night_strategy()
    now = datetime(2026, 5, 23, 9, 0, 0)
    flat = [100.0 for _ in range(10)]
    decision = strategy.generate_signal(
        make_context(now=now, trade_price=100.0, closes=flat)
    )

    assert decision.signal == Signal.HOLD
    assert "entry wait" in decision.reason

    trending = [
        100.00, 100.03, 100.01, 100.05, 100.09,
        100.07, 100.12, 100.18, 100.23, 100.30,
    ]
    decision = strategy.generate_signal(
        make_context(now=now, trade_price=100.30, closes=trending, tick_size_krw=0.02)
    )

    assert decision.signal == Signal.BUY
    assert "pulse entry" in decision.reason
    assert "EV=" in decision.reason
    print("  ✓ 무조건 진입이 아니라 1분봉 변동성/모멘텀 조건 통과 시에만 진입")


def test_night_micro_blocks_wide_spread_entry():
    print("\n[테스트 8] Night Micro 스프레드 차단")
    print("-" * 60)

    strategy = night_strategy()
    now = datetime(2026, 5, 23, 9, 0, 0)
    trending = [
        100.00, 100.03, 100.01, 100.05, 100.09,
        100.07, 100.12, 100.18, 100.23, 100.30,
    ]
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=100.30,
            closes=trending,
            spread_pct=0.00050,
            tick_size_krw=0.02,
        )
    )

    assert decision.signal == Signal.HOLD
    assert "spread too wide" in decision.reason
    print("  ✓ 신호가 있어도 스프레드가 넓으면 진입하지 않음")


def test_night_micro_requires_positive_post_rebate_ev():
    print("\n[테스트 9] Night Micro 페이백 감안 기대값")
    print("-" * 60)

    strategy = night_strategy()
    strategy.params["min_expected_net_pct"] = 0.0020
    now = datetime(2026, 5, 23, 9, 0, 0)
    trending = [
        100.00, 100.03, 100.01, 100.05, 100.09,
        100.07, 100.12, 100.18, 100.23, 100.30,
    ]
    decision = strategy.generate_signal(
        make_context(now=now, trade_price=100.30, closes=trending, tick_size_krw=0.02)
    )

    assert decision.signal == Signal.HOLD
    assert "expected EV too low" in decision.reason
    print("  ✓ 페이백을 감안해도 기대값 버퍼가 부족하면 진입하지 않음")


def test_night_micro_take_profit_and_hard_stop():
    print("\n[테스트 10] Night Micro 틱 기반 익절/하드 손절")
    print("-" * 60)

    strategy = night_strategy()
    now = datetime(2026, 5, 23, 9, 0, 1)
    closes = [
        100.00, 100.06, 100.02, 100.10, 100.15,
        100.11, 100.20, 100.28, 100.36, 100.44,
    ]
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=100.15,
            current_position=10.0,
            avg_buy_price=100.0,
            last_pulse_buy_at=now - timedelta(seconds=1),
            closes=closes,
        )
    )

    assert decision.signal == Signal.SELL
    assert "take-profit" in decision.reason
    assert decision.take_profit_pct == 0.0010

    eth_hold = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=3_003_000.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(seconds=1),
            closes=[3_000_000.0 + idx * 500 for idx in range(10)],
            tick_size_krw=1000.0,
        )
    )
    assert eth_hold.signal == Signal.HOLD

    eth_take_profit = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=3_010_000.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(seconds=1),
            closes=[3_000_000.0 + idx * 500 for idx in range(10)],
            tick_size_krw=1000.0,
        )
    )
    assert eth_take_profit.signal == Signal.SELL
    assert eth_take_profit.take_profit_pct > 0.003

    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=99.29,
            current_position=10.0,
            avg_buy_price=100.0,
            last_pulse_buy_at=now - timedelta(seconds=1),
            closes=closes,
        )
    )

    assert decision.signal == Signal.SELL
    assert "stop-loss" in decision.reason
    print("  ✓ 익절은 +0.10%와 10틱을 모두 넘을 때, 하드 손절은 -0.70%와 30틱 기준으로 작동")


def test_night_micro_averages_after_tick_drawdown():
    print("\n[테스트 11] Night Micro 틱 기반 추매")
    print("-" * 60)

    strategy = night_strategy()
    now = datetime(2026, 5, 23, 9, 0, 1)
    closes = [
        100.00, 100.06, 100.02, 100.10, 100.15,
        100.11, 100.20, 100.28, 100.36, 100.44,
    ]
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=99.79,
            current_position=10.0,
            avg_buy_price=100.0,
            last_pulse_buy_at=now - timedelta(seconds=1),
            closes=closes,
        )
    )

    assert decision.signal == Signal.BUY
    assert "controlled averaging" in decision.reason
    assert decision.order_krw == 20000

    second_add = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=2_980_000.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(seconds=1),
            closes=[3_000_000.0 - idx * 1000 for idx in range(10)],
            tick_size_krw=1000.0,
            averaging_count=1,
        )
    )
    assert second_add.signal == Signal.BUY

    max_adds_exit = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=2_980_000.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(seconds=1),
            closes=[3_000_000.0 - idx * 1000 for idx in range(10)],
            tick_size_krw=1000.0,
            averaging_count=2,
        )
    )
    assert max_adds_exit.signal == Signal.SELL
    assert "stop auto trade" in max_adds_exit.reason
    print("  ✓ 1차는 10틱, 2차는 18틱 기준으로 최대 2회까지만 추매")


def test_night_micro_sells_and_stops_when_averaging_cash_unavailable():
    print("\n[테스트 12] Night Micro 추매 현금 부족 시 전량매도 종료")
    print("-" * 60)

    strategy = night_strategy()
    now = datetime(2026, 5, 23, 9, 0, 1)
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=99.79,
            current_position=10.0,
            avg_buy_price=100.0,
            last_pulse_buy_at=now - timedelta(seconds=1),
            closes=[100.00, 100.06, 100.02, 100.10, 100.15, 100.11, 100.20, 100.28, 100.36, 100.44],
            cash=4_000.0,
        )
    )

    assert decision.signal == Signal.SELL
    assert "stop auto trade" in decision.reason
    print("  ✓ 추매할 현금이 최소 주문금액보다 작으면 전량매도 후 자동거래 종료 신호")


def test_night_micro_reentry_guard_after_sell():
    print("\n[테스트 13] Night Micro 재진입 가드")
    print("-" * 60)

    strategy = night_strategy()
    now = datetime(2026, 5, 23, 9, 0, 1)
    trending = [
        100.00, 100.03, 100.01, 100.05, 100.09,
        100.07, 100.12, 100.18, 100.23, 100.30,
    ]
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=100.30,
            closes=trending,
            last_order_at=now - timedelta(seconds=1),
            last_action="SELL",
        )
    )

    assert decision.signal == Signal.HOLD
    assert "reentry guard" in decision.reason
    print("  ✓ 매도 직후 즉시 재매수 반복을 막아 과도한 왕복 패턴을 줄임")


def main():
    print("=" * 60)
    print("Pulse Scalper 테스트")
    print("=" * 60)

    tests = [
        test_pulse_buys_when_no_prior_buy,
        test_pulse_waits_until_ten_minutes,
        test_pulse_adds_every_ten_minutes_even_while_holding,
        test_pulse_takes_profit_from_live_ticker,
        test_pulse_stops_loss_from_live_ticker,
        test_pulse_does_not_exit_before_three_seconds,
        test_night_micro_requires_micro_edge_before_entry,
        test_night_micro_blocks_wide_spread_entry,
        test_night_micro_requires_positive_post_rebate_ev,
        test_night_micro_take_profit_and_hard_stop,
        test_night_micro_averages_after_tick_drawdown,
        test_night_micro_sells_and_stops_when_averaging_cash_unavailable,
        test_night_micro_reentry_guard_after_sell,
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
            failed += 1

    print("\n" + "=" * 60)
    print(f"결과: {passed}개 통과 / {failed}개 실패 (총 {len(tests)}개)")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
