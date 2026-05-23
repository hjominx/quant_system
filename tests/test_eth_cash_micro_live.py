"""
ETH/Cash micro live runner safety tests.

The tests focus on pure safety gates and do not call Bithumb.
"""
from decimal import Decimal
from datetime import datetime, timedelta
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.models import Account, Balances
from risk.live_guard import (
    RiskGateError,
    RiskLimits,
    RiskState,
    RiskStateError,
    assert_risk_allows_action,
    create_initial_risk_state,
    evaluate_risk,
    iso_week_key,
    load_risk_state,
    roll_risk_state,
    save_risk_state,
)
from scripts.run_eth_cash_micro_live import (
    PilotState,
    assert_order_quota,
    assert_spread_ok,
    build_strategy_for_profile,
    build_pilot_plan,
    calculate_buy_amount,
    calculate_sleep_seconds,
    calculate_sell_volume,
    clock_status_alert_due,
    format_sleep_wait,
    format_cooldown,
    load_runtime_profile_mode,
    record_order,
    record_status_alert,
    resolve_runtime_profile,
    save_runtime_profile_mode,
    status_alert_due,
    strategy_params_for_profile,
    validate_live_permission,
)
from strategies.base import Signal, TradeDecision
from strategies.eth_adaptive_alpha import ETHAdaptiveAlphaStrategy
from strategies.eth_micro_scalper import ETHMicroScalperStrategy


class FakeConfig:
    """Tiny config stand-in for live permission tests."""

    def __init__(self, is_live_trading: bool):
        self.is_live_trading = is_live_trading


def make_balances(krw: str = "50000", eth: str = "0") -> Balances:
    """Create synthetic balances with Decimal fields."""
    return Balances(accounts=[
        Account(
            currency="KRW",
            balance=Decimal(krw),
            locked=Decimal("0"),
            avg_buy_price=Decimal("0"),
            avg_buy_price_modified=False,
            unit_currency="KRW",
        ),
        Account(
            currency="ETH",
            balance=Decimal(eth),
            locked=Decimal("0"),
            avg_buy_price=Decimal("3000000"),
            avg_buy_price_modified=False,
            unit_currency="KRW",
        ),
    ])


def test_live_permission_requires_confirmation_and_live_mode():
    print("\n[테스트 1] LIVE 권한 3중 확인")
    print("-" * 60)

    validate_live_permission(FakeConfig(is_live_trading=False), live=False, confirmed=False)

    try:
        validate_live_permission(FakeConfig(is_live_trading=True), live=True, confirmed=False)
        raise AssertionError("확인 플래그 없이 LIVE가 허용됨")
    except ValueError:
        pass

    try:
        validate_live_permission(FakeConfig(is_live_trading=False), live=True, confirmed=True)
        raise AssertionError("TRADING_MODE=LIVE 없이 LIVE가 허용됨")
    except ValueError:
        pass

    validate_live_permission(FakeConfig(is_live_trading=True), live=True, confirmed=True)
    print("  ✓ live + confirm + TRADING_MODE=LIVE 조합만 허용")


def test_buy_amount_respects_micro_caps():
    print("\n[테스트 2] 매수 금액 한도")
    print("-" * 60)

    amount = calculate_buy_amount(
        target_weight=1.0,
        portfolio_value_krw=Decimal("100000"),
        current_position_krw=Decimal("0"),
        krw_balance=Decimal("80000"),
        max_order_krw=Decimal("10000"),
        max_position_krw=Decimal("30000"),
        min_order_krw=Decimal("5000"),
    )
    assert amount == Decimal("10000")

    blocked = calculate_buy_amount(
        target_weight=1.0,
        portfolio_value_krw=Decimal("100000"),
        current_position_krw=Decimal("28500"),
        krw_balance=Decimal("80000"),
        max_order_krw=Decimal("10000"),
        max_position_krw=Decimal("30000"),
        min_order_krw=Decimal("5000"),
    )
    assert blocked == Decimal("0")
    print("  ✓ 주문금액/포지션/최소주문 한도 적용")


def test_sell_volume_requires_minimum_value():
    print("\n[테스트 3] 매도 최소 평가액")
    print("-" * 60)

    volume = calculate_sell_volume(
        coin_balance=Decimal("0.02"),
        trade_price=Decimal("4000000"),
        min_order_krw=Decimal("5000"),
    )
    assert volume == Decimal("0.02000000")

    blocked = calculate_sell_volume(
        coin_balance=Decimal("0.0005"),
        trade_price=Decimal("4000000"),
        min_order_krw=Decimal("5000"),
    )
    assert blocked == Decimal("0")
    print("  ✓ 최소 주문 평가액 미만 매도 차단")


def test_plan_builder_never_exceeds_caps():
    print("\n[테스트 4] 주문 계획 한도")
    print("-" * 60)

    plan = build_pilot_plan(
        decision=TradeDecision(
            signal=Signal.BUY,
            target_weight=1.0,
            reason="test buy",
        ),
        balances=make_balances(krw="100000", eth="0"),
        trade_price=Decimal("4000000"),
        market="KRW-ETH",
        max_order_krw=Decimal("10000"),
        max_position_krw=Decimal("30000"),
        min_order_krw=Decimal("5000"),
        dry_run=True,
    )
    assert plan.action == "BUY"
    assert plan.krw_amount == Decimal("10000")
    assert plan.dry_run is True

    sell_plan = build_pilot_plan(
        decision=TradeDecision(
            signal=Signal.SELL,
            target_weight=0.0,
            reason="test sell",
        ),
        balances=make_balances(krw="10000", eth="0.02"),
        trade_price=Decimal("4000000"),
        market="KRW-ETH",
        max_order_krw=Decimal("10000"),
        max_position_krw=Decimal("30000"),
        min_order_krw=Decimal("5000"),
        dry_run=True,
    )
    assert sell_plan.action == "SELL"
    assert sell_plan.volume == Decimal("0.02000000")
    print("  ✓ BUY/SELL 계획 모두 Decimal 한도 준수")


def test_spread_gate_blocks_wide_spread():
    print("\n[테스트 5] 스프레드 게이트")
    print("-" * 60)

    assert_spread_ok(Decimal("0.002"), Decimal("0.005"))
    try:
        assert_spread_ok(Decimal("0.010"), Decimal("0.005"))
        raise AssertionError("넓은 스프레드가 통과됨")
    except ValueError:
        pass
    print("  ✓ 과도한 스프레드 차단")


def test_intraday_profile_uses_4h_volatility_scaling():
    print("\n[테스트 6] 4시간봉 프로필")
    print("-" * 60)

    params = strategy_params_for_profile("intraday4h")

    assert params["periods_per_year"] == 365 * 6
    assert params["slow_window"] < 200
    assert params["trailing_stop"] < 0.20

    print("  ✓ 4시간봉 연환산/윈도우/트레일링 설정")


def test_aggressive_profile_allows_dip_probe():
    print("\n[테스트 7] 공격형 4시간봉 프로필")
    print("-" * 60)

    params = strategy_params_for_profile("aggressive4h")

    assert params["periods_per_year"] == 365 * 6
    assert params["dip_probe_enabled"] is True
    assert params["use_exit_ma"] is False
    assert params["max_position"] == 0.70
    assert params["dip_max_position"] == 0.55

    print("  ✓ 저점 탐색 진입 활성화")
    print("  ✓ 공격형 비중/청산 설정 적용")


def test_stable_quality_profile_uses_adaptive_alpha():
    print("\n[테스트 8] 안정형 적응 알파 프로필")
    print("-" * 60)

    strategy = build_strategy_for_profile("stable_quality")

    assert isinstance(strategy, ETHAdaptiveAlphaStrategy)
    assert strategy.params["periods_per_year"] == 365 * 6
    assert strategy.params["max_realized_volatility"] == 0.90
    assert strategy.params["hard_stop_pct"] == 0.035
    assert strategy.params["max_position"] == 0.55

    print("  ✓ stable_quality가 ETHAdaptiveAlphaStrategy로 연결됨")
    print("  ✓ 리서치 프리셋의 방어형 파라미터 적용")


def test_scalp_active_profile_uses_micro_scalper():
    print("\n[테스트 9] 스캘핑 프로필")
    print("-" * 60)

    strategy = build_strategy_for_profile("scalp_active")

    assert isinstance(strategy, ETHMicroScalperStrategy)
    assert strategy.params["periods_per_year"] == 365 * 24 * 12
    assert strategy.params["take_profit_pct"] == 0.0060
    assert strategy.params["stop_loss_pct"] == 0.0045
    assert strategy.params["max_position"] == 0.85

    print("  ✓ scalp_active가 ETHMicroScalperStrategy로 연결됨")
    print("  ✓ 5분봉 스캘핑 파라미터 적용")


def test_scalp_1m_profile_uses_micro_scalper():
    print("\n[테스트 10] 1분봉 스캘핑 프로필")
    print("-" * 60)

    strategy = build_strategy_for_profile("scalp_1m")

    assert isinstance(strategy, ETHMicroScalperStrategy)
    assert strategy.params["periods_per_year"] == 365 * 24 * 60
    assert strategy.params["take_profit_pct"] == 0.0040
    assert strategy.params["stop_loss_pct"] == 0.0030
    assert strategy.params["max_position"] == 0.85

    print("  ✓ scalp_1m이 ETHMicroScalperStrategy로 연결됨")
    print("  ✓ 1분봉 초공격 스캘핑 파라미터 적용")


def test_runtime_profile_mode_file_switches_stable_and_scalp():
    print("\n[테스트 11] 런타임 프로필 파일")
    print("-" * 60)

    class Args:
        profile = "stable_quality"
        timeframe = "4h"
        candle_count = 200

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "profile.json"
        fallback = resolve_runtime_profile(Args, path)
        assert fallback.profile == "stable_quality"
        assert fallback.timeframe == "4h"
        assert fallback.source == "service args"

        save_runtime_profile_mode(path, "scalp_active", updated_by="test")
        assert load_runtime_profile_mode(path) == "scalp_active"
        scalp = resolve_runtime_profile(Args, path)
        assert scalp.profile == "scalp_active"
        assert scalp.timeframe == "5m"
        assert scalp.candle_count == 200

        save_runtime_profile_mode(path, "stable_quality", updated_by="test")
        stable = resolve_runtime_profile(Args, path)
        assert stable.profile == "stable_quality"
        assert stable.timeframe == "4h"

        save_runtime_profile_mode(path, "scalp_1m", updated_by="test")
        scalp_1m = resolve_runtime_profile(Args, path)
        assert scalp_1m.profile == "scalp_1m"
        assert scalp_1m.timeframe == "1m"

    print("  ✓ Discord 모드 파일로 stable/scalp 전환 가능")


def test_order_quota_blocks_cooldown_and_daily_limit():
    print("\n[테스트 12] 주문 쿨다운/일일 한도")
    print("-" * 60)

    now = datetime(2026, 5, 22, 9, 0, 0)
    fresh = PilotState(date=now.date().isoformat())
    assert_order_quota(
        state=fresh,
        now=now,
        max_orders_per_day=4,
        cooldown_minutes=120,
    )

    after_order = record_order(fresh, now)
    assert after_order.orders_today == 1

    try:
        assert_order_quota(
            state=after_order,
            now=now + timedelta(minutes=30),
            max_orders_per_day=4,
            cooldown_minutes=120,
        )
        raise AssertionError("쿨다운 중 주문이 허용됨")
    except RuntimeError:
        pass

    maxed = PilotState(
        date=now.date().isoformat(),
        orders_today=4,
        last_order_at=(now - timedelta(hours=5)).isoformat(),
    )
    try:
        assert_order_quota(
            state=maxed,
            now=now,
            max_orders_per_day=4,
            cooldown_minutes=120,
        )
        raise AssertionError("일일 한도 초과 주문이 허용됨")
    except RuntimeError:
        pass

    print("  ✓ 쿨다운과 일일 주문 한도 적용")


def test_order_quota_supports_second_cooldown():
    print("\n[테스트 13] 초 단위 주문 쿨다운")
    print("-" * 60)

    now = datetime(2026, 5, 22, 9, 0, 0)
    state = PilotState(
        date=now.date().isoformat(),
        orders_today=1,
        last_order_at=now.isoformat(),
    )

    try:
        assert_order_quota(
            state=state,
            now=now + timedelta(seconds=5),
            max_orders_per_day=300,
            cooldown_minutes=15,
            cooldown_seconds=10,
        )
        raise AssertionError("10초 쿨다운 중 주문이 허용됨")
    except RuntimeError as exc:
        assert "초" in str(exc)

    assert_order_quota(
        state=state,
        now=now + timedelta(seconds=10),
        max_orders_per_day=300,
        cooldown_minutes=15,
        cooldown_seconds=10,
    )
    assert format_cooldown(15, 10) == "10s"

    print("  ✓ 10초 쿨다운 적용")


def test_record_order_tracks_averaging_count():
    print("\n[테스트 14] 추매 카운트 저장")
    print("-" * 60)

    now = datetime(2026, 5, 22, 9, 0, 0)
    state = PilotState(date=now.date().isoformat())

    entry = record_order(state, now, action="BUY")
    assert entry.averaging_count == 0

    first_add = record_order(
        entry,
        now + timedelta(minutes=1),
        action="BUY",
        is_averaging=True,
    )
    assert first_add.averaging_count == 1

    second_add = record_order(
        first_add,
        now + timedelta(minutes=2),
        action="BUY",
        is_averaging=True,
    )
    assert second_add.averaging_count == 2

    exit_state = record_order(
        second_add,
        now + timedelta(minutes=3),
        action="SELL",
    )
    assert exit_state.averaging_count == 0

    print("  ✓ 일반 매수는 유지, 추매는 증가, 매도는 리셋")


def test_risk_state_initializes_and_rolls_periods():
    print("\n[테스트 8] 위험 상태 기준값 초기화/리셋")
    print("-" * 60)

    now = datetime(2026, 5, 22, 9, 0, 0)
    state = create_initial_risk_state(now, Decimal("100000"))
    assert state.date == "2026-05-22"
    assert state.iso_week == iso_week_key(now)
    assert state.day_start_equity == Decimal("100000")
    assert state.week_start_equity == Decimal("100000")
    assert state.peak_equity == Decimal("100000")

    old = RiskState(
        date="2026-05-21",
        iso_week=iso_week_key(now),
        day_start_equity=Decimal("120000"),
        week_start_equity=Decimal("150000"),
        peak_equity=Decimal("150000"),
    )
    rolled = roll_risk_state(old, now, Decimal("100000"))
    assert rolled.day_start_equity == Decimal("100000")
    assert rolled.week_start_equity == Decimal("150000")
    assert rolled.peak_equity == Decimal("150000")

    next_week = datetime(2026, 5, 25, 9, 0, 0)
    rolled_week = roll_risk_state(rolled, next_week, Decimal("90000"))
    assert rolled_week.week_start_equity == Decimal("90000")

    print("  ✓ 일/주 기준값과 최고자산 추적 정상")


def test_daily_loss_blocks_buy_but_allows_sell():
    print("\n[테스트 9] 일일 손실 한도")
    print("-" * 60)

    limits = RiskLimits(
        daily_loss_limit=Decimal("0.03"),
        weekly_loss_limit=Decimal("0.07"),
        max_drawdown_limit=Decimal("0.15"),
    )
    state = RiskState(
        date="2026-05-22",
        iso_week="2026-W21",
        day_start_equity=Decimal("100000"),
        week_start_equity=Decimal("100000"),
        peak_equity=Decimal("100000"),
    )
    report = evaluate_risk(state=state, equity=Decimal("96000"), limits=limits)

    assert report.daily_breached is True
    try:
        assert_risk_allows_action("BUY", report)
        raise AssertionError("일일 손실 초과 후 매수가 허용됨")
    except RiskGateError:
        pass
    assert_risk_allows_action("SELL", report)

    print("  ✓ 일일 손실 초과 시 신규 매수만 차단")


def test_weekly_loss_blocks_buy():
    print("\n[테스트 10] 주간 손실 한도")
    print("-" * 60)

    limits = RiskLimits(
        daily_loss_limit=Decimal("0.03"),
        weekly_loss_limit=Decimal("0.07"),
        max_drawdown_limit=Decimal("0.15"),
    )
    state = RiskState(
        date="2026-05-22",
        iso_week="2026-W21",
        day_start_equity=Decimal("94000"),
        week_start_equity=Decimal("100000"),
        peak_equity=Decimal("100000"),
    )
    report = evaluate_risk(state=state, equity=Decimal("92000"), limits=limits)

    assert report.weekly_breached is True
    try:
        assert_risk_allows_action("BUY", report)
        raise AssertionError("주간 손실 초과 후 매수가 허용됨")
    except RiskGateError:
        pass
    assert_risk_allows_action("SELL", report)

    print("  ✓ 주간 손실 초과 시 신규 매수 차단")


def test_mdd_breach_blocks_buy_and_sell():
    print("\n[테스트 11] MDD 한도")
    print("-" * 60)

    limits = RiskLimits(
        daily_loss_limit=Decimal("0.03"),
        weekly_loss_limit=Decimal("0.07"),
        max_drawdown_limit=Decimal("0.15"),
    )
    state = RiskState(
        date="2026-05-22",
        iso_week="2026-W21",
        day_start_equity=Decimal("90000"),
        week_start_equity=Decimal("90000"),
        peak_equity=Decimal("120000"),
    )
    report = evaluate_risk(state=state, equity=Decimal("100000"), limits=limits)

    assert report.drawdown_breached is True
    for action in ("BUY", "SELL"):
        try:
            assert_risk_allows_action(action, report)
            raise AssertionError(f"MDD 초과 후 {action}이 허용됨")
        except RiskGateError:
            pass

    print("  ✓ MDD 초과 시 모든 실주문 차단")


def test_kill_switch_blocks_all_orders():
    print("\n[테스트 12] 킬스위치")
    print("-" * 60)

    limits = RiskLimits(
        daily_loss_limit=Decimal("0.03"),
        weekly_loss_limit=Decimal("0.07"),
        max_drawdown_limit=Decimal("0.15"),
    )
    state = RiskState(
        date="2026-05-22",
        iso_week="2026-W21",
        day_start_equity=Decimal("100000"),
        week_start_equity=Decimal("100000"),
        peak_equity=Decimal("100000"),
    )
    report = evaluate_risk(
        state=state,
        equity=Decimal("100000"),
        limits=limits,
        kill_switch_file_active=True,
    )

    for action in ("BUY", "SELL"):
        try:
            assert_risk_allows_action(action, report)
            raise AssertionError(f"킬스위치 중 {action}이 허용됨")
        except RiskGateError:
            pass

    print("  ✓ 킬스위치 활성 시 모든 주문 차단")


def test_risk_state_round_trip():
    print("\n[테스트 13] 위험 상태 저장/로드")
    print("-" * 60)

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "risk_state.json"
        now = datetime(2026, 5, 22, 9, 0, 0)
        state = RiskState(
            date="2026-05-22",
            iso_week=iso_week_key(now),
            day_start_equity=Decimal("100000"),
            week_start_equity=Decimal("120000"),
            peak_equity=Decimal("150000"),
            kill_switch=True,
        )
        save_risk_state(path, state)
        loaded = load_risk_state(path, now, Decimal("110000"))

        assert loaded.day_start_equity == Decimal("100000")
        assert loaded.week_start_equity == Decimal("120000")
        assert loaded.peak_equity == Decimal("150000")
        assert loaded.kill_switch is True

    print("  ✓ Decimal 상태 저장/로드 정상")


def test_corrupt_risk_state_fails_closed():
    print("\n[테스트 14] 위험 상태 손상 시 fail-closed")
    print("-" * 60)

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "risk_state.json"
        path.write_text("{not valid json", encoding="utf-8")
        try:
            load_risk_state(path, datetime(2026, 5, 22, 9, 0, 0), Decimal("100000"))
            raise AssertionError("손상된 위험 상태가 통과됨")
        except RiskStateError:
            pass

    print("  ✓ 위험 상태 파일 손상 시 주문 전 차단")


def test_status_alert_interval_state():
    print("\n[테스트 15] Discord 상태 알림 주기")
    print("-" * 60)

    now = datetime(2026, 5, 22, 9, 0, 0)
    state = PilotState(date="2026-05-22", orders_today=2)

    assert status_alert_due(state, now, 30) is True
    updated = record_status_alert(state, now)
    assert updated.orders_today == 2
    assert updated.last_status_alert_at == now.isoformat(timespec="seconds")
    assert status_alert_due(updated, now + timedelta(minutes=29), 30) is False
    assert status_alert_due(updated, now + timedelta(minutes=30), 30) is True

    ordered = record_order(updated, now + timedelta(minutes=31))
    assert ordered.last_status_alert_at == updated.last_status_alert_at
    assert ordered.orders_today == 3

    print("  ✓ 상태 알림 주기와 주문 카운터 보존 정상")


def test_hourly_status_alert_and_clock_aligned_sleep():
    print("\n[테스트 16] 정각 상태 알림과 시계 정렬 대기")
    print("-" * 60)

    state = PilotState(date="2026-05-23", orders_today=0)
    before_hour = datetime(2026, 5, 23, 9, 45, 0)
    on_hour = datetime(2026, 5, 23, 10, 0, 4)
    same_hour = record_status_alert(state, on_hour)

    assert clock_status_alert_due(state, before_hour, 60) is False
    assert clock_status_alert_due(state, on_hour, 60) is True
    assert clock_status_alert_due(same_hour, on_hour.replace(second=50), 60) is False
    assert clock_status_alert_due(same_hour, datetime(2026, 5, 23, 11, 0, 1), 60) is True

    assert calculate_sleep_seconds(
        now=datetime(2026, 5, 23, 9, 20, 41),
        interval_minutes=15,
        align_to_clock=True,
    ) == 559
    assert calculate_sleep_seconds(
        now=datetime(2026, 5, 23, 9, 20, 41),
        interval_minutes=15,
        align_to_clock=False,
    ) == 900
    assert format_sleep_wait(559) == "9.3분"

    print("  ✓ 정각 알림은 00분 사이클에서만 발생")
    print("  ✓ 15분 루프가 다음 00/15/30/45분 경계로 정렬됨")


def main():
    print("=" * 60)
    print("ETH/Cash Micro Live 안전 테스트")
    print("=" * 60)

    tests = [
        test_live_permission_requires_confirmation_and_live_mode,
        test_buy_amount_respects_micro_caps,
        test_sell_volume_requires_minimum_value,
        test_plan_builder_never_exceeds_caps,
        test_spread_gate_blocks_wide_spread,
        test_intraday_profile_uses_4h_volatility_scaling,
        test_aggressive_profile_allows_dip_probe,
        test_stable_quality_profile_uses_adaptive_alpha,
        test_scalp_active_profile_uses_micro_scalper,
        test_scalp_1m_profile_uses_micro_scalper,
        test_runtime_profile_mode_file_switches_stable_and_scalp,
        test_order_quota_blocks_cooldown_and_daily_limit,
        test_order_quota_supports_second_cooldown,
        test_record_order_tracks_averaging_count,
        test_risk_state_initializes_and_rolls_periods,
        test_daily_loss_blocks_buy_but_allows_sell,
        test_weekly_loss_blocks_buy,
        test_mdd_breach_blocks_buy_and_sell,
        test_kill_switch_blocks_all_orders,
        test_risk_state_round_trip,
        test_corrupt_risk_state_fails_closed,
        test_status_alert_interval_state,
        test_hourly_status_alert_and_clock_aligned_sleep,
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
