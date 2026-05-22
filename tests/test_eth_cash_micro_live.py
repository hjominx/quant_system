"""
ETH/Cash micro live runner safety tests.

The tests focus on pure safety gates and do not call Bithumb.
"""
from decimal import Decimal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.models import Account, Balances
from scripts.run_eth_cash_micro_live import (
    assert_spread_ok,
    build_pilot_plan,
    calculate_buy_amount,
    calculate_sell_volume,
    validate_live_permission,
)
from strategies.base import Signal, TradeDecision


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
