"""Discord control bot pure safety tests."""
from decimal import Decimal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from notifications.discord_control_safety import (
    calculate_manual_buy_amount,
    parse_admin_user_ids,
)


def test_parse_admin_user_ids():
    print("\n[테스트 1] Discord 관리자 ID 파싱")
    print("-" * 60)

    ids = parse_admin_user_ids("111, 222 333,,")

    assert ids == {"111", "222", "333"}

    print("  ✓ 쉼표/공백 구분 관리자 ID 파싱 정상")


def test_manual_buy_amount_respects_caps():
    print("\n[테스트 2] 수동 매수 금액 한도")
    print("-" * 60)

    amount = calculate_manual_buy_amount(
        requested_amount=Decimal("25000"),
        krw_balance=Decimal("100000"),
        current_position_krw=Decimal("140000"),
        max_position_krw=Decimal("150000"),
        max_manual_order_krw=Decimal("25000"),
        min_order_krw=Decimal("5000"),
    )

    assert amount == Decimal("10000")

    print("  ✓ 총 포지션 한도까지만 수동 매수")


def test_manual_buy_rejects_too_large_request():
    print("\n[테스트 3] 수동 매수 1회 한도 초과 거부")
    print("-" * 60)

    try:
        calculate_manual_buy_amount(
            requested_amount=Decimal("30000"),
            krw_balance=Decimal("100000"),
            current_position_krw=Decimal("0"),
            max_position_krw=Decimal("150000"),
            max_manual_order_krw=Decimal("25000"),
            min_order_krw=Decimal("5000"),
        )
        raise AssertionError("1회 수동 매수 한도 초과가 통과됨")
    except RuntimeError:
        pass

    print("  ✓ 1회 수동 매수 한도 초과 거부")


def test_manual_buy_rejects_below_minimum_after_caps():
    print("\n[테스트 4] 수동 매수 최소 주문금액 미만 거부")
    print("-" * 60)

    try:
        calculate_manual_buy_amount(
            requested_amount=Decimal("25000"),
            krw_balance=Decimal("100000"),
            current_position_krw=Decimal("147000"),
            max_position_krw=Decimal("150000"),
            max_manual_order_krw=Decimal("25000"),
            min_order_krw=Decimal("5000"),
        )
        raise AssertionError("최소 주문금액 미만 수동 매수가 통과됨")
    except RuntimeError:
        pass

    print("  ✓ 한도 적용 후 최소 주문금액 미만이면 거부")


def main():
    print("=" * 60)
    print("Discord 컨트롤 봇 안전 테스트")
    print("=" * 60)

    tests = [
        test_parse_admin_user_ids,
        test_manual_buy_amount_respects_caps,
        test_manual_buy_rejects_too_large_request,
        test_manual_buy_rejects_below_minimum_after_caps,
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
