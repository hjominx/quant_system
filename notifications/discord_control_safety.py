"""Pure safety helpers for Discord manual trade controls."""
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Optional

from notifications.discord_status import format_krw


KRW_QUANT = Decimal("1")


def parse_admin_user_ids(raw: Optional[str]) -> set[str]:
    """Parse comma/space separated Discord user IDs."""
    if not raw:
        return set()
    return {
        chunk.strip()
        for item in raw.split(",")
        for chunk in item.split()
        if chunk.strip()
    }


def calculate_manual_buy_amount(
    *,
    requested_amount: Decimal,
    krw_balance: Decimal,
    current_position_krw: Decimal,
    max_position_krw: Decimal,
    max_manual_order_krw: Decimal,
    min_order_krw: Decimal,
) -> Decimal:
    """Calculate a safe KRW amount for manual Discord buys."""
    if requested_amount <= 0:
        raise RuntimeError("수동 매수 금액은 0보다 커야 합니다.")
    if requested_amount > max_manual_order_krw:
        raise RuntimeError(
            f"수동 매수 한도 초과: {format_krw(requested_amount)} > "
            f"{format_krw(max_manual_order_krw)}"
        )

    remaining_position_cap = max_position_krw - current_position_krw
    amount = min(requested_amount, krw_balance, remaining_position_cap)
    amount = amount.quantize(KRW_QUANT, rounding=ROUND_DOWN)
    if amount < min_order_krw:
        raise RuntimeError(
            f"주문 가능 금액이 최소 주문금액 미만입니다: {format_krw(amount)} KRW"
        )
    return amount
