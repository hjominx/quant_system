"""
Live account-level risk guard.

This module is intentionally independent from exchange clients so it can be
tested without touching Bithumb. It tracks account equity baselines and decides
whether a planned live order is allowed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


class RiskStateError(RuntimeError):
    """Raised when risk state is missing required fields or is corrupt."""


class RiskGateError(RuntimeError):
    """Raised when a planned order is blocked by the live risk guard."""


@dataclass(frozen=True)
class RiskLimits:
    """Positive loss limits expressed as fractions, e.g. Decimal("0.03")."""
    daily_loss_limit: Decimal
    weekly_loss_limit: Decimal
    max_drawdown_limit: Decimal

    @classmethod
    def from_config(cls, config) -> "RiskLimits":
        return cls(
            daily_loss_limit=Decimal(str(config.daily_loss_limit)),
            weekly_loss_limit=Decimal(str(config.weekly_loss_limit)),
            max_drawdown_limit=Decimal(str(config.max_drawdown_limit)),
        )

    def validate(self) -> None:
        _assert_positive_decimal(self.daily_loss_limit, "daily_loss_limit")
        _assert_positive_decimal(self.weekly_loss_limit, "weekly_loss_limit")
        _assert_positive_decimal(self.max_drawdown_limit, "max_drawdown_limit")


@dataclass(frozen=True)
class RiskState:
    """Persistent state for account equity baselines and kill switch."""
    date: str
    iso_week: str
    day_start_equity: Decimal
    week_start_equity: Decimal
    peak_equity: Decimal
    kill_switch: bool = False


@dataclass(frozen=True)
class RiskReport:
    """Risk snapshot for one decision cycle."""
    state: RiskState
    equity: Decimal
    daily_loss: Decimal
    weekly_loss: Decimal
    drawdown: Decimal
    daily_breached: bool
    weekly_breached: bool
    drawdown_breached: bool
    kill_switch_active: bool

    def blocking_reasons(self, action: str) -> list[str]:
        """Return human-readable reasons that block the given action."""
        action = action.upper()
        reasons: list[str] = []
        if action not in {"BUY", "SELL"}:
            return reasons
        if self.kill_switch_active:
            reasons.append("kill switch active")
        if self.drawdown_breached:
            reasons.append(
                f"MDD breach {format_pct(self.drawdown)}"
            )
        if action == "BUY":
            if self.daily_breached:
                reasons.append(
                    f"daily loss breach {format_pct(self.daily_loss)}"
                )
            if self.weekly_breached:
                reasons.append(
                    f"weekly loss breach {format_pct(self.weekly_loss)}"
                )
        return reasons


def iso_week_key(now: datetime) -> str:
    """Return a stable ISO week key such as 2026-W21."""
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def create_initial_risk_state(now: datetime, equity: Decimal) -> RiskState:
    """Create a risk state using current equity as every baseline."""
    _assert_positive_equity(equity)
    return RiskState(
        date=now.date().isoformat(),
        iso_week=iso_week_key(now),
        day_start_equity=equity,
        week_start_equity=equity,
        peak_equity=equity,
        kill_switch=False,
    )


def roll_risk_state(state: RiskState, now: datetime, equity: Decimal) -> RiskState:
    """Reset daily/weekly baselines when their periods change and update peak."""
    _assert_positive_equity(equity)
    today = now.date().isoformat()
    week = iso_week_key(now)
    day_start = state.day_start_equity if state.date == today else equity
    week_start = state.week_start_equity if state.iso_week == week else equity
    peak = max(state.peak_equity, equity)
    return RiskState(
        date=today,
        iso_week=week,
        day_start_equity=day_start,
        week_start_equity=week_start,
        peak_equity=peak,
        kill_switch=state.kill_switch,
    )


def load_risk_state(path: Path, now: datetime, equity: Decimal) -> RiskState:
    """Load risk state and fail closed when the file is corrupt."""
    if not path.exists():
        return create_initial_risk_state(now, equity)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = RiskState(
            date=str(raw["date"]),
            iso_week=str(raw["iso_week"]),
            day_start_equity=_decimal_from_json(raw["day_start_equity"], "day_start_equity"),
            week_start_equity=_decimal_from_json(raw["week_start_equity"], "week_start_equity"),
            peak_equity=_decimal_from_json(raw["peak_equity"], "peak_equity"),
            kill_switch=bool(raw.get("kill_switch", False)),
        )
    except (OSError, KeyError, TypeError, ValueError, InvalidOperation, json.JSONDecodeError) as exc:
        raise RiskStateError(f"risk state is corrupt: {path}") from exc

    _assert_positive_equity(state.day_start_equity)
    _assert_positive_equity(state.week_start_equity)
    _assert_positive_equity(state.peak_equity)
    return roll_risk_state(state, now, equity)


def save_risk_state(path: Path, state: RiskState) -> None:
    """Persist risk state using strings for Decimal precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": state.date,
        "iso_week": state.iso_week,
        "day_start_equity": str(state.day_start_equity),
        "week_start_equity": str(state.week_start_equity),
        "peak_equity": str(state.peak_equity),
        "kill_switch": state.kill_switch,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_risk(
    *,
    state: RiskState,
    equity: Decimal,
    limits: RiskLimits,
    kill_switch_file_active: bool = False,
) -> RiskReport:
    """Evaluate account risk against configured limits."""
    _assert_positive_equity(equity)
    limits.validate()
    daily_loss = _loss_fraction(state.day_start_equity, equity)
    weekly_loss = _loss_fraction(state.week_start_equity, equity)
    drawdown = _loss_fraction(state.peak_equity, equity)
    return RiskReport(
        state=state,
        equity=equity,
        daily_loss=daily_loss,
        weekly_loss=weekly_loss,
        drawdown=drawdown,
        daily_breached=daily_loss >= limits.daily_loss_limit,
        weekly_breached=weekly_loss >= limits.weekly_loss_limit,
        drawdown_breached=drawdown >= limits.max_drawdown_limit,
        kill_switch_active=state.kill_switch or kill_switch_file_active,
    )


def assert_risk_allows_action(action: str, report: RiskReport) -> None:
    """Raise when the risk report blocks the planned order action."""
    reasons = report.blocking_reasons(action)
    if reasons:
        raise RiskGateError(
            f"risk gate blocked {action.upper()}: " + "; ".join(reasons)
        )


def format_pct(value: Decimal) -> str:
    """Format a fractional Decimal as a percentage string."""
    return f"{value * Decimal('100'):.2f}%"


def _loss_fraction(start: Decimal, current: Decimal) -> Decimal:
    _assert_positive_equity(start)
    _assert_positive_equity(current)
    loss = (start - current) / start
    return max(loss, Decimal("0"))


def _decimal_from_json(value, field_name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise RiskStateError(f"{field_name} is not a Decimal") from exc
    if not parsed.is_finite():
        raise RiskStateError(f"{field_name} is not finite")
    return parsed


def _assert_positive_equity(equity: Decimal) -> None:
    _assert_positive_decimal(equity, "equity")


def _assert_positive_decimal(value: Decimal, name: str) -> None:
    if not value.is_finite() or value <= 0:
        raise RiskStateError(f"{name} must be positive: {value}")
