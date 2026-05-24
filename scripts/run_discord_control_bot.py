"""Run an interactive Discord control panel bot.

This is separate from webhook alerts. It uses a Discord bot token and button
interactions to provide a small operations console for the live trader.

Manual market orders and kill switch controls require DISCORD_ADMIN_USER_IDS.
Manual orders run through the same live risk, spread, cooldown, and quota gates
as the automated runner.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.bithumb_client import BithumbClient
from notifications.discord import (
    COLOR_DANGER,
    COLOR_INFO,
    COLOR_SUCCESS,
    COLOR_WARNING,
    DiscordEmbed,
    DiscordField,
)
from notifications.discord_status import (
    build_snapshot_embeds,
    collect_account_snapshot,
    estimate_total_equity,
    format_krw,
    select_snapshot_embeds,
)
from notifications.discord_control_safety import (
    calculate_manual_buy_amount,
    parse_admin_user_ids,
)
from risk.live_guard import (
    RiskLimits,
    assert_risk_allows_action,
    evaluate_risk,
    format_pct,
    load_risk_state,
    save_risk_state,
)
from scripts.run_eth_cash_micro_live import (
    VOLUME_QUANT,
    assert_order_quota,
    assert_spread_ok,
    calculate_sell_volume,
    default_timeframe_for_profile,
    disable_auto_trade,
    disable_risk_override,
    enable_auto_trade,
    enable_risk_override,
    format_rebate_turnover_progress,
    auto_trade_enabled,
    load_rebate_turnover_progress,
    load_pilot_state,
    load_runtime_market_mode,
    load_runtime_profile_mode,
    market_base_currency,
    normalize_krw_market,
    record_order,
    save_runtime_profile_mode,
    save_runtime_market_mode,
    save_pilot_state,
)
from strategies.rebate_market_scanner import (
    RebateMarketCandidate,
    format_candidate_lines,
    scan_rebate_micro_candidates,
)
from utils.config import get_config
from utils.logger import setup_logging


try:
    import discord
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on VPS setup
    raise SystemExit(
        "discord.py가 필요합니다. 설치: pip install discord.py"
    ) from exc


PANEL_TITLE = "Lo-Fi Quant Console"


def load_panel_message_id(path: Path) -> Optional[int]:
    """Load persisted panel message ID."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    value = raw.get("panel_message_id")
    return int(value) if value else None


def save_panel_message_id(path: Path, message_id: int) -> None:
    """Persist panel message ID so restarts edit instead of spamming."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"panel_message_id": str(message_id)}, f, ensure_ascii=False, indent=2)


def to_discord_embed(embed: DiscordEmbed) -> discord.Embed:
    """Convert the repo's webhook embed wrapper to discord.py Embed."""
    return discord.Embed.from_dict(embed.to_payload())


def to_discord_embeds(embeds: Iterable[DiscordEmbed]) -> list[discord.Embed]:
    """Convert several wrapped embeds."""
    return [to_discord_embed(embed) for embed in embeds]


async def respond_interaction_error(
    interaction: discord.Interaction,
    error: Exception,
    *,
    prefix: str = "상호작용 처리 실패",
) -> None:
    """Send an ephemeral interaction error instead of leaving Discord silent."""
    message = f"{prefix}: `{type(error).__name__}: {error}`"
    try:
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
    except Exception as notify_error:
        print(
            "Failed to report interaction error: "
            f"{type(notify_error).__name__}: {notify_error}"
        )
    print(f"{prefix}: {type(error).__name__}: {error}")


class QuantControlController:
    """Controller backing the Discord UI buttons."""

    def __init__(
        self,
        *,
        config,
        market: str,
        order_limit: int,
        kill_switch_path: Path,
        risk_state_path: Path,
        pilot_state_path: Path,
        profile_mode_path: Path,
        market_mode_path: Path,
        auto_trade_path: Path,
        force_start_path: Path,
        turnover_path: Path,
        service_name: str,
        admin_user_ids: set[str],
        max_manual_order_krw: Decimal,
        max_position_krw: Decimal,
        min_order_krw: Decimal,
        max_spread_pct: Decimal,
        max_orders_per_day: int,
        cooldown_minutes: int,
        cooldown_seconds: Optional[int],
        force_start_minutes: int,
        turnover_target_krw: Decimal,
        scanner_min_liquidity_krw: Decimal,
        scanner_max_spread_pct: Decimal,
        scanner_min_volatility_pct: Decimal,
        scanner_fee_round_trip_pct: Decimal,
        scanner_rebate_round_trip_pct: Decimal,
        scanner_slippage_buffer_pct: Decimal,
        scanner_prefilter_limit: int,
    ):
        self.config = config
        self.market = market
        self.order_limit = order_limit
        self.kill_switch_path = kill_switch_path
        self.risk_state_path = risk_state_path
        self.pilot_state_path = pilot_state_path
        self.profile_mode_path = profile_mode_path
        self.market_mode_path = market_mode_path
        self.auto_trade_path = auto_trade_path
        self.force_start_path = force_start_path
        self.turnover_path = turnover_path
        self.service_name = service_name
        self.admin_user_ids = admin_user_ids
        self.max_manual_order_krw = max_manual_order_krw
        self.max_position_krw = max_position_krw
        self.min_order_krw = min_order_krw
        self.max_spread_pct = max_spread_pct
        self.max_orders_per_day = max_orders_per_day
        self.cooldown_minutes = cooldown_minutes
        self.cooldown_seconds = cooldown_seconds
        self.force_start_minutes = force_start_minutes
        self.turnover_target_krw = turnover_target_krw
        self.scanner_min_liquidity_krw = scanner_min_liquidity_krw
        self.scanner_max_spread_pct = scanner_max_spread_pct
        self.scanner_min_volatility_pct = scanner_min_volatility_pct
        self.scanner_fee_round_trip_pct = scanner_fee_round_trip_pct
        self.scanner_rebate_round_trip_pct = scanner_rebate_round_trip_pct
        self.scanner_slippage_buffer_pct = scanner_slippage_buffer_pct
        self.scanner_prefilter_limit = scanner_prefilter_limit
        self.client = BithumbClient.from_config(
            config,
            dry_run=True,
            allow_live_orders=False,
        )
        self.lock = asyncio.Lock()

    def is_admin(self, user_id: int) -> bool:
        """Return whether this Discord user can run mutating controls."""
        return str(user_id) in self.admin_user_ids

    def current_profile_mode(self) -> str:
        """Return the active runtime profile, falling back to stable mode."""
        return load_runtime_profile_mode(self.profile_mode_path) or "stable_quality"

    def set_profile_mode(self, profile: str, *, updated_by: str) -> None:
        """Persist a runtime stable/scalp profile selection."""
        save_runtime_profile_mode(self.profile_mode_path, profile, updated_by=updated_by)

    def current_market_mode(self) -> str:
        """Return the active market for the currently selected profile."""
        if self.current_profile_mode() in ("pulse_10s", "rebate_micro"):
            return load_runtime_market_mode(self.market_mode_path) or self.market
        return self.market

    def set_pulse_market(self, market: str, *, updated_by: str) -> str:
        """Persist the Pulse KRW market and return the normalized value."""
        normalized = normalize_krw_market(market)
        save_runtime_market_mode(self.market_mode_path, normalized, updated_by=updated_by)
        return normalized

    def micro_auto_state(self) -> str:
        """Return start/stop controlled micro-trader state."""
        return "enabled" if auto_trade_enabled(self.auto_trade_path) else "stopped"

    def start_micro_trader(
        self,
        *,
        updated_by: str,
        market: Optional[str] = None,
        force_loss_limits: bool = False,
    ) -> str:
        """Switch to rebate_micro and enable start/stop controlled trading."""
        selected_market = self.current_market_mode() if market is None else normalize_krw_market(market)
        save_runtime_market_mode(
            self.market_mode_path,
            selected_market,
            updated_by=updated_by,
        )
        save_runtime_profile_mode(
            self.profile_mode_path,
            "rebate_micro",
            updated_by=updated_by,
        )
        enable_auto_trade(
            self.auto_trade_path,
            profile="rebate_micro",
            updated_by=updated_by,
        )
        if force_loss_limits:
            enable_risk_override(
                self.force_start_path,
                profile="rebate_micro",
                market=selected_market,
                updated_by=updated_by,
                minutes=self.force_start_minutes,
            )
        else:
            disable_risk_override(self.force_start_path)
        return selected_market

    def stop_micro_trader(self) -> None:
        """Stop start/stop controlled micro trading without forcing liquidation."""
        disable_auto_trade(self.auto_trade_path)
        disable_risk_override(self.force_start_path)

    def rebate_turnover_progress_text(self) -> str:
        """Return formatted Rebate Micro turnover progress."""
        progress = load_rebate_turnover_progress(
            self.turnover_path,
            target_krw=self.turnover_target_krw,
        )
        return format_rebate_turnover_progress(progress)

    async def recommend_rebate_markets(self) -> list[RebateMarketCandidate]:
        """Scan live Bithumb markets and return Rebate Micro candidates."""
        async with self.lock:
            return await asyncio.to_thread(
                scan_rebate_micro_candidates,
                self.client,
                min_liquidity_krw=self.scanner_min_liquidity_krw,
                max_spread_pct=self.scanner_max_spread_pct,
                min_volatility_pct=self.scanner_min_volatility_pct,
                fee_round_trip_pct=self.scanner_fee_round_trip_pct,
                rebate_round_trip_pct=self.scanner_rebate_round_trip_pct,
                slippage_buffer_pct=self.scanner_slippage_buffer_pct,
                prefilter_limit=self.scanner_prefilter_limit,
            )

    async def collect_snapshot_embeds(self, report: str) -> list[discord.Embed]:
        """Collect Bithumb account state and convert it to Discord embeds."""
        market = self.current_market_mode()
        async with self.lock:
            snapshot = await asyncio.to_thread(
                collect_account_snapshot,
                self.client,
                market=market,
                order_limit=self.order_limit,
            )
        embeds = build_snapshot_embeds(
            snapshot,
            title_prefix="Button query",
            kill_switch_active=self.kill_switch_path.exists(),
            timestamp=datetime.now(),
        )
        return to_discord_embeds(select_snapshot_embeds(embeds, report))

    async def build_risk_embed(self) -> discord.Embed:
        """Build risk and guard status embed."""
        market = self.current_market_mode()
        async with self.lock:
            snapshot = await asyncio.to_thread(
                collect_account_snapshot,
                self.client,
                market=market,
                order_limit=self.order_limit,
            )
        equity = estimate_total_equity(snapshot.balances, snapshot.prices_by_currency)
        now = datetime.now()
        risk_state = load_risk_state(self.risk_state_path, now, equity)
        limits = RiskLimits.from_config(self.config)
        report = evaluate_risk(
            state=risk_state,
            equity=equity,
            limits=limits,
            kill_switch_file_active=self.kill_switch_path.exists(),
        )
        block_summary = _risk_block_summary(report)
        if self.current_profile_mode() == "rebate_micro":
            block_summary = (
                f"{block_summary}\n"
                "Rebate Micro: daily/weekly/MDD loss gates do not block auto orders; kill switch remains active."
            )
        has_block = bool(
            report.kill_switch_active or
            report.daily_breached or
            report.weekly_breached or
            report.drawdown_breached
        )
        color = COLOR_DANGER if has_block else COLOR_SUCCESS
        embed = DiscordEmbed(
            title="Risk board",
            description="Live guard and account drawdown state",
            color=color,
            fields=(
                DiscordField("Equity", f"{format_krw(report.equity)} KRW"),
                DiscordField("Daily", f"{format_pct(report.daily_loss)} / {format_pct(limits.daily_loss_limit)}"),
                DiscordField("Weekly", f"{format_pct(report.weekly_loss)} / {format_pct(limits.weekly_loss_limit)}"),
                DiscordField("MDD", f"{format_pct(report.drawdown)} / {format_pct(limits.max_drawdown_limit)}"),
                DiscordField(
                    "Kill switch",
                    "ACTIVE" if report.kill_switch_active else "inactive",
                ),
                DiscordField(
                    "Blocks",
                    block_summary,
                    inline=False,
                ),
            ),
        )
        return to_discord_embed(embed)

    async def build_panel_embed(self) -> discord.Embed:
        """Build the public control panel embed."""
        service_state = await asyncio.to_thread(systemctl_state, self.service_name)
        kill_state = "ACTIVE" if self.kill_switch_path.exists() else "inactive"
        profile = self.current_profile_mode()
        timeframe = default_timeframe_for_profile(profile)
        market = self.current_market_mode()
        loss_gate_state = "bypassed" if profile == "rebate_micro" else "active"
        color = COLOR_DANGER if self.kill_switch_path.exists() else COLOR_INFO
        admin_state = (
            f"{len(self.admin_user_ids)} admin id(s)"
            if self.admin_user_ids
            else "not configured"
        )
        embed = DiscordEmbed(
            title=PANEL_TITLE,
            description=(
                "Button console for the ETH/Cash live trader. "
                "조회 버튼은 상태를 읽고, 킬스위치와 수동매매는 관리자만 실행합니다."
            ),
            color=color,
            fields=(
                DiscordField("Market", market),
                DiscordField("Mode", f"{profile} ({timeframe})"),
                DiscordField("Live service", service_state),
                DiscordField("Kill switch", kill_state),
                DiscordField("Micro auto", self.micro_auto_state()),
                DiscordField("Micro loss gate", loss_gate_state),
                DiscordField("Turnover", self.rebate_turnover_progress_text(), inline=False),
                DiscordField("Manual max", f"{format_krw(self.max_manual_order_krw)} KRW/order"),
                DiscordField("Alerts", "Hourly status + order/risk/error webhooks"),
                DiscordField("Admin gate", admin_state),
                DiscordField("Updated", datetime.now().isoformat(timespec="minutes")),
            ),
        )
        return to_discord_embed(embed)

    async def execute_manual_buy(self, amount: Decimal) -> discord.Embed:
        """Execute one guarded manual market buy."""
        async with self.lock:
            return await asyncio.to_thread(self._execute_manual_buy_sync, amount)

    async def execute_manual_sell_all(self) -> discord.Embed:
        """Execute one guarded manual all-position market sell."""
        async with self.lock:
            return await asyncio.to_thread(self._execute_manual_sell_all_sync)

    def _execute_manual_buy_sync(self, amount: Decimal) -> discord.Embed:
        now = datetime.now()
        market = self.current_market_mode()
        with BithumbClient.from_config(
            self.config,
            dry_run=False,
            allow_live_orders=self.config.is_live_trading,
        ) as live_client:
            self._assert_manual_trade_preflight(live_client, "BUY", now, market=market)
            ticker = live_client.get_ticker(market)
            balances = live_client.get_balances()
            base = market_base_currency(market)
            current_position_krw = balances.get_total(base) * ticker.trade_price
            krw_amount = calculate_manual_buy_amount(
                requested_amount=amount,
                krw_balance=balances.krw_balance,
                current_position_krw=current_position_krw,
                max_position_krw=self.max_position_krw,
                max_manual_order_krw=self.max_manual_order_krw,
                min_order_krw=self.min_order_krw,
            )
            order = live_client.market_buy(market, krw_amount)
            self._record_manual_order(now, action="BUY")
            return build_manual_order_embed(
                title="Manual BUY requested",
                action="BUY",
                amount=krw_amount,
                volume=None,
                order_uuid=order.uuid,
                state=order.state.value,
            )

    def _execute_manual_sell_all_sync(self) -> discord.Embed:
        now = datetime.now()
        market = self.current_market_mode()
        with BithumbClient.from_config(
            self.config,
            dry_run=False,
            allow_live_orders=self.config.is_live_trading,
        ) as live_client:
            self._assert_manual_trade_preflight(live_client, "SELL", now, market=market)
            ticker = live_client.get_ticker(market)
            balances = live_client.get_balances()
            base = market_base_currency(market)
            volume = calculate_sell_volume(
                coin_balance=balances.get_total(base),
                trade_price=ticker.trade_price,
                min_order_krw=self.min_order_krw,
            )
            if volume <= 0:
                raise RuntimeError("매도 가능한 선택 마켓 평가액이 최소 주문금액 미만입니다.")
            order = live_client.market_sell(market, volume)
            self._record_manual_order(now, action="SELL")
            return build_manual_order_embed(
                title="Manual SELL ALL requested",
                action="SELL",
                amount=None,
                volume=volume,
                order_uuid=order.uuid,
                state=order.state.value,
            )

    def _assert_manual_trade_preflight(
        self,
        live_client: BithumbClient,
        action: str,
        now: datetime,
        *,
        market: str,
    ) -> None:
        """Apply the same live safety gates before manual Discord orders."""
        if not self.config.is_live_trading:
            raise RuntimeError("TRADING_MODE=LIVE가 아니어서 수동매매를 차단합니다.")

        open_orders = live_client.get_open_orders(market)
        if open_orders:
            raise RuntimeError(f"미체결 주문 {len(open_orders)}개가 있어 수동매매를 차단합니다.")

        orderbook = live_client.get_orderbook(market)
        assert_spread_ok(orderbook.spread_pct, self.max_spread_pct)

        ticker = live_client.get_ticker(market)
        balances = live_client.get_balances()
        base = market_base_currency(market)
        equity = balances.krw_balance + balances.get_total(base) * ticker.trade_price
        risk_state = load_risk_state(self.risk_state_path, now, equity)
        risk_report = evaluate_risk(
            state=risk_state,
            equity=equity,
            limits=RiskLimits.from_config(self.config),
            kill_switch_file_active=self.kill_switch_path.exists(),
        )
        save_risk_state(self.risk_state_path, risk_report.state)
        assert_risk_allows_action(action, risk_report)

        state = load_pilot_state(self.pilot_state_path, now)
        assert_order_quota(
            state=state,
            now=now,
            max_orders_per_day=self.max_orders_per_day,
            cooldown_minutes=self.cooldown_minutes,
            cooldown_seconds=self.cooldown_seconds,
        )

    def _record_manual_order(self, now: datetime, *, action: str) -> None:
        """Record manual orders in the live runner quota/cooldown state."""
        state = load_pilot_state(self.pilot_state_path, now)
        save_pilot_state(self.pilot_state_path, record_order(state, now, action=action))

    async def build_log_embed(self) -> discord.Embed:
        """Build a compact journalctl tail embed."""
        output = await asyncio.to_thread(read_service_log_tail, self.service_name)
        embed = DiscordEmbed(
            title="Live service log",
            description=f"```text\n{output}\n```",
            color=COLOR_INFO,
        )
        return to_discord_embed(embed)

    def enable_kill_switch(self) -> None:
        """Create the kill switch file."""
        self.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
        self.kill_switch_path.write_text(
            f"enabled_at={datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )

    def disable_kill_switch(self) -> None:
        """Remove the kill switch file."""
        if self.kill_switch_path.exists():
            self.kill_switch_path.unlink()

    def assert_reboot_allowed(self) -> None:
        """Fail fast if the service user cannot run passwordless sudo reboot."""
        result = subprocess.run(
            ["/usr/bin/sudo", "-n", "true"],
            text=True,
            capture_output=True,
            check=False,
            timeout=3,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "sudo permission denied").strip()
            raise RuntimeError(f"passwordless sudo check failed: {detail}")

    def launch_server_reboot(self) -> None:
        """Request an immediate VPS reboot without waiting for the process."""
        subprocess.Popen(
            ["/usr/bin/sudo", "-n", "/usr/bin/systemctl", "reboot"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


class ControlPanelView(discord.ui.View):
    """Persistent Discord button panel."""

    def __init__(self, controller: QuantControlController):
        super().__init__(timeout=None)
        self.controller = controller

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item,
    ) -> None:
        await respond_interaction_error(interaction, error)

    async def _send_report(self, interaction: discord.Interaction, report: str) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            embeds = await self.controller.collect_snapshot_embeds(report)
            await interaction.followup.send(embeds=embeds, ephemeral=True)
        except Exception as exc:
            await interaction.followup.send(
                f"조회 실패: `{type(exc).__name__}`",
                ephemeral=True,
            )

    async def _require_admin(self, interaction: discord.Interaction) -> bool:
        if self.controller.is_admin(interaction.user.id):
            return True
        message = (
            "관리자 권한이 없습니다. `.env`의 `DISCORD_ADMIN_USER_IDS`에 "
            f"`{interaction.user.id}`를 추가해야 킬스위치/수동매매 버튼을 쓸 수 있습니다."
        )
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        return False

    @discord.ui.button(label="Status", style=discord.ButtonStyle.primary, custom_id="quant:status", row=0)
    async def status_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._send_report(interaction, "all")

    @discord.ui.button(label="Price", style=discord.ButtonStyle.secondary, custom_id="quant:price", row=0)
    async def price_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._send_report(interaction, "price")

    @discord.ui.button(label="Assets", style=discord.ButtonStyle.secondary, custom_id="quant:assets", row=0)
    async def assets_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._send_report(interaction, "assets")

    @discord.ui.button(label="Orders", style=discord.ButtonStyle.secondary, custom_id="quant:orders", row=0)
    async def orders_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._send_report(interaction, "orders")

    @discord.ui.button(label="Risk", style=discord.ButtonStyle.secondary, custom_id="quant:risk", row=1)
    async def risk_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            await interaction.followup.send(
                embed=await self.controller.build_risk_embed(),
                ephemeral=True,
            )
        except Exception as exc:
            await interaction.followup.send(
                f"리스크 조회 실패: `{type(exc).__name__}`",
                ephemeral=True,
            )

    @discord.ui.button(label="Logs", style=discord.ButtonStyle.secondary, custom_id="quant:logs", row=1)
    async def logs_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not await self._require_admin(interaction):
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        await interaction.followup.send(
            embed=await self.controller.build_log_embed(),
            ephemeral=True,
        )

    @discord.ui.button(label="Recommend", style=discord.ButtonStyle.primary, custom_id="quant:micro:recommend", row=1)
    async def recommend_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not await self._require_admin(interaction):
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            candidates = await self.controller.recommend_rebate_markets()
            embed = build_rebate_recommendation_embed(candidates)
            await interaction.followup.send(
                embed=embed,
                view=RebateRecommendationView(
                    self.controller,
                    candidates=candidates,
                    panel_message=interaction.message,
                ) if candidates else None,
                ephemeral=True,
            )
        except Exception as exc:
            await interaction.followup.send(
                f"종목 추천 실패: `{type(exc).__name__}: {exc}`",
                ephemeral=True,
            )

    @discord.ui.button(label="Progress", style=discord.ButtonStyle.secondary, custom_id="quant:micro:progress", row=1)
    async def progress_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_message(
            f"Rebate Micro 거래대금 진행률: `{self.controller.rebate_turnover_progress_text()}`",
            ephemeral=True,
        )

    @discord.ui.button(label="Refresh Panel", style=discord.ButtonStyle.secondary, custom_id="quant:refresh", row=1)
    async def refresh_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True)
        await interaction.message.edit(
            embed=await self.controller.build_panel_embed(),
            view=self,
        )
        await interaction.followup.send("패널을 새로고침했습니다.", ephemeral=True)

    @discord.ui.button(label="Buy 5k", style=discord.ButtonStyle.success, custom_id="quant:manual:buy:5000", row=2)
    async def buy_5k_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._request_manual_trade(interaction, action="BUY", amount=Decimal("5000"))

    @discord.ui.button(label="Buy 10k", style=discord.ButtonStyle.success, custom_id="quant:manual:buy:10000", row=2)
    async def buy_10k_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._request_manual_trade(interaction, action="BUY", amount=Decimal("10000"))

    @discord.ui.button(label="Buy 25k", style=discord.ButtonStyle.success, custom_id="quant:manual:buy:25000", row=2)
    async def buy_25k_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._request_manual_trade(interaction, action="BUY", amount=Decimal("25000"))

    @discord.ui.button(label="Sell ALL", style=discord.ButtonStyle.danger, custom_id="quant:manual:sell:all", row=2)
    async def sell_all_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._request_manual_trade(interaction, action="SELL", amount=None)

    async def _request_manual_trade(
        self,
        interaction: discord.Interaction,
        *,
        action: str,
        amount: Optional[Decimal],
    ) -> None:
        if not await self._require_admin(interaction):
            return
        if action == "BUY":
            description = f"{format_krw(amount or Decimal('0'))} KRW 시장가 매수를 요청할까요?"
        else:
            description = "보유 ETH 전체 시장가 매도를 요청할까요?"
        await interaction.response.send_message(
            description,
            view=ConfirmManualTradeView(
                self.controller,
                action=action,
                amount=amount,
                panel_message=interaction.message,
            ),
            ephemeral=True,
        )

    @discord.ui.button(label="Kill ON", style=discord.ButtonStyle.danger, custom_id="quant:kill:on", row=3)
    async def kill_on_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not await self._require_admin(interaction):
            return
        self.controller.enable_kill_switch()
        await interaction.response.send_message(
            "킬스위치를 켰습니다. 다음 사이클부터 BUY/SELL 실주문이 차단됩니다.",
            ephemeral=True,
        )
        await interaction.message.edit(
            embed=await self.controller.build_panel_embed(),
            view=self,
        )

    @discord.ui.button(label="Kill OFF", style=discord.ButtonStyle.success, custom_id="quant:kill:off", row=3)
    async def kill_off_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not await self._require_admin(interaction):
            return
        await interaction.response.send_message(
            "정말 킬스위치를 해제할까요?",
            view=ConfirmKillSwitchOffView(self.controller, interaction.message),
            ephemeral=True,
        )

    @discord.ui.button(label="Reboot VPS", style=discord.ButtonStyle.primary, custom_id="quant:vps:reboot", row=3)
    async def reboot_vps_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not await self._require_admin(interaction):
            return
        await interaction.response.send_message(
            "AWS VPS를 리부트할까요? 봇과 라이브 서비스가 잠시 끊겼다가 systemd로 다시 올라옵니다.",
            view=ConfirmServerRebootView(self.controller, interaction.message),
            ephemeral=True,
        )

    @discord.ui.button(label="Start Micro", style=discord.ButtonStyle.success, custom_id="quant:micro:start", row=3)
    async def start_micro_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not await self._require_admin(interaction):
            return
        await interaction.response.send_message(
            (
                "Rebate Micro 자동매매를 시작할까요? 선택 마켓에서 1분봉 마이크로 신호가 있을 때만 "
                "진입하고, +0.03~1.00% 동적 익절 / -0.04% 손절 모델이 다음 사이클부터 작동합니다."
            ),
            view=ConfirmStartMicroView(self.controller, interaction.message),
            ephemeral=True,
        )

    @discord.ui.button(label="Stop Micro", style=discord.ButtonStyle.secondary, custom_id="quant:micro:stop", row=3)
    async def stop_micro_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not await self._require_admin(interaction):
            return
        self.controller.stop_micro_trader()
        await interaction.response.send_message(
            (
                "Rebate Micro 자동매매를 중지했습니다. 보유분은 강제청산하지 않습니다.\n"
                f"거래대금 진행률: `{self.controller.rebate_turnover_progress_text()}`"
            ),
            ephemeral=True,
        )
        await interaction.message.edit(
            embed=await self.controller.build_panel_embed(),
            view=self,
        )

    @discord.ui.button(label="Stable Mode", style=discord.ButtonStyle.secondary, custom_id="quant:mode:stable", row=4)
    async def stable_mode_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._request_profile_mode(interaction, "stable_quality")

    @discord.ui.button(label="Scalp 5m", style=discord.ButtonStyle.danger, custom_id="quant:mode:scalp5m", row=4)
    async def scalp_5m_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._request_profile_mode(interaction, "scalp_active")

    @discord.ui.button(label="Scalp 1m", style=discord.ButtonStyle.danger, custom_id="quant:mode:scalp1m", row=4)
    async def scalp_1m_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._request_profile_mode(interaction, "scalp_1m")

    @discord.ui.button(label="Pulse 1s", style=discord.ButtonStyle.danger, custom_id="quant:mode:pulse10s", row=4)
    async def pulse_10s_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._request_profile_mode(interaction, "pulse_10s")

    @discord.ui.button(label="Set Market", style=discord.ButtonStyle.secondary, custom_id="quant:pulse:market", row=4)
    async def set_market_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not await self._require_admin(interaction):
            return
        await interaction.response.send_modal(
            PulseMarketModal(self.controller, panel_message=interaction.message)
        )

    async def _request_profile_mode(self, interaction: discord.Interaction, profile: str) -> None:
        if not await self._require_admin(interaction):
            return
        timeframe = default_timeframe_for_profile(profile)
        await interaction.response.send_message(
            f"`{profile}` ({timeframe}) 모드로 전환할까요? 다음 판단 사이클부터 적용됩니다.",
            view=ConfirmProfileModeView(
                self.controller,
                profile=profile,
                panel_message=interaction.message,
            ),
            ephemeral=True,
        )


class PulseMarketModal(discord.ui.Modal, title="Set Pulse Market"):
    """Admin modal for changing the selected Pulse KRW market."""

    market = discord.ui.TextInput(
        label="KRW market",
        placeholder="KRW-SOL",
        max_length=20,
    )

    def __init__(
        self,
        controller: QuantControlController,
        *,
        panel_message: discord.Message,
    ):
        super().__init__()
        self.controller = controller
        self.panel_message = panel_message

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await respond_interaction_error(interaction, error, prefix="시장 변경 처리 실패")

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if not self.controller.is_admin(interaction.user.id):
            await interaction.response.send_message("관리자 권한이 없습니다.", ephemeral=True)
            return
        try:
            market = self.controller.set_pulse_market(
                str(self.market.value),
                updated_by=str(interaction.user.id),
            )
            await interaction.response.send_message(
                f"Pulse 시장을 `{market}`로 저장했습니다. Pulse 모드의 다음 판단부터 적용됩니다.",
                ephemeral=True,
            )
            await self.panel_message.edit(
                embed=await self.controller.build_panel_embed(),
                view=ControlPanelView(self.controller),
            )
        except Exception as exc:
            await interaction.response.send_message(
                f"시장 변경 실패: `{type(exc).__name__}: {exc}`",
                ephemeral=True,
            )


class RebateRecommendationView(discord.ui.View):
    """Ephemeral buttons for selecting a recommended Rebate Micro market."""

    def __init__(
        self,
        controller: QuantControlController,
        *,
        candidates: list[RebateMarketCandidate],
        panel_message: discord.Message,
    ):
        super().__init__(timeout=300)
        self.controller = controller
        self.candidates = candidates[:5]
        self.panel_message = panel_message
        for idx, candidate in enumerate(self.candidates):
            self.add_item(RebateCandidateButton(idx, candidate))

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item,
    ) -> None:
        await respond_interaction_error(interaction, error, prefix="추천 종목 선택 실패")


class RebateCandidateButton(discord.ui.Button):
    """A single candidate selection button."""

    def __init__(self, index: int, candidate: RebateMarketCandidate):
        label = f"{index + 1}. {candidate.market.replace('KRW-', '')}"
        style = discord.ButtonStyle.success if candidate.expected_net_pct > 0 else discord.ButtonStyle.secondary
        super().__init__(
            label=label,
            style=style,
            custom_id=f"quant:micro:candidate:{index}",
            row=0,
        )
        self.candidate = candidate

    async def callback(self, interaction: discord.Interaction) -> None:
        view = self.view
        if not isinstance(view, RebateRecommendationView):
            await interaction.response.send_message("추천 뷰 상태를 확인할 수 없습니다.", ephemeral=True)
            return
        if not view.controller.is_admin(interaction.user.id):
            await interaction.response.send_message("관리자 권한이 없습니다.", ephemeral=True)
            return
        await interaction.response.send_message(
            (
                f"`{self.candidate.market}`로 Rebate Micro를 시작할까요?\n"
                f"{format_candidate_line(self.candidate)}"
            ),
            view=ConfirmRebateMarketStartView(
                view.controller,
                candidate=self.candidate,
                panel_message=view.panel_message,
            ),
            ephemeral=True,
        )


class ConfirmRebateMarketStartView(discord.ui.View):
    """Confirm normal or forced Rebate Micro start for a selected market."""

    def __init__(
        self,
        controller: QuantControlController,
        *,
        candidate: RebateMarketCandidate,
        panel_message: discord.Message,
    ):
        super().__init__(timeout=300)
        self.controller = controller
        self.candidate = candidate
        self.panel_message = panel_message
        self.submitted = False

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item,
    ) -> None:
        await respond_interaction_error(interaction, error, prefix="Micro 시작 처리 실패")

    async def _start(self, interaction: discord.Interaction, *, force_loss_limits: bool) -> None:
        if not self.controller.is_admin(interaction.user.id):
            await interaction.response.send_message("관리자 권한이 없습니다.", ephemeral=True)
            return
        if self.submitted:
            await interaction.response.send_message("이미 시작 요청을 처리 중입니다.", ephemeral=True)
            return
        self.submitted = True
        for item in self.children:
            item.disabled = True
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            await interaction.message.edit(view=self)
        except Exception:
            pass
        try:
            market = self.controller.start_micro_trader(
                updated_by=str(interaction.user.id),
                market=self.candidate.market,
                force_loss_limits=force_loss_limits,
            )
            mode = "Force Start" if force_loss_limits else "Start"
            detail = (
                f"{mode}: `{market}` Rebate Micro를 시작했습니다.\n"
                f"거래대금 진행률: `{self.controller.rebate_turnover_progress_text()}`"
            )
            if force_loss_limits:
                detail += (
                    "\nRebate Micro는 기본적으로 일일/주간/MDD 손실률 게이트를 자동매매 차단에 쓰지 않습니다. "
                    "킬스위치/스프레드/미체결/잔고/주문횟수 한도는 계속 적용됩니다."
                )
            await interaction.followup.send(detail, ephemeral=True)
            await self.panel_message.edit(
                embed=await self.controller.build_panel_embed(),
                view=ControlPanelView(self.controller),
            )
        except Exception as exc:
            self.submitted = False
            await interaction.followup.send(
                f"Micro 시작 실패: `{type(exc).__name__}: {exc}`",
                ephemeral=True,
            )

    @discord.ui.button(label="Start", style=discord.ButtonStyle.success)
    async def start(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._start(interaction, force_loss_limits=False)

    @discord.ui.button(label="Force Start", style=discord.ButtonStyle.danger)
    async def force_start(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._start(interaction, force_loss_limits=True)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.edit_message(content="Micro 시작을 취소했습니다.", view=None)


class ConfirmManualTradeView(discord.ui.View):
    """Two-step confirmation before manual market orders."""

    def __init__(
        self,
        controller: QuantControlController,
        *,
        action: str,
        amount: Optional[Decimal],
        panel_message: discord.Message,
    ):
        super().__init__(timeout=60)
        self.controller = controller
        self.action = action
        self.amount = amount
        self.panel_message = panel_message
        self.submitted = False

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item,
    ) -> None:
        await respond_interaction_error(interaction, error, prefix="수동매매 처리 실패")

    @discord.ui.button(label="Confirm Manual Trade", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.controller.is_admin(interaction.user.id):
            await interaction.response.send_message("관리자 권한이 없습니다.", ephemeral=True)
            return
        if self.submitted:
            await interaction.response.send_message("이미 처리 중인 수동매매 요청입니다.", ephemeral=True)
            return
        self.submitted = True
        for item in self.children:
            item.disabled = True

        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            await interaction.message.edit(view=self)
        except Exception:
            pass
        try:
            if self.action == "BUY":
                result_embed = await self.controller.execute_manual_buy(self.amount or Decimal("0"))
            else:
                result_embed = await self.controller.execute_manual_sell_all()
            await interaction.followup.send(embed=result_embed, ephemeral=True)
            await self.panel_message.edit(
                embed=await self.controller.build_panel_embed(),
                view=ControlPanelView(self.controller),
            )
        except Exception as exc:
            self.submitted = False
            await interaction.followup.send(
                f"수동매매 차단/실패: `{type(exc).__name__}: {exc}`",
                ephemeral=True,
            )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.edit_message(content="수동매매를 취소했습니다.", view=None)


class ConfirmKillSwitchOffView(discord.ui.View):
    """Two-step confirmation before removing the kill switch."""

    def __init__(self, controller: QuantControlController, panel_message: discord.Message):
        super().__init__(timeout=300)
        self.controller = controller
        self.panel_message = panel_message
        self.submitted = False

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item,
    ) -> None:
        await respond_interaction_error(interaction, error, prefix="킬스위치 처리 실패")

    @discord.ui.button(label="Confirm Kill OFF", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.controller.is_admin(interaction.user.id):
            await interaction.response.send_message("관리자 권한이 없습니다.", ephemeral=True)
            return
        if self.submitted:
            await interaction.response.send_message("이미 처리 중입니다.", ephemeral=True)
            return
        self.submitted = True
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            self.controller.disable_kill_switch()
            await interaction.followup.send("킬스위치를 해제했습니다.", ephemeral=True)
            await self.panel_message.edit(
                embed=await self.controller.build_panel_embed(),
                view=ControlPanelView(self.controller),
            )
        except Exception as exc:
            self.submitted = False
            await interaction.followup.send(
                f"킬스위치 해제 실패: `{type(exc).__name__}: {exc}`",
                ephemeral=True,
            )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.edit_message(content="취소했습니다.", view=None)


class ConfirmServerRebootView(discord.ui.View):
    """Two-step confirmation before rebooting the VPS."""

    def __init__(self, controller: QuantControlController, panel_message: discord.Message):
        super().__init__(timeout=300)
        self.controller = controller
        self.panel_message = panel_message
        self.submitted = False

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item,
    ) -> None:
        await respond_interaction_error(interaction, error, prefix="VPS 리부트 처리 실패")

    @discord.ui.button(label="Confirm Reboot", style=discord.ButtonStyle.primary)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.controller.is_admin(interaction.user.id):
            await interaction.response.send_message("관리자 권한이 없습니다.", ephemeral=True)
            return
        if self.submitted:
            await interaction.response.send_message("이미 리부트 요청을 처리 중입니다.", ephemeral=True)
            return
        self.submitted = True
        for item in self.children:
            item.disabled = True

        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            await interaction.message.edit(view=self)
        except Exception:
            pass
        try:
            await asyncio.to_thread(self.controller.assert_reboot_allowed)
            await interaction.followup.send(
                "AWS VPS 리부트를 요청합니다. 약 1~2분 뒤 봇과 라이브 서비스가 다시 올라옵니다.",
                ephemeral=True,
            )
            await asyncio.to_thread(self.controller.launch_server_reboot)
        except Exception as exc:
            self.submitted = False
            await interaction.followup.send(
                f"VPS 리부트 실패: `{type(exc).__name__}: {exc}`",
                ephemeral=True,
            )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.edit_message(content="리부트를 취소했습니다.", view=None)


class ConfirmStartMicroView(discord.ui.View):
    """Two-step confirmation before enabling Rebate Micro auto-trading."""

    def __init__(self, controller: QuantControlController, panel_message: discord.Message):
        super().__init__(timeout=300)
        self.controller = controller
        self.panel_message = panel_message
        self.submitted = False

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item,
    ) -> None:
        await respond_interaction_error(interaction, error, prefix="Micro 시작 처리 실패")

    @discord.ui.button(label="Confirm Start Micro", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.controller.is_admin(interaction.user.id):
            await interaction.response.send_message("관리자 권한이 없습니다.", ephemeral=True)
            return
        if self.submitted:
            await interaction.response.send_message("이미 Micro 시작 요청을 처리 중입니다.", ephemeral=True)
            return
        self.submitted = True
        for item in self.children:
            item.disabled = True

        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            market = self.controller.start_micro_trader(updated_by=str(interaction.user.id))
            await interaction.followup.send(
                f"Rebate Micro 자동매매를 `{market}`에서 시작했습니다. 다음 판단 사이클부터 작동합니다.",
                ephemeral=True,
            )
            await self.panel_message.edit(
                embed=await self.controller.build_panel_embed(),
                view=ControlPanelView(self.controller),
            )
        except Exception as exc:
            self.submitted = False
            await interaction.followup.send(
                f"Micro 시작 실패: `{type(exc).__name__}: {exc}`",
                ephemeral=True,
            )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.edit_message(content="Micro 시작을 취소했습니다.", view=None)


class ConfirmProfileModeView(discord.ui.View):
    """Two-step confirmation before changing the live trading profile."""

    def __init__(
        self,
        controller: QuantControlController,
        *,
        profile: str,
        panel_message: discord.Message,
    ):
        super().__init__(timeout=300)
        self.controller = controller
        self.profile = profile
        self.panel_message = panel_message
        self.submitted = False

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item,
    ) -> None:
        await respond_interaction_error(interaction, error, prefix="모드 전환 처리 실패")

    @discord.ui.button(label="Confirm Mode Switch", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.controller.is_admin(interaction.user.id):
            await interaction.response.send_message("관리자 권한이 없습니다.", ephemeral=True)
            return
        if self.submitted:
            await interaction.response.send_message("이미 처리 중입니다.", ephemeral=True)
            return
        self.submitted = True
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            self.controller.set_profile_mode(
                self.profile,
                updated_by=str(interaction.user.id),
            )
            timeframe = default_timeframe_for_profile(self.profile)
            await interaction.followup.send(
                f"`{self.profile}` ({timeframe}) 모드로 전환했습니다. 다음 판단 사이클부터 적용됩니다.",
                ephemeral=True,
            )
            await self.panel_message.edit(
                embed=await self.controller.build_panel_embed(),
                view=ControlPanelView(self.controller),
            )
        except Exception as exc:
            self.submitted = False
            await interaction.followup.send(
                f"모드 전환 실패: `{type(exc).__name__}: {exc}`",
                ephemeral=True,
            )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.edit_message(content="취소했습니다.", view=None)


class QuantControlClient(discord.Client):
    """Discord client that owns and posts the control panel."""

    def __init__(
        self,
        *,
        controller: QuantControlController,
        channel_id: int,
        state_file: Path,
        sticky_panel: bool,
        sticky_bump_delay_seconds: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.controller = controller
        self.channel_id = channel_id
        self.state_file = state_file
        self.sticky_panel = sticky_panel
        self.sticky_bump_delay_seconds = sticky_bump_delay_seconds
        self._panel_lock = asyncio.Lock()
        self._pending_bump_task: Optional[asyncio.Task] = None

    async def setup_hook(self) -> None:
        self.add_view(ControlPanelView(self.controller))

    async def on_ready(self) -> None:
        print(f"Discord control bot logged in as {self.user}")
        await self.post_or_update_panel()

    async def on_message(self, message: discord.Message) -> None:
        """Move the control panel back to the bottom after channel activity."""
        if not self.sticky_panel:
            return
        if message.channel.id != self.channel_id:
            return
        if self.user is not None and message.author.id == self.user.id:
            return
        self.schedule_panel_bump()

    def schedule_panel_bump(self) -> None:
        """Debounce panel bumps so bursts of messages only repost once."""
        if self._pending_bump_task and not self._pending_bump_task.done():
            self._pending_bump_task.cancel()
        self._pending_bump_task = asyncio.create_task(self._delayed_panel_bump())

    async def _delayed_panel_bump(self) -> None:
        try:
            await asyncio.sleep(self.sticky_bump_delay_seconds)
            await self.bump_panel_to_bottom()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            print(f"Sticky panel bump failed: {type(exc).__name__}: {exc}")

    async def post_or_update_panel(self) -> None:
        channel = await self.resolve_panel_channel()

        embed = await self.controller.build_panel_embed()
        view = ControlPanelView(self.controller)
        message_id = load_panel_message_id(self.state_file)
        if message_id is not None:
            try:
                message = await channel.fetch_message(message_id)
                await message.edit(embed=embed, view=view)
                print(f"Discord control panel updated: message_id={message_id}")
                return
            except Exception as exc:
                print(f"Panel update failed, creating new panel: {type(exc).__name__}")

        message = await channel.send(embed=embed, view=view)
        save_panel_message_id(self.state_file, message.id)
        print(f"Discord control panel posted: message_id={message.id}")

    async def bump_panel_to_bottom(self) -> None:
        """Delete the old panel and send a fresh one as the latest message."""
        async with self._panel_lock:
            channel = await self.resolve_panel_channel()
            message_id = load_panel_message_id(self.state_file)
            if message_id is not None:
                try:
                    message = await channel.fetch_message(message_id)
                    await message.delete()
                except discord.NotFound:
                    pass
                except discord.Forbidden:
                    print("Sticky panel could not delete old panel: missing permission")
                except discord.HTTPException as exc:
                    print(f"Sticky panel could not delete old panel: {exc}")

            message = await channel.send(
                embed=await self.controller.build_panel_embed(),
                view=ControlPanelView(self.controller),
            )
            save_panel_message_id(self.state_file, message.id)
            print(f"Discord control panel bumped: message_id={message.id}")

    async def resolve_panel_channel(self):
        """Return the configured Discord text channel."""
        channel = self.get_channel(self.channel_id)
        if channel is None:
            channel = await self.fetch_channel(self.channel_id)
        return channel


def _risk_block_summary(report) -> str:
    """Format risk report blocking flags."""
    blocks = []
    if report.kill_switch_active:
        blocks.append("kill switch")
    if report.daily_breached:
        blocks.append("daily loss")
    if report.weekly_breached:
        blocks.append("weekly loss")
    if report.drawdown_breached:
        blocks.append("MDD")
    return ", ".join(blocks) if blocks else "none"


def build_rebate_recommendation_embed(
    candidates: list[RebateMarketCandidate],
) -> discord.Embed:
    """Build Discord embed for Rebate Micro candidate ranking."""
    color = COLOR_SUCCESS if any(c.expected_net_pct > 0 for c in candidates) else COLOR_WARNING
    fields = []
    for idx, candidate in enumerate(candidates[:5], 1):
        fields.append(
            DiscordField(
                f"{idx}. {candidate.market}",
                format_candidate_line(candidate),
                inline=False,
            )
        )
    if not fields:
        fields.append(
            DiscordField(
                "No candidate",
                "유동성/스프레드/변동성/기대값 필터를 통과한 후보가 없습니다.",
                inline=False,
            )
        )
    return to_discord_embed(
        DiscordEmbed(
            title="Rebate Micro market recommendations",
            description=(
                "유동성 -> 스프레드 -> 단기 변동성 -> 모멘텀/반전 -> 비용 이후 기대값 순으로 "
                "실시간 점수화했습니다.\n"
                f"```text\n{format_candidate_lines(candidates)}\n```"
            ),
            color=color,
            fields=tuple(fields),
        )
    )


def format_candidate_line(candidate: RebateMarketCandidate) -> str:
    """Format one Rebate Micro candidate."""
    return (
        f"score {candidate.score:.2f} | "
        f"24h {format_krw(candidate.liquidity_krw)} KRW | "
        f"spread {candidate.spread_pct * Decimal('100'):.3f}% | "
        f"vol {candidate.volatility_pct * Decimal('100'):.3f}% | "
        f"mom {candidate.momentum_pct * Decimal('100'):+.3f}% | "
        f"EV {candidate.expected_net_pct * Decimal('100'):+.3f}% | "
        f"{candidate.reason}"
    )


def build_manual_order_embed(
    *,
    title: str,
    action: str,
    amount: Optional[Decimal],
    volume: Optional[Decimal],
    order_uuid: str,
    state: str,
) -> discord.Embed:
    """Build an ephemeral manual order result embed."""
    fields = [
        DiscordField("Action", action),
        DiscordField("State", state),
        DiscordField("UUID", order_uuid, inline=False),
    ]
    if amount is not None:
        fields.append(DiscordField("KRW Amount", f"{format_krw(amount)} KRW"))
    if volume is not None:
        fields.append(
            DiscordField(
                "Volume",
                str(volume.quantize(VOLUME_QUANT, rounding=ROUND_DOWN)),
            )
        )
    return to_discord_embed(
        DiscordEmbed(
            title=title,
            description="Admin-only manual market order passed live safety gates.",
            color=COLOR_WARNING,
            fields=tuple(fields),
        )
    )


def systemctl_state(service_name: str) -> str:
    """Return compact systemd service state."""
    result = subprocess.run(
        ["systemctl", "is-active", service_name],
        text=True,
        capture_output=True,
        check=False,
    )
    state = (result.stdout or result.stderr).strip()
    return state or "unknown"


def read_service_log_tail(service_name: str, *, lines: int = 18) -> str:
    """Read the latest service logs for Discord display."""
    result = subprocess.run(
        ["journalctl", "-u", service_name, "-n", str(lines), "--no-pager"],
        text=True,
        capture_output=True,
        check=False,
    )
    output = result.stdout if result.returncode == 0 else result.stderr
    output = output.strip() or "No logs."
    if len(output) > 1800:
        output = output[-1800:]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Discord button control panel.")
    parser.add_argument("--market", default="KRW-ETH")
    parser.add_argument("--order-limit", type=int, default=5)
    parser.add_argument("--kill-switch-file", default="logs/KILL_SWITCH_ACTIVE")
    parser.add_argument("--risk-state-file", default="logs/eth_cash_micro_live_risk_state.json")
    parser.add_argument("--pilot-state-file", default="logs/eth_cash_micro_live_state.json")
    parser.add_argument("--profile-mode-file", default="logs/eth_cash_live_profile.json")
    parser.add_argument("--market-mode-file", default="logs/pulse_market.json")
    parser.add_argument("--auto-trade-file", default="logs/AUTO_TRADE_ENABLED")
    parser.add_argument("--force-start-file", default="logs/REBATE_MICRO_FORCE_START.json")
    parser.add_argument("--force-start-minutes", type=int, default=60)
    parser.add_argument("--turnover-file", default="logs/rebate_micro_turnover.json")
    parser.add_argument("--turnover-target-krw", type=Decimal, default=Decimal("100000000000"))
    parser.add_argument("--service-name", default="quant-eth-cash-live")
    parser.add_argument("--panel-state-file", default="logs/discord_control_panel_state.json")
    parser.add_argument("--max-manual-order-krw", type=Decimal, default=Decimal("25000"))
    parser.add_argument("--max-position-krw", type=Decimal, default=Decimal("150000"))
    parser.add_argument("--min-order-krw", type=Decimal, default=Decimal("5000"))
    parser.add_argument("--max-spread-pct", type=Decimal, default=Decimal("0.005"))
    parser.add_argument("--max-orders-per-day", type=int, default=12)
    parser.add_argument("--cooldown-minutes", type=int, default=60)
    parser.add_argument("--cooldown-seconds", type=int, default=None)
    parser.add_argument("--scanner-min-liquidity-krw", type=Decimal, default=Decimal("1000000000"))
    parser.add_argument("--scanner-max-spread-pct", type=Decimal, default=Decimal("0.0015"))
    parser.add_argument("--scanner-min-volatility-pct", type=Decimal, default=Decimal("0.00025"))
    parser.add_argument("--scanner-fee-round-trip-pct", type=Decimal, default=Decimal("0.0008"))
    parser.add_argument("--scanner-rebate-round-trip-pct", type=Decimal, default=Decimal("0.0008"))
    parser.add_argument("--scanner-slippage-buffer-pct", type=Decimal, default=Decimal("0.0002"))
    parser.add_argument("--scanner-prefilter-limit", type=int, default=30)
    parser.add_argument(
        "--no-sticky-panel",
        action="store_true",
        help="Do not repost the control panel after new channel messages.",
    )
    parser.add_argument("--sticky-bump-delay-seconds", type=float, default=2.0)
    parser.add_argument("--no-file-log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    setup_logging(log_level=cfg.log_level, enable_file_log=not args.no_file_log)
    control_channel_id = cfg.discord_control_target_channel_id
    if not cfg.discord_bot_token or not control_channel_id:
        raise RuntimeError(
            "DISCORD_BOT_TOKEN과 DISCORD_CONTROL_CHANNEL_ID 또는 "
            "DISCORD_CHANNEL_ID가 필요합니다."
        )

    controller = QuantControlController(
        config=cfg,
        market=args.market,
        order_limit=args.order_limit,
        kill_switch_path=Path(args.kill_switch_file),
        risk_state_path=Path(args.risk_state_file),
        pilot_state_path=Path(args.pilot_state_file),
        profile_mode_path=Path(args.profile_mode_file),
        market_mode_path=Path(args.market_mode_file),
        auto_trade_path=Path(args.auto_trade_file),
        force_start_path=Path(args.force_start_file),
        turnover_path=Path(args.turnover_file),
        service_name=args.service_name,
        admin_user_ids=parse_admin_user_ids(cfg.discord_admin_user_ids),
        max_manual_order_krw=args.max_manual_order_krw,
        max_position_krw=args.max_position_krw,
        min_order_krw=args.min_order_krw,
        max_spread_pct=args.max_spread_pct,
        max_orders_per_day=args.max_orders_per_day,
        cooldown_minutes=args.cooldown_minutes,
        cooldown_seconds=args.cooldown_seconds,
        force_start_minutes=args.force_start_minutes,
        turnover_target_krw=args.turnover_target_krw,
        scanner_min_liquidity_krw=args.scanner_min_liquidity_krw,
        scanner_max_spread_pct=args.scanner_max_spread_pct,
        scanner_min_volatility_pct=args.scanner_min_volatility_pct,
        scanner_fee_round_trip_pct=args.scanner_fee_round_trip_pct,
        scanner_rebate_round_trip_pct=args.scanner_rebate_round_trip_pct,
        scanner_slippage_buffer_pct=args.scanner_slippage_buffer_pct,
        scanner_prefilter_limit=args.scanner_prefilter_limit,
    )
    intents = discord.Intents.default()
    if not args.no_sticky_panel:
        intents.messages = True
    bot = QuantControlClient(
        controller=controller,
        channel_id=int(control_channel_id),
        state_file=Path(args.panel_state_file),
        sticky_panel=not args.no_sticky_panel,
        sticky_bump_delay_seconds=args.sticky_bump_delay_seconds,
        intents=intents,
    )
    bot.run(cfg.discord_bot_token)


if __name__ == "__main__":
    main()
