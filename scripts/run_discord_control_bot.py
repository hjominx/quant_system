"""Run an interactive Discord control panel bot.

This is separate from webhook alerts. It uses a Discord bot token and button
interactions to provide a small operations console for the live trader.

The panel does not place market orders. Mutating controls are limited to the
local kill switch file, and require DISCORD_ADMIN_USER_IDS.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
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
from risk.live_guard import (
    RiskLimits,
    evaluate_risk,
    format_pct,
    load_risk_state,
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
        service_name: str,
        admin_user_ids: set[str],
    ):
        self.config = config
        self.market = market
        self.order_limit = order_limit
        self.kill_switch_path = kill_switch_path
        self.risk_state_path = risk_state_path
        self.service_name = service_name
        self.admin_user_ids = admin_user_ids
        self.client = BithumbClient.from_config(
            config,
            dry_run=True,
            allow_live_orders=False,
        )
        self.lock = asyncio.Lock()

    def is_admin(self, user_id: int) -> bool:
        """Return whether this Discord user can run mutating controls."""
        return str(user_id) in self.admin_user_ids

    async def collect_snapshot_embeds(self, report: str) -> list[discord.Embed]:
        """Collect Bithumb account state and convert it to Discord embeds."""
        async with self.lock:
            snapshot = await asyncio.to_thread(
                collect_account_snapshot,
                self.client,
                market=self.market,
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
        async with self.lock:
            snapshot = await asyncio.to_thread(
                collect_account_snapshot,
                self.client,
                market=self.market,
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
                    _risk_block_summary(report),
                    inline=False,
                ),
            ),
        )
        return to_discord_embed(embed)

    async def build_panel_embed(self) -> discord.Embed:
        """Build the public control panel embed."""
        service_state = await asyncio.to_thread(systemctl_state, self.service_name)
        kill_state = "ACTIVE" if self.kill_switch_path.exists() else "inactive"
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
                "조회 버튼은 상태를 읽고, 킬스위치는 관리자만 실행합니다."
            ),
            color=color,
            fields=(
                DiscordField("Market", self.market),
                DiscordField("Live service", service_state),
                DiscordField("Kill switch", kill_state),
                DiscordField("Alerts", "Hourly status + order/risk/error webhooks"),
                DiscordField("Admin gate", admin_state),
                DiscordField("Updated", datetime.now().isoformat(timespec="minutes")),
            ),
        )
        return to_discord_embed(embed)

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


class ControlPanelView(discord.ui.View):
    """Persistent Discord button panel."""

    def __init__(self, controller: QuantControlController):
        super().__init__(timeout=None)
        self.controller = controller

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
            f"`{interaction.user.id}`를 추가해야 킬스위치 버튼을 쓸 수 있습니다."
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

    @discord.ui.button(label="Refresh Panel", style=discord.ButtonStyle.secondary, custom_id="quant:refresh", row=1)
    async def refresh_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True)
        await interaction.message.edit(
            embed=await self.controller.build_panel_embed(),
            view=self,
        )
        await interaction.followup.send("패널을 새로고침했습니다.", ephemeral=True)

    @discord.ui.button(label="Kill ON", style=discord.ButtonStyle.danger, custom_id="quant:kill:on", row=2)
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

    @discord.ui.button(label="Kill OFF", style=discord.ButtonStyle.success, custom_id="quant:kill:off", row=2)
    async def kill_off_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not await self._require_admin(interaction):
            return
        await interaction.response.send_message(
            "정말 킬스위치를 해제할까요?",
            view=ConfirmKillSwitchOffView(self.controller, interaction.message),
            ephemeral=True,
        )


class ConfirmKillSwitchOffView(discord.ui.View):
    """Two-step confirmation before removing the kill switch."""

    def __init__(self, controller: QuantControlController, panel_message: discord.Message):
        super().__init__(timeout=30)
        self.controller = controller
        self.panel_message = panel_message

    @discord.ui.button(label="Confirm Kill OFF", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.controller.is_admin(interaction.user.id):
            await interaction.response.send_message("관리자 권한이 없습니다.", ephemeral=True)
            return
        self.controller.disable_kill_switch()
        await interaction.response.edit_message(content="킬스위치를 해제했습니다.", view=None)
        await self.panel_message.edit(
            embed=await self.controller.build_panel_embed(),
            view=ControlPanelView(self.controller),
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.controller = controller
        self.channel_id = channel_id
        self.state_file = state_file

    async def setup_hook(self) -> None:
        self.add_view(ControlPanelView(self.controller))

    async def on_ready(self) -> None:
        print(f"Discord control bot logged in as {self.user}")
        await self.post_or_update_panel()

    async def post_or_update_panel(self) -> None:
        channel = self.get_channel(self.channel_id)
        if channel is None:
            channel = await self.fetch_channel(self.channel_id)

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
    parser.add_argument("--service-name", default="quant-eth-cash-live")
    parser.add_argument("--panel-state-file", default="logs/discord_control_panel_state.json")
    parser.add_argument("--no-file-log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    setup_logging(log_level=cfg.log_level, enable_file_log=not args.no_file_log)
    if not cfg.discord_bot_token or not cfg.discord_channel_id:
        raise RuntimeError("DISCORD_BOT_TOKEN과 DISCORD_CHANNEL_ID가 필요합니다.")

    controller = QuantControlController(
        config=cfg,
        market=args.market,
        order_limit=args.order_limit,
        kill_switch_path=Path(args.kill_switch_file),
        risk_state_path=Path(args.risk_state_file),
        service_name=args.service_name,
        admin_user_ids=parse_admin_user_ids(cfg.discord_admin_user_ids),
    )
    intents = discord.Intents.default()
    bot = QuantControlClient(
        controller=controller,
        channel_id=int(cfg.discord_channel_id),
        state_file=Path(args.panel_state_file),
        intents=intents,
    )
    bot.run(cfg.discord_bot_token)


if __name__ == "__main__":
    main()
