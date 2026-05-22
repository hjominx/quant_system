"""Poll Discord commands and reply with Bithumb account/status reports.

Webhook URLs can only send messages. This bot-token based poller makes simple
outbound REST calls, so it works from a home Mac or VPS without exposing an
inbound HTTP endpoint.

Commands:
    !q help
    !q status
    !q price
    !q assets
    !q orders
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.bithumb_client import BithumbClient
from notifications.discord import DiscordBotClient
from notifications.discord_status import (
    build_snapshot_embeds,
    collect_account_snapshot,
    select_snapshot_embeds,
)
from utils.config import get_config
from utils.logger import setup_logging


HELP_TEXT = """Quant commands
`!q status` - price + assets + recent orders
`!q price` - ETH price/orderbook snapshot
`!q assets` - KRW/coin balance snapshot
`!q orders` - recent filled/canceled/open orders
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Discord polling query bot.")
    parser.add_argument("--market", default="KRW-ETH")
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--order-limit", type=int, default=5)
    parser.add_argument("--state-file", default="logs/discord_query_bot_state.json")
    parser.add_argument("--process-existing", action="store_true")
    parser.add_argument("--no-file-log", action="store_true")
    return parser.parse_args()


def load_last_message_id(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    value = raw.get("last_message_id")
    return str(value) if value else None


def save_last_message_id(path: Path, message_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"last_message_id": str(message_id)}, f, ensure_ascii=False, indent=2)


def is_newer(message_id: str, last_message_id: str | None) -> bool:
    if last_message_id is None:
        return True
    return int(message_id) > int(last_message_id)


def normalize_command(content: str, prefix: str) -> str | None:
    content = content.strip()
    if not content.startswith(prefix):
        return None
    parts = content[len(prefix):].strip().split()
    if not parts:
        return "help"
    return parts[0].lower()


def handle_command(
    *,
    command: str,
    bot: DiscordBotClient,
    client: BithumbClient,
    market: str,
    order_limit: int,
) -> None:
    if command in {"help", "도움", "도움말"}:
        bot.send(content=HELP_TEXT)
        return

    aliases = {
        "status": "all",
        "상태": "all",
        "price": "price",
        "시세": "price",
        "assets": "assets",
        "asset": "assets",
        "잔고": "assets",
        "자산": "assets",
        "orders": "orders",
        "order": "orders",
        "체결": "orders",
        "거래": "orders",
    }
    report = aliases.get(command)
    if report is None:
        bot.send(content=f"Unknown command: `{command}`\n\n{HELP_TEXT}")
        return

    snapshot = collect_account_snapshot(
        client,
        market=market,
        order_limit=order_limit,
    )
    embeds = build_snapshot_embeds(
        snapshot,
        title_prefix="Query status",
    )
    bot.send(embeds=select_snapshot_embeds(embeds, report))


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 3:
        raise ValueError("--poll-seconds는 3 이상을 권장/강제합니다.")

    cfg = get_config()
    setup_logging(log_level=cfg.log_level, enable_file_log=not args.no_file_log)
    bot = DiscordBotClient.from_config(cfg)
    if not bot.enabled:
        raise RuntimeError("DISCORD_BOT_TOKEN과 DISCORD_CHANNEL_ID가 필요합니다.")

    prefix = cfg.discord_command_prefix
    state_path = Path(args.state_file)
    last_message_id = load_last_message_id(state_path)

    if last_message_id is None and not args.process_existing:
        messages = bot.get_recent_messages(limit=20)
        if messages:
            last_message_id = max(messages, key=lambda item: int(item.id)).id
            save_last_message_id(state_path, last_message_id)
        print(f"Discord query bot initialized. prefix={prefix}")

    client = BithumbClient.from_config(cfg, dry_run=True, allow_live_orders=False)
    try:
        print(f"Discord query bot running. prefix={prefix} poll={args.poll_seconds}s")
        while True:
            try:
                messages = bot.get_recent_messages(limit=20)
                new_messages = [
                    message
                    for message in messages
                    if is_newer(message.id, last_message_id) and not message.author_is_bot
                ]
                for message in sorted(new_messages, key=lambda item: int(item.id)):
                    command = normalize_command(message.content, prefix)
                    if command is not None:
                        handle_command(
                            command=command,
                            bot=bot,
                            client=client,
                            market=args.market,
                            order_limit=args.order_limit,
                        )
                    last_message_id = message.id
                    save_last_message_id(state_path, last_message_id)
            except Exception as exc:
                print(f"[Discord query bot error] {type(exc).__name__}: {exc}")
                try:
                    bot.send(content=f"Query bot error: `{type(exc).__name__}`")
                except Exception:
                    pass
            time.sleep(args.poll_seconds)
    finally:
        client.close()


if __name__ == "__main__":
    main()
