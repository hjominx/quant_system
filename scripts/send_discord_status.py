"""Send one Discord account/status report.

This script never places orders. It only reads Bithumb price/account/order data
and posts the selected report to Discord.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.bithumb_client import BithumbClient
from notifications.discord import DiscordNotifier
from notifications.discord_status import (
    build_snapshot_embeds,
    collect_account_snapshot,
    select_snapshot_embeds,
)
from utils.config import get_config
from utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a Discord account/status report.")
    parser.add_argument("--market", default="KRW-ETH")
    parser.add_argument(
        "--report",
        choices=["all", "price", "assets", "orders"],
        default="all",
        help="Report section to send.",
    )
    parser.add_argument("--order-limit", type=int, default=5)
    parser.add_argument("--title-prefix", default="Manual status")
    parser.add_argument("--no-file-log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    setup_logging(log_level=cfg.log_level, enable_file_log=not args.no_file_log)
    notifier = DiscordNotifier.from_config(cfg)
    if not notifier.enabled:
        raise RuntimeError("DISCORD_WEBHOOK_URL이 설정되어 있지 않습니다.")

    client = BithumbClient.from_config(cfg, dry_run=True, allow_live_orders=False)
    try:
        snapshot = collect_account_snapshot(
            client,
            market=args.market,
            order_limit=args.order_limit,
        )
        embeds = build_snapshot_embeds(
            snapshot,
            title_prefix=args.title_prefix,
        )
        notifier.send(embeds=select_snapshot_embeds(embeds, args.report))
        print(f"Discord report sent: report={args.report} market={args.market}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
