"""
Discord webhook notifications.

Only incoming webhooks are used here. The webhook URL is a secret and must not
be printed or committed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import requests


DISCORD_WEBHOOK_PREFIXES = (
    "https://discord.com/api/webhooks/",
    "https://discordapp.com/api/webhooks/",
)

COLOR_INFO = 0x3498DB
COLOR_SUCCESS = 0x2ECC71
COLOR_WARNING = 0xF1C40F
COLOR_DANGER = 0xE74C3C


class DiscordNotificationError(RuntimeError):
    """Raised when Discord rejects or cannot receive a notification."""


@dataclass(frozen=True)
class DiscordField:
    """One Discord embed field."""
    name: str
    value: str
    inline: bool = True

    def to_payload(self) -> dict:
        return {
            "name": _limit_text(self.name, 256),
            "value": _limit_text(self.value, 1024),
            "inline": self.inline,
        }


@dataclass(frozen=True)
class DiscordEmbed:
    """Small typed wrapper around a Discord embed payload."""
    title: str
    description: str
    color: int = COLOR_INFO
    fields: tuple[DiscordField, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict:
        payload = {
            "title": _limit_text(self.title, 256),
            "description": _limit_text(self.description, 4096),
            "color": self.color,
        }
        if self.fields:
            payload["fields"] = [field.to_payload() for field in self.fields[:25]]
        return payload


class DiscordNotifier:
    """Thin, fail-explicit Discord webhook client."""

    def __init__(
        self,
        webhook_url: Optional[str],
        *,
        username: str = "Quant System",
        timeout: float = 5.0,
    ):
        self.webhook_url = (webhook_url or "").strip()
        self.username = username
        self.timeout = timeout
        if self.webhook_url:
            validate_webhook_url(self.webhook_url)

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    @classmethod
    def from_config(cls, config) -> "DiscordNotifier":
        return cls(getattr(config, "discord_webhook_url", None))

    def send(
        self,
        *,
        content: Optional[str] = None,
        embeds: Iterable[DiscordEmbed] = (),
    ) -> bool:
        """Send a Discord message. Returns False when no webhook is configured."""
        embeds = tuple(embeds)
        if not self.enabled:
            return False
        if not content and not embeds:
            raise ValueError("content or embeds is required")

        payload: dict = {"username": self.username}
        if content:
            payload["content"] = _limit_text(content, 2000)
        if embeds:
            payload["embeds"] = [embed.to_payload() for embed in embeds[:10]]

        response = requests.post(
            self.webhook_url,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code in (200, 204):
            return True
        raise DiscordNotificationError(
            f"Discord webhook failed with HTTP {response.status_code}"
        )


def validate_webhook_url(url: str) -> None:
    """Reject obviously invalid webhook URLs before runtime."""
    if not any(url.startswith(prefix) for prefix in DISCORD_WEBHOOK_PREFIXES):
        raise ValueError("DISCORD_WEBHOOK_URL must be a Discord webhook URL")


def _limit_text(value: str, limit: int) -> str:
    value = str(value)
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)] + "..."
