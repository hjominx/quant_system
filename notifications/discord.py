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


class DiscordBotError(RuntimeError):
    """Raised when Discord bot API calls fail."""


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


@dataclass(frozen=True)
class DiscordMessage:
    """Small wrapper for a Discord channel message."""
    id: str
    content: str
    author_id: str
    author_is_bot: bool = False

    @classmethod
    def from_payload(cls, payload: dict) -> "DiscordMessage":
        author = payload.get("author") or {}
        return cls(
            id=str(payload["id"]),
            content=str(payload.get("content") or ""),
            author_id=str(author.get("id") or ""),
            author_is_bot=bool(author.get("bot", False)),
        )


class DiscordBotClient:
    """Minimal Discord REST client for polling command messages.

    This intentionally avoids an inbound web server. A local process can poll a
    private channel, parse a tiny command prefix, and post embeds back.
    """

    api_base = "https://discord.com/api/v10"

    def __init__(
        self,
        bot_token: Optional[str],
        channel_id: Optional[str],
        *,
        timeout: float = 5.0,
    ):
        self.bot_token = (bot_token or "").strip()
        self.channel_id = (channel_id or "").strip()
        self.timeout = timeout
        if self.channel_id and not self.channel_id.isdigit():
            raise ValueError("DISCORD_CHANNEL_ID must be a numeric Discord channel ID")

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.channel_id)

    @classmethod
    def from_config(cls, config) -> "DiscordBotClient":
        return cls(
            getattr(config, "discord_bot_token", None),
            getattr(config, "discord_channel_id", None),
        )

    def get_recent_messages(self, *, limit: int = 20) -> list[DiscordMessage]:
        """Fetch recent channel messages newest-first."""
        if not self.enabled:
            return []
        if not 1 <= limit <= 100:
            raise ValueError("limit은 1~100 사이여야 합니다.")
        response = requests.get(
            f"{self.api_base}/channels/{self.channel_id}/messages",
            headers=self._headers,
            params={"limit": limit},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise DiscordBotError(
                f"Discord messages fetch failed with HTTP {response.status_code}"
            )
        return [DiscordMessage.from_payload(item) for item in response.json()]

    def send(
        self,
        *,
        content: Optional[str] = None,
        embeds: Iterable[DiscordEmbed] = (),
    ) -> bool:
        """Post a message to the configured channel."""
        embeds = tuple(embeds)
        if not self.enabled:
            return False
        if not content and not embeds:
            raise ValueError("content or embeds is required")

        payload: dict = {}
        if content:
            payload["content"] = _limit_text(content, 2000)
        if embeds:
            payload["embeds"] = [embed.to_payload() for embed in embeds[:10]]

        response = requests.post(
            f"{self.api_base}/channels/{self.channel_id}/messages",
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code in (200, 201):
            return True
        raise DiscordBotError(
            f"Discord bot message failed with HTTP {response.status_code}"
        )

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json",
        }


def validate_webhook_url(url: str) -> None:
    """Reject obviously invalid webhook URLs before runtime."""
    if not any(url.startswith(prefix) for prefix in DISCORD_WEBHOOK_PREFIXES):
        raise ValueError("DISCORD_WEBHOOK_URL must be a Discord webhook URL")


def _limit_text(value: str, limit: int) -> str:
    value = str(value)
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)] + "..."
