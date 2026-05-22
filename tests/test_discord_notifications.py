"""
Discord webhook notification tests.

The tests mock requests.post and never call Discord.
"""
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from notifications.discord import (
    COLOR_SUCCESS,
    DiscordEmbed,
    DiscordField,
    DiscordNotificationError,
    DiscordNotifier,
)


class FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


def test_disabled_notifier_does_not_send():
    print("\n[테스트 1] Webhook 미설정")
    print("-" * 60)

    notifier = DiscordNotifier(None)
    with patch("notifications.discord.requests.post") as post:
        sent = notifier.send(content="hello")

    assert sent is False
    assert post.call_count == 0
    print("  ✓ Webhook 없으면 네트워크 호출 없음")


def test_invalid_webhook_url_is_rejected():
    print("\n[테스트 2] Webhook URL 형식 검증")
    print("-" * 60)

    try:
        DiscordNotifier("https://example.com/not-discord")
        raise AssertionError("잘못된 Discord webhook URL이 통과됨")
    except ValueError:
        pass

    print("  ✓ Discord webhook URL이 아니면 거부")


def test_send_embed_payload_without_leaking_url():
    print("\n[테스트 3] Embed 전송 payload")
    print("-" * 60)

    secret_url = "https://discord.com/api/webhooks/1234567890/secret_token"
    notifier = DiscordNotifier(secret_url)
    embed = DiscordEmbed(
        title="LIVE order requested",
        description="test trade",
        color=COLOR_SUCCESS,
        fields=(DiscordField("Market", "KRW-ETH"),),
    )

    with patch("notifications.discord.requests.post", return_value=FakeResponse(204)) as post:
        sent = notifier.send(embeds=(embed,))

    assert sent is True
    assert post.call_count == 1
    kwargs = post.call_args.kwargs
    assert kwargs["json"]["username"] == "Quant System"
    assert kwargs["json"]["embeds"][0]["title"] == "LIVE order requested"
    assert secret_url not in str(kwargs["json"])

    print("  ✓ Embed payload 생성 및 URL 비노출")


def test_discord_error_is_explicit():
    print("\n[테스트 4] Discord 실패 응답")
    print("-" * 60)

    notifier = DiscordNotifier("https://discord.com/api/webhooks/1234567890/secret_token")
    with patch("notifications.discord.requests.post", return_value=FakeResponse(429)):
        try:
            notifier.send(content="rate limited")
            raise AssertionError("Discord 실패 응답이 통과됨")
        except DiscordNotificationError:
            pass

    print("  ✓ Discord 실패 응답은 명시적 에러")


def main():
    print("=" * 60)
    print("Discord 알림 테스트")
    print("=" * 60)

    tests = [
        test_disabled_notifier_does_not_send,
        test_invalid_webhook_url_is_rejected,
        test_send_embed_payload_without_leaking_url,
        test_discord_error_is_explicit,
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
