# AWS deployment notes

This directory stores non-secret systemd service templates for the Lightsail
pilot server. Secrets stay in `.env` on the server and must not be committed.

## ETH/Cash live service

Current pilot profile:

- Strategy profile: `stable_quality`
- Runtime modes: `stable_quality` (`4h`), `scalp_active` (`5m`), `scalp_1m` (`1m`), `pulse_10s` (`1m` ticker pulse), or `rebate_micro` (`1m` start/stop micro)
- Mode file: `logs/eth_cash_live_profile.json`
- Pulse market file: `logs/pulse_market.json`
- Auto-trade start file: `logs/AUTO_TRADE_ENABLED`
- Force-start file: `logs/REBATE_MICRO_FORCE_START.json`
- Rebate Micro turnover file: `logs/rebate_micro_turnover.json`
- Rebate Micro turnover target: 100,000,000,000 KRW
- Loop interval: 1 second
- Max order: 100,000 KRW
- Max selected-market position: 200,000 KRW
- Cooldown: 0 seconds
- Max orders per day: 300
- Discord status: hourly, on the hour
- Scalp modes allow controlled averaging up to 2 add-on buys per position,
  then keep hard-stop behavior if the position does not recover.
- Pulse mode buys up to 100,000 KRW every 10 minutes on the selected KRW
  market, starts TP/SL checks 3 seconds after each pulse buy, and uses
  +0.30% take-profit / -0.04% stop-loss.
- Rebate Micro mode is enabled only while `logs/AUTO_TRADE_ENABLED` exists.
  It attempts fast selected-market micro entries only when the 1m micro
  signal filter is active, uses dynamic +0.10% to +2.00% take-profit and
  -0.02% hard stop-loss, and still respects spread, kill switch,
  open-order, position, and daily order gates.
- Before each Rebate Micro BUY, the strategy estimates post-rebate expected
  edge from spread, round-trip fee, rebate, slippage buffer, micro momentum,
  micro volatility, and reward/risk. It holds if the estimate is not positive
  enough.
- In Rebate Micro, daily/weekly/MDD loss-rate gates do not block automated
  orders. Kill switch, spread, open-order, balance, position, and daily
  order-count gates still block orders.
- Rebate Micro avoids pure churn by disabling add-while-holding and adding
  a short post-sell reentry guard before opening a new position.
- The Discord `Recommend` button scans KRW markets by liquidity, spread, 1m
  short volatility, momentum/reversal signal, and estimated post-cost edge.
  Candidate buttons start the selected market. Rebate Micro does not use
  daily/weekly/MDD loss-rate gates as automated-order blockers.
- The Discord `Set Market` button no longer opens a text input. It scans live
  KRW markets and shows a dropdown sorted by recent 1m volatility.

Install or refresh the service on AWS:

```bash
cd ~/quant_system
git pull --ff-only

sudo cp deploy/quant-eth-cash-live.service /etc/systemd/system/quant-eth-cash-live.service
sudo cp deploy/quant-discord-control.service /etc/systemd/system/quant-discord-control.service
sudo systemctl daemon-reload
sudo systemctl restart quant-eth-cash-live
sudo systemctl restart quant-discord-control

systemctl cat quant-eth-cash-live | grep -E -- '--profile|--profile-mode-file|--market-mode-file|--auto-trade-file|--force-start-file|--turnover-file|--max-order-krw|--max-position-krw|--cooldown-seconds|--max-orders-per-day|--interval-seconds'
journalctl -u quant-eth-cash-live -n 80 --no-pager
```

The kill switch is still controlled by `logs/KILL_SWITCH_ACTIVE` on the server.

Switch modes from SSH if Discord is unavailable:

```bash
cd ~/quant_system

# Stable 4h mode
python - <<'PY'
from pathlib import Path
from scripts.run_eth_cash_micro_live import save_runtime_profile_mode
save_runtime_profile_mode(Path("logs/eth_cash_live_profile.json"), "stable_quality", updated_by="ssh")
PY

# Scalp 5m mode
python - <<'PY'
from pathlib import Path
from scripts.run_eth_cash_micro_live import save_runtime_profile_mode
save_runtime_profile_mode(Path("logs/eth_cash_live_profile.json"), "scalp_active", updated_by="ssh")
PY

# Scalp 1m mode
python - <<'PY'
from pathlib import Path
from scripts.run_eth_cash_micro_live import save_runtime_profile_mode
save_runtime_profile_mode(Path("logs/eth_cash_live_profile.json"), "scalp_1m", updated_by="ssh")
PY

# Pulse mode on a selected KRW alt market
python - <<'PY'
from pathlib import Path
from scripts.run_eth_cash_micro_live import save_runtime_market_mode, save_runtime_profile_mode
save_runtime_market_mode(Path("logs/pulse_market.json"), "KRW-SOL", updated_by="ssh")
save_runtime_profile_mode(Path("logs/eth_cash_live_profile.json"), "pulse_10s", updated_by="ssh")
PY

# Start Rebate Micro on a selected KRW alt market
python - <<'PY'
from pathlib import Path
from scripts.run_eth_cash_micro_live import enable_auto_trade, save_runtime_market_mode, save_runtime_profile_mode
save_runtime_market_mode(Path("logs/pulse_market.json"), "KRW-SOL", updated_by="ssh")
save_runtime_profile_mode(Path("logs/eth_cash_live_profile.json"), "rebate_micro", updated_by="ssh")
enable_auto_trade(Path("logs/AUTO_TRADE_ENABLED"), profile="rebate_micro", updated_by="ssh")
PY

# Force-start Rebate Micro on a selected KRW alt market for 60 minutes
python - <<'PY'
from pathlib import Path
from scripts.run_eth_cash_micro_live import (
    enable_auto_trade,
    enable_risk_override,
    save_runtime_market_mode,
    save_runtime_profile_mode,
)
market = "KRW-SOL"
save_runtime_market_mode(Path("logs/pulse_market.json"), market, updated_by="ssh")
save_runtime_profile_mode(Path("logs/eth_cash_live_profile.json"), "rebate_micro", updated_by="ssh")
enable_auto_trade(Path("logs/AUTO_TRADE_ENABLED"), profile="rebate_micro", updated_by="ssh")
enable_risk_override(
    Path("logs/REBATE_MICRO_FORCE_START.json"),
    profile="rebate_micro",
    market=market,
    updated_by="ssh",
    minutes=60,
)
PY

# Stop Rebate Micro without forcing liquidation
rm -f logs/AUTO_TRADE_ENABLED logs/REBATE_MICRO_FORCE_START.json

# Show Rebate Micro turnover progress
python - <<'PY'
from pathlib import Path
from scripts.run_eth_cash_micro_live import (
    format_rebate_turnover_progress,
    load_rebate_turnover_progress,
)
print(format_rebate_turnover_progress(
    load_rebate_turnover_progress(Path("logs/rebate_micro_turnover.json"))
))
PY
```
