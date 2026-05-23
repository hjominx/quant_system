# AWS deployment notes

This directory stores non-secret systemd service templates for the Lightsail
pilot server. Secrets stay in `.env` on the server and must not be committed.

## ETH/Cash live service

Current pilot profile:

- Strategy profile: `stable_quality`
- Runtime modes: `stable_quality` (`4h`), `scalp_active` (`5m`), `scalp_1m` (`1m`), or `pulse_10s` (`1m` ticker pulse)
- Mode file: `logs/eth_cash_live_profile.json`
- Pulse market file: `logs/pulse_market.json`
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

Install or refresh the service on AWS:

```bash
cd ~/quant_system
git pull --ff-only

sudo cp deploy/quant-eth-cash-live.service /etc/systemd/system/quant-eth-cash-live.service
sudo cp deploy/quant-discord-control.service /etc/systemd/system/quant-discord-control.service
sudo systemctl daemon-reload
sudo systemctl restart quant-eth-cash-live
sudo systemctl restart quant-discord-control

systemctl cat quant-eth-cash-live | grep -E -- '--profile|--profile-mode-file|--market-mode-file|--max-order-krw|--max-position-krw|--cooldown-seconds|--max-orders-per-day|--interval-seconds'
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
```
