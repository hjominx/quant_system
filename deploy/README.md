# AWS deployment notes

This directory stores non-secret systemd service templates for the Lightsail
pilot server. Secrets stay in `.env` on the server and must not be committed.

## ETH/Cash live service

Current pilot profile:

- Strategy profile: `stable_quality`
- Runtime modes: `stable_quality` (`4h`) or `scalp_active` (`5m`)
- Mode file: `logs/eth_cash_live_profile.json`
- Loop interval: 5 minutes, aligned to clock boundaries
- Max order: 50,000 KRW
- Max ETH position: 200,000 KRW
- Cooldown: 15 minutes
- Max orders per day: 48
- Discord status: hourly, on the hour

Install or refresh the service on AWS:

```bash
cd ~/quant_system
git pull --ff-only

sudo cp deploy/quant-eth-cash-live.service /etc/systemd/system/quant-eth-cash-live.service
sudo cp deploy/quant-discord-control.service /etc/systemd/system/quant-discord-control.service
sudo systemctl daemon-reload
sudo systemctl restart quant-eth-cash-live
sudo systemctl restart quant-discord-control

systemctl cat quant-eth-cash-live | grep -E -- '--profile|--profile-mode-file|--max-order-krw|--max-position-krw|--cooldown-minutes|--max-orders-per-day|--interval-minutes'
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
```
