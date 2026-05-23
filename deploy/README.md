# AWS deployment notes

This directory stores non-secret systemd service templates for the Lightsail
pilot server. Secrets stay in `.env` on the server and must not be committed.

## ETH/Cash live service

Current pilot profile:

- Strategy profile: `stable_quality`
- Timeframe: `4h`
- Loop interval: 15 minutes, aligned to clock boundaries
- Max order: 50,000 KRW
- Max ETH position: 200,000 KRW
- Cooldown: 15 minutes
- Max orders per day: 24
- Discord status: hourly, on the hour

Install or refresh the service on AWS:

```bash
cd ~/quant_system
git pull --ff-only

sudo cp deploy/quant-eth-cash-live.service /etc/systemd/system/quant-eth-cash-live.service
sudo systemctl daemon-reload
sudo systemctl restart quant-eth-cash-live

systemctl cat quant-eth-cash-live | grep -E -- '--profile|--max-order-krw|--max-position-krw|--cooldown-minutes|--max-orders-per-day'
journalctl -u quant-eth-cash-live -n 80 --no-pager
```

The kill switch is still controlled by `logs/KILL_SWITCH_ACTIVE` on the server.
