# DecaBot Bridge

DecaBot is a QuantRabbit-derived live experiment, but it is operationally
separate from the main QuantRabbit trader.

## Boundary

- QuantRabbit main live trader root: `/Users/tossaki/App/QuantRabbit-live`
- DecaBot root: `/Users/tossaki/App/DecaBot`
- QuantRabbit main OANDA account: configured by `.env.local`
- DecaBot OANDA account: configured by `/Users/tossaki/App/DecaBot/.env`
- DecaBot Slack messages still appear in the QuantRabbit trades channel, but
  they are posted by DecaBot monitor text such as `DecaBot-AI` / `DecaBot`.

Do not route DecaBot orders through QuantRabbit `LiveOrderGateway`. DecaBot is
an explicit separate-account experiment with its own OANDA client, data, logs,
and launchd agents.

## QR Entry Point

From the QuantRabbit root:

```bash
./scripts/qr-decabot.sh status
./scripts/qr-decabot.sh logs ai
./scripts/qr-decabot.sh cycle
./scripts/qr-decabot.sh monitor
./scripts/qr-decabot.sh start
./scripts/qr-decabot.sh stop
./scripts/qr-decabot.sh shell
```

`cycle` runs one DecaBot-AI cycle immediately. If DecaBot
`data/ai_config.json` has `"dry_run": false`, that command can place, close, or
adjust live OANDA orders in the DecaBot account.

`start` and `stop` intentionally manage only:

- `com.decabot.ai` — 10-minute AI cycle
- `com.decabot.monitor` — 5-minute OANDA/Slack monitor
- `com.decabot.review` — daily review

They do not load `com.decabot.live`. The legacy 50-second range engine is a
separate strategy and should remain off unless explicitly requested.

## Weekend Guard

`qr-weekend-market-off` runs every Saturday 06:00 JST and
`qr-weekend-market-off` checks Saturday 06:00 and 07:00 JST and pauses only after the DST-aware New York weekly close. `qr-weekend-market-on` checks every Monday at 06:00 and 07:00 JST, restoring at the first DST-aware New York weekly market open. The shared weekend switcher
snapshots and stops the same DecaBot launchd labels managed by `start` / `stop`:

- `com.decabot.ai`
- `com.decabot.monitor`
- `com.decabot.review`

On Monday it restores only labels that were loaded in the Saturday snapshot.
It does not manage `com.decabot.live`.

## Current Intent

The bridge exists so a QuantRabbit operator can inspect and operate the
DecaBot derivative without remembering another project path. It is not a
shared execution path, not a gateway bypass, and not a second QuantRabbit
trader scheduler.
