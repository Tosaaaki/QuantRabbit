from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from validate_trader_state import _tautological_dead_closure_phrase, validate_state
from quality_audit import load_logged_trade_ids, parse_state_positions


VALID_STATE_TEXT = """# Trader State - 2026-04-23
**Last Updated**: 2026-04-23 19:45 UTC

## Self-check
Entries today: 9 fills / 6 new entry orders / 20 rejects. Sessions elapsed: ~1. Margin used: 12.7%.

## Slack Response
Pending user ts: none
Latest handled user ts: 1776944936.909969
Message class: question-observation
Trade consequence: held existing plan
Reply receipt: python3 tools/slack_post.py "Holding the plan." --channel C0APAELAQDN --reply-to 1776944936.909969

## Hot Updates
- 2026-04-23 19:48 UTC | EUR_USD retest short armed | fresh retest LIMIT id=`469529` carries the direct-USD short idea.

## Market Narrative
Best expression NOW: EUR_JPY SHORT live trade id=`469502`.
Backup vehicle: USD_JPY LONG LIMIT id=`469528`.
Next fresh risk allowed NOW: USD_JPY LONG LIMIT id=`469528`.
20-minute backup trigger armed NOW: GBP_JPY LONG LIMIT id=`469521`.
Best direct-USD seat NOW: EUR_USD SHORT LIMIT id=`469529`.

## Positions (Current)
Open trades: 0 LONG / 1 SHORT / 1 pair.
Current book: EUR_JPY SHORT trade id=`469502` is live on OANDA.
Pending orders: USD_JPY LONG LIMIT id=`469528`; EUR_USD SHORT LIMIT id=`469529`; GBP_JPY LONG LIMIT id=`469521`.

## OODA / Decision Journal
Observe: one live cross receipt plus three armed limits.
Orient: passive entries only.
Decide: hold live EUR_JPY and leave armed limits.
Act: no market chase.

## Deepening Pass
Best direct-USD seat: EUR_USD SHORT LIMIT id=`469529`.
Best cross seat: EUR_JPY SHORT id=`469502`.

## Directional Mix
Positions: 0 LONG / 1 SHORT / 1 pair.

## 7-Pair Scan
USD_JPY: Best expression LONG LIMIT | I would enter if id=`469528` fills.
EUR_USD: Best expression SHORT LIMIT | I would enter if id=`469529` fills.
GBP_USD: Best expression SHORT watch | I would enter only on retest failure.
AUD_USD: Best expression SHORT watch | I would enter only if path expands.
EUR_JPY: Best expression live SHORT | I hold id=`469502`.
GBP_JPY: Best expression LONG LIMIT | I leave id=`469521`.
AUD_JPY: Best expression WAIT | I would enter only from deeper rail.

## S Excavation Matrix
USD_JPY: Best expression LONG LIMIT id=`469528` | Why not S now: CPI catalyst keeps it tactical | Upgrade to S only if 159.67 fills and 159.73 breaks | Dead if 159.63 fails.
EUR_USD: Best expression SHORT LIMIT id=`469529` | Why not S now: pending retest receipt, not filled | Upgrade to S only if retest fails under 1.16945 | Dead if 1.16945 accepts.
GBP_USD: Best expression SHORT watch | Why not S now: low orderability | Upgrade to S only if bounce shelf fails cleanly | Dead if 1.34765 accepts.
AUD_USD: Best expression SHORT watch | Why not S now: no-edge pair | Upgrade to S only if AUD remains weakest | Dead if 0.71395 accepts.
EUR_JPY: Best expression SHORT live id=`469502` | Why not S now: live receipt is not expanding yet | Upgrade to S only if 186.63 fails again | Dead if 186.63 accepts.
GBP_JPY: Best expression LONG LIMIT id=`469521` | Why not S now: spread-heavy | Upgrade to S only if the shelf fills and holds | Dead if 214.99 shelf fails.
AUD_JPY: Best expression WAIT | Why not S now: right edge not mature | Upgrade to S only if deeper rail defends | Dead if 114.03 accepts.
Podium #1: EUR_JPY SHORT | Closest-to-S because live receipt already exists | Still blocked by: no lower expansion yet | If it upgrades: MARKET
Podium #2: EUR_USD SHORT | Closest-to-S because EUR offered and USD strongest | Still blocked by: pending retest fill id=`469529` | If it upgrades: LIMIT
Podium #3: USD_JPY LONG | Closest-to-S because USD bid is broad | Still blocked by: CPI timing and pullback fill | If it upgrades: LIMIT

## Gold Mine Inventory
Gold #1: USD_JPY LONG LIMIT [AUDIT] | armed LIMIT id=`469528` | exact contradiction if killed: 159.63 body close fails.
Gold #2: EUR_USD SHORT LIMIT [AUDIT+1] | armed LIMIT id=`469529` | exact contradiction if killed: 1.16945 body close reclaims.
Gold #3: GBP_JPY LONG LIMIT [PENDING] | armed LIMIT id=`469521` | exact contradiction if killed: 214.99 shelf fails before fill.
Gold #4: GBP_JPY SHORT STOP-ENTRY [AUDIT] | dead thesis because opposite armed long shelf id=`469521` plus spread leaves no clean role map.
Gold #5: EUR_JPY SHORT STOP-ENTRY [AUDIT+2] | dead thesis because existing live receipt id=`469502` is unpaid and an add would stack without risk reduction.

## A/S Excavation Mandate
Best A/S live now: EUR_JPY SHORT id=`469502`.
  Why this is A/S: live receipt exists.
  Order now: live trade id=`469502`.
Best A/S one print away: EUR_USD SHORT LIMIT id=`469529`.
  Missing print: pending retest fill.
  Arm now as: armed LIMIT id=`469529`.
Best A/S I am explicitly rejecting: GBP_JPY SHORT STOP-ENTRY.
  Exact contradiction: opposite GBP_JPY LONG LIMIT id=`469521` is live.

## S Hunt
Short-term S (5-30m):
  Pair / dir / type: EUR_JPY SHORT / live tactical short / hold
  Why this is S on this horizon: the receipt is live and protected.
  Promotion proof: blocker was missing fill -> cleared by live trade id=`469502`.
  MTF chain: H4 range | H1 trend-bear | M5 box | M1 upper shelf.
  Payout path: 186.60 -> 186.522.
  Orderability: live
  If not live: exact trigger 186.63 failure | invalidation 186.63 acceptance.
  Deployment result: live trade id=`469502`.
Medium-term S (30m-2h):
  Pair / dir / type: USD_JPY LONG / pullback limit / armed
  Why this is S on this horizon: USD bid is broad and the order waits for pullback.
  Promotion proof: blocker was overextended M1 price -> cleared by passive LIMIT id=`469528`.
  MTF chain: H4 range | H1 bullish lean | M5 squeeze high | M1 bid plateau.
  Payout path: pullback fill then 159.73.
  Orderability: LIMIT
  If not live: exact trigger 159.670 fill | invalidation 159.63 failure.
  Deployment result: armed LIMIT id=`469528`.
Long-term S (2h-1day):
  Pair / dir / type: EUR_USD SHORT / direct-USD retest / armed
  Why this is S on this horizon: EUR remains offered and USD is strongest.
  Promotion proof: blocker was stale receipt -> cleared by fresh LIMIT id=`469529`.
  MTF chain: H4 neutral-bear | H1 strong bear | M5 lower half | M1 shelf.
  Payout path: retest fill then 1.16825.
  Orderability: LIMIT
  If not live: exact trigger 1.16868 fill | invalidation 1.16945 acceptance.
  Deployment result: armed LIMIT id=`469529`.

## Multi-Vehicle Deployment
Lane 1 / PRIMARY: USD_JPY LONG LIMIT [audit] -> armed LIMIT id=`469528`.
Lane 2 / BACKUP: EUR_USD SHORT LIMIT [audit] -> armed LIMIT id=`469529`.
Lane 3 / THIRD CURRENCY: GBP_JPY LONG LIMIT [pending] -> armed LIMIT id=`469521`.
Lane 4 / FOURTH SEAT: GBP_JPY SHORT STOP-ENTRY [audit] -> dead thesis because opposite armed long shelf id=`469521` plus spread leaves no clean same-pair role map.
Lane 5 / FIFTH SEAT: EUR_JPY SHORT STOP-ENTRY [audit] -> dead thesis because the existing live cross receipt id=`469502` is unpaid and M1 is lifting, so an add would stack without risk reduction.

## Pending LIMITs
USD_JPY LONG LIMIT id=`469528` | entry=159.670 TP=159.730 SL=159.556 | freshness: leave while shelf holds.
EUR_USD SHORT LIMIT id=`469529` | entry=1.16868 TP=1.16825 SL=1.16928 | freshness: leave while shelf caps.
GBP_JPY LONG LIMIT id=`469521` | entry=214.990 TP=215.610 SL=214.711 | freshness: leave only while lower shelf survives.

## Capital Deployment
Lane 1 / PRIMARY: USD_JPY LONG LIMIT [audit] -> armed LIMIT id=`469528`.
Lane 2 / BACKUP: EUR_USD SHORT LIMIT [audit] -> armed LIMIT id=`469529`.
Lane 3 / THIRD CURRENCY: GBP_JPY LONG LIMIT [pending] -> armed LIMIT id=`469521`.
Lane 4 / FOURTH SEAT: GBP_JPY SHORT STOP-ENTRY [audit] -> dead thesis because opposite armed long shelf id=`469521` plus spread leaves no clean same-pair role map.
Lane 5 / FIFTH SEAT: EUR_JPY SHORT STOP-ENTRY [audit] -> dead thesis because the existing live cross receipt id=`469502` is unpaid and M1 is lifting, so an add would stack without risk reduction.
Execution count this session: 1 live receipts | 3 armed receipts
Horizon deployment:
  Short-term: EUR_JPY SHORT live trade id=`469502`.
  Medium-term: USD_JPY LONG armed LIMIT id=`469528`.
  Long-term: EUR_USD SHORT armed LIMIT id=`469529`.
LIVE NOW: EUR_JPY SHORT live trade id=`469502`.
RELOAD: EUR_USD SHORT armed LIMIT id=`469529`.
SECOND SHOT / OTHER SIDE: GBP_JPY LONG armed LIMIT id=`469521`.
Armed backup lane for this cadence: USD_JPY LONG LIMIT id=`469528`; EUR_USD SHORT LIMIT id=`469529`; GBP_JPY LONG LIMIT id=`469521`.
Flat-book status: not flat.
If broad tape but fewer than 2 live/armed lanes survived: not applicable.

## Action Tracking
USD_JPY LONG LIMIT placed -> ORDER_OK id=`469528`.
EUR_USD SHORT LIMIT placed -> ORDER_OK id=`469529`.
EUR_JPY SHORT id=`469502` held.

## Lessons (Recent)
Closed broker receipts must be reconciled before fresh risk.
"""


def _valid_state_text() -> str:
    return VALID_STATE_TEXT


def _validate_fixture_state(path: Path, monkeypatch, **kwargs) -> list[str]:
    monkeypatch.setattr("validate_trader_state._load_recent_action_board_snapshot", lambda: None)
    return validate_state(path, **kwargs)


def _with_slack_response(
    state_text: str,
    *,
    pending: str,
    handled: str,
    message_class: str,
    trade_consequence: str,
    reply_receipt: str,
) -> str:
    block = (
        "## Slack Response\n"
        f"Pending user ts: {pending}\n"
        f"Latest handled user ts: {handled}\n"
        f"Message class: {message_class}\n"
        f"Trade consequence: {trade_consequence}\n"
        f"Reply receipt: {reply_receipt}\n\n"
    )
    if "## Slack Response\n" in state_text:
        return re.sub(
            r"## Slack Response\n.*?\n(?=## Hot Updates\n)",
            block,
            state_text,
            count=1,
            flags=re.S,
        )
    return state_text.replace("## Market Narrative\n", block + "## Market Narrative\n", 1)


def test_tautological_dead_closure_requires_real_market_reason() -> None:
    assert (
        _tautological_dead_closure_phrase(
            "dead thesis because no live pending entry order exists"
        )
        == "no live pending entry order exists"
    )
    assert (
        _tautological_dead_closure_phrase(
            "dead thesis because no seat cleared promotion gate"
        )
        == "no seat cleared promotion gate"
    )
    assert (
        _tautological_dead_closure_phrase(
            "dead thesis because no seat cleared promotion gate: "
            "first floor-defense wick never printed and M1 kept stair-stepping lower"
        )
        is None
    )


def test_validate_state_rejects_execution_only_as_header_and_horizon_close(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_text = _valid_state_text()
    state_text = state_text.replace(
        "Best A/S one print away: EUR_USD SHORT LIMIT id=`469529`.",
        "Best A/S one print away: EUR_USD SHORT LIMIT id=`469529` "
        "dead thesis because no live pending entry order exists.",
        1,
    )
    state_text = state_text.replace(
        "Deployment result: armed LIMIT id=`469528`.",
        "Deployment result: dead thesis because no live pending entry order exists.",
        1,
    )
    path = tmp_path / "state.md"
    path.write_text(state_text)

    errors = _validate_fixture_state(
        path,
        monkeypatch,
        verify_live_oanda=False,
        verify_live_book_coverage=False,
        check_action_board=False,
        verify_live_entry_orderability=False,
    )
    joined = "\n".join(errors)

    assert "`Best A/S one print away` header closes the seat only because" in joined
    assert "`Medium-term S` deployment result closes only because" in joined


def test_validate_state_rejects_execution_only_hot_update(tmp_path: Path, monkeypatch) -> None:
    state_text = _valid_state_text()
    state_text = state_text.replace(
        "## Hot Updates\n",
        "## Hot Updates\n"
        "- 2026-04-23 19:46 UTC | test | no seat cleared promotion gate: no live pending entry order exists.\n",
        1,
    )
    path = tmp_path / "state.md"
    path.write_text(state_text)

    errors = _validate_fixture_state(
        path,
        monkeypatch,
        verify_live_oanda=False,
        verify_live_book_coverage=False,
        check_action_board=False,
        verify_live_entry_orderability=False,
    )
    joined = "\n".join(errors)

    assert "`Hot Updates` still explains a carry-forward seat only with missing execution" in joined


def test_validate_state_rejects_pending_lane_killed_only_because_trigger_not_printed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_text = _valid_state_text()
    state_text = state_text.replace(
        "Lane 5 / FIFTH SEAT: EUR_JPY SHORT STOP-ENTRY [audit] -> dead thesis because the existing live cross receipt id=`469502` is unpaid and M1 is lifting, so an add would stack without risk reduction.",
        "Lane 5 / FIFTH SEAT: EUR_JPY SHORT STOP-ENTRY [audit] -> dead thesis because trigger has not printed yet.",
        1,
    )
    path = tmp_path / "state.md"
    path.write_text(state_text)

    monkeypatch.setattr(
        "validate_trader_state._load_recent_action_board_snapshot",
        lambda: {
            "session_intent": {"mode": "FULL_TRADER"},
            "market_now": [],
            "multi_vehicle_lanes": [
                {
                    "pair": "EUR_JPY",
                    "direction": "SHORT",
                    "default_expression": "STOP-ENTRY",
                    "default_orderability": "STOP-ENTRY",
                    "source": "audit",
                }
            ],
        },
    )

    errors = validate_state(
        path,
        verify_live_oanda=False,
        verify_live_book_coverage=False,
        check_action_board=True,
        verify_live_entry_orderability=False,
    )
    joined = "\n".join(errors)

    assert "waiting conditions are the reason to arm the order, not thesis death" in joined
    assert "Latest session action board still had pending-style lane(s) EUR_JPY SHORT STOP-ENTRY [audit]" in joined


def test_validate_state_requires_gold_mine_inventory_for_full_trader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_text = _valid_state_text()
    state_text = re.sub(
        r"## Gold Mine Inventory\n.*?\n(?=## A/S Excavation Mandate\n)",
        "",
        state_text,
        count=1,
        flags=re.S,
    )
    path = tmp_path / "state.md"
    path.write_text(state_text)

    monkeypatch.setattr(
        "validate_trader_state._load_recent_action_board_snapshot",
        lambda: {
            "session_intent": {"mode": "FULL_TRADER"},
            "market_now": [],
            "multi_vehicle_lanes": [
                {
                    "pair": "USD_JPY",
                    "direction": "LONG",
                    "default_expression": "LIMIT",
                    "default_orderability": "LIMIT",
                    "source": "audit",
                }
            ],
        },
    )

    errors = validate_state(
        path,
        verify_live_oanda=False,
        verify_live_book_coverage=False,
        check_action_board=False,
        verify_live_entry_orderability=False,
    )

    assert any("Gold Mine Inventory" in error for error in errors)


def test_quality_audit_treats_bare_logged_trade_receipts_as_logged(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "logs" / "live_trade_log.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "[2026-04-23 16:43:00 UTC] EUR_JPY SHORT 2000u @ 186.623 | TP:186.522 SL:186.724 | id=469502\n"
        "[2026-04-23 20:06:00 UTC] STOP_FILL EUR_JPY SHORT 2000u @186.596 trade id=469539 TP=186.489 SL=186.729 tag=trader\n"
        "[2026-04-23 16:06:53 UTC] LIMIT EUR_JPY SHORT 2000u @186.622 id=469501 TP=186.522 SL=186.724\n"
    )
    monkeypatch.setattr("quality_audit.ROOT", tmp_path)

    assert "469502" in load_logged_trade_ids()
    assert "469539" in load_logged_trade_ids()
    assert "469501" not in load_logged_trade_ids()


def test_quality_audit_parses_prose_current_book_position() -> None:
    positions = parse_state_positions(
        "## Positions (Current)\n"
        "Open trades: 0 LONG / 1 SHORT / 1 pair.\n"
        "Current book: EUR_JPY SHORT trade id=`469502` is live on OANDA.\n"
        "Pending orders: EUR_USD SHORT LIMIT id=`469529`.\n"
    )

    assert positions == [
        {
            "pair": "EUR_JPY",
            "direction": "SHORT",
            "line": "Current book: EUR_JPY SHORT trade id=`469502` is live on OANDA.",
            "source": "positions_current",
        }
    ]


def test_validate_state_requires_slack_response_block_during_live_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_text = _valid_state_text()
    state_text = re.sub(
        r"## Slack Response\n.*?\n(?=## Hot Updates\n)",
        "",
        state_text,
        count=1,
        flags=re.S,
    )
    path = tmp_path / "state.md"
    path.write_text(state_text)

    monkeypatch.setattr("validate_trader_state._load_last_handled_slack_ts", lambda: None)
    monkeypatch.setattr("validate_trader_state._load_pending_slack_user_messages", lambda limit=20: [])

    errors = _validate_fixture_state(
        path,
        monkeypatch,
        verify_live_oanda=False,
        verify_live_book_coverage=False,
        check_action_board=False,
        verify_live_entry_orderability=False,
        verify_live_slack_replies=True,
    )
    joined = "\n".join(errors)

    assert "Missing `## Slack Response` block." in joined


def test_validate_state_rejects_unreplied_slack_user_message(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_text = _valid_state_text()
    state_text = _with_slack_response(
        state_text,
        pending="none",
        handled="1776926295.487409",
        message_class="question-observation",
        trade_consequence="no trade change",
        reply_receipt=(
            'python3 tools/slack_post.py "了解。現状は保持です。" '
            "--channel C0APAELAQDN --reply-to 1776926295.487409"
        ),
    )
    path = tmp_path / "state.md"
    path.write_text(state_text)

    monkeypatch.setattr(
        "validate_trader_state._load_last_handled_slack_ts",
        lambda: "1776926295.487409",
    )
    monkeypatch.setattr(
        "validate_trader_state._load_pending_slack_user_messages",
        lambda limit=20: [
            {
                "ts": "1776944936.909969",
                "text": "ユーロのストップ外した。TPまで待てばいいんじゃない？不都合ある？",
            }
        ],
    )

    errors = _validate_fixture_state(
        path,
        monkeypatch,
        verify_live_oanda=False,
        verify_live_book_coverage=False,
        check_action_board=False,
        verify_live_entry_orderability=False,
        verify_live_slack_replies=True,
    )
    joined = "\n".join(errors)

    assert "must name the latest pending user ts=`1776944936.909969`" in joined
    assert "Unread Slack user message still exists" in joined


def test_validate_state_accepts_closed_slack_response_when_queue_is_clear(
    tmp_path: Path,
    monkeypatch,
) -> None:
    handled_ts = "1776944936.909969"
    state_text = _valid_state_text()
    state_text = _with_slack_response(
        state_text,
        pending="none because no unread user message",
        handled=handled_ts,
        message_class="question-observation",
        trade_consequence="hold existing plan; no trade change",
        reply_receipt=(
            'python3 tools/slack_post.py "了解。TPまで待機で進めます。急反転時だけ再評価します。" '
            f"--channel C0APAELAQDN --reply-to {handled_ts}"
        ),
    )
    path = tmp_path / "state.md"
    path.write_text(state_text)

    monkeypatch.setattr("validate_trader_state._load_last_handled_slack_ts", lambda: handled_ts)
    monkeypatch.setattr("validate_trader_state._load_pending_slack_user_messages", lambda limit=20: [])

    errors = _validate_fixture_state(
        path,
        monkeypatch,
        verify_live_oanda=False,
        verify_live_book_coverage=False,
        check_action_board=False,
        verify_live_entry_orderability=False,
        verify_live_slack_replies=True,
    )
    slack_errors = [err for err in errors if "Slack" in err or "slack_post.py" in err]

    assert not slack_errors
