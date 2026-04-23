"""
QuantRabbit Trading Memory — Session Ingest
After session end, split trades.md, notes.md, state.md from daily/ into
QA chunks → embed with Ruri v3 → save to memory.db

v2 (2026-03-25): Added feature to fetch today's trades directly from OANDA API.
trades.md parsing continues to be used as supplementary info (thesis, lessons).
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import json
import re
import sys
import urllib.request
import urllib.parse
from datetime import date, datetime, timezone
from pathlib import Path

from schema import get_conn, init_db, serialize_f32, fetchone_val, fetchall_dict, DB_PATH
from parse_structured import parse_trades, parse_user_calls, parse_market_events
from lesson_registry import refresh_registry, sync_strategy_memory_states

DAILY_DIR = Path(__file__).parent.parent / "daily"
STATE_MD = Path(__file__).parent.parent / "state.md"
STRATEGY_MD = Path(__file__).parent.parent / "strategy_memory.md"
INGEST_LOCK_PATH = DB_PATH.with_name(f"{DB_PATH.name}.ingest.lock")


# --- Embedding ---

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("cl-nagoya/ruri-v3-30m")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Ruri v3 uses 'query: ' prefix for search; documents are passed as-is"""
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


@contextmanager
def ingest_write_lock():
    """Prevent concurrent writers from duplicating rows during rebuilds."""
    INGEST_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INGEST_LOCK_PATH.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# --- Chunking ---

def extract_pair(text: str) -> str | None:
    pairs = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
    for p in pairs:
        if p in text:
            return p
    return None


def split_markdown_sections(text: str, header_levels: tuple[str, ...] = ("##", "###")) -> list[str]:
    escaped = "|".join(re.escape(level) for level in header_levels)
    pattern = rf"\n(?=(?:{escaped})\s)"
    return [section.strip() for section in re.split(pattern, text) if section.strip()]


def extract_header(section: str) -> tuple[str, str] | None:
    match = re.match(r"(#{2,3})\s+(.+)", section)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def has_trade_table(section: str) -> bool:
    return bool(re.search(r"^\|\s*(Pair|Direction|Side|Units|Entry|Close|P/L|Trade ID|Order ID|id)\s*\|", section, re.M | re.I))


def extract_direction(text: str) -> str | None:
    table_match = re.search(r"^\|\s*(?:Direction|Side)\s*\|\s*(LONG|SHORT)\s*\|$", text, re.M | re.I)
    if table_match:
        return table_match.group(1).upper()
    match = re.search(r"\b(LONG|SHORT)\b", text, re.I)
    if match:
        return match.group(1).upper()
    return None


def extract_trade_pair(header: str, section: str) -> str | None:
    header_pair = extract_pair(header)
    if header_pair:
        return header_pair
    table_match = re.search(r"^\|\s*Pair\s*\|\s*(\w+_\w+)\s*\|$", section, re.M | re.I)
    if table_match:
        return table_match.group(1)
    return extract_pair(section)


def extract_trade_stage(header: str, section: str) -> str | None:
    header_upper = header.upper()
    upper = f"{header}\n{section}".upper()
    if any(token in header_upper for token in ["ENTRY ORDER", "PENDING", "LIMIT PLACED", "STOP ENTRY", "STOP-ENTRY"]):
        return "pending"
    if header_upper.startswith("ENTRY ") or " FILLED" in header_upper:
        return "entry"
    if header_upper.startswith("CLOSE") or any(token in header_upper for token in ["HALF-TP", "HALF TP"]):
        return "exit"
    if "POSITIONS CARRIED" in upper:
        return "carry"
    if any(token in upper for token in ["CANCEL", "CANCELLED"]):
        return "cancel"
    if any(token in upper for token in ["CLOSE", "HALF-TP", "HALF TP", "TAKE_PROFIT", "STOP-LOSS", "STOP LOSS"]):
        return "exit"
    if any(token in upper for token in ["FILLED", "OPEN TIME", "TRADE ID"]):
        return "entry"
    if any(token in upper for token in ["PENDING", "ENTRY ORDER", "LIMIT PLACED", "STOP ENTRY", "STOP-ENTRY", "GTD"]):
        return "pending"
    if any(token in upper for token in ["ENTRY", "MARKET ORDER", "MARKET ENTRY"]):
        return "entry"
    if "POSITION" in upper:
        return "position"
    return None


def extract_strategy_update_date(text: str) -> str:
    match = re.search(r"最終更新:\s*(\d{4}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)
    return str(date.today())


def normalize_heading_text(text: str) -> str:
    cleaned = re.sub(r"（.*?）", "", text)
    cleaned = re.sub(r"\(.*?\)", "", cleaned)
    cleaned = cleaned.replace("##", "").replace("###", "")
    return " ".join(cleaned.strip().split())


def parse_state_marker(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    match = re.match(r"^-?\s*\[(CANDIDATE|WATCH|CONFIRMED|DEPRECATED)\]\s*(.*)$", stripped, re.I)
    if match:
        return match.group(1).lower(), match.group(2).strip()
    if stripped.startswith("- "):
        return None, stripped[2:].strip()
    if stripped.startswith("-"):
        return None, stripped[1:].strip()
    return None, stripped


def strip_state_marker(text: str) -> str:
    _, core = parse_state_marker(text)
    return core


def section_tag(label: str) -> str:
    normalized = normalize_heading_text(label).lower()
    replacements = {
        "⚡ read this first opportunity cost matters, but forced action kills expectancy too": "read_first",
        "confirmed patterns": "confirmed_pattern",
        "active observations": "active_observation",
        "deprecated": "deprecated",
        "per-pair learnings": "pair_learning",
        "pretrade feedback": "pretrade_feedback",
        "指標組み合わせの学び": "indicator_combo",
        "s-scan recipe scorecard": "s_scan_scorecard",
        "event day + thin market rules": "event_rule",
        "メンタル・行動": "mental",
    }
    return replacements.get(normalized, re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "strategy_memory")


def bullet_title(text: str) -> str:
    line = strip_state_marker(text)
    line = re.sub(r"^\[\d+/\d+\]\s*", "", line)
    for sep in (":", "—", "->"):
        if sep in line:
            return line.split(sep, 1)[0].strip()
    return line[:96].strip()


def make_strategy_question(section_name: str, subsection_name: str | None, pair: str | None, title: str) -> str:
    scope = pair or normalize_heading_text(subsection_name or section_name)
    title_text = title or normalize_heading_text(section_name)
    if normalize_heading_text(title_text) == normalize_heading_text(scope):
        title_text = normalize_heading_text(section_name)
    return f"strategy memory: {scope} / {title_text}"


def split_strategy_subsections(section_text: str) -> list[tuple[str | None, str]]:
    sections = split_markdown_sections(section_text, ("###",))
    if len(sections) == 1 and not sections[0].startswith("###"):
        return [(None, sections[0])]

    out = []
    for subsection in sections:
        header_info = extract_header(subsection)
        if header_info and header_info[0] == "###":
            out.append((header_info[1], subsection))
        else:
            out.append((None, subsection))
    return out


def collect_strategy_bullets(text: str) -> list[str]:
    bullets = []
    table_mode = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("|"):
            table_mode = True
            continue
        if table_mode and not stripped:
            table_mode = False
            continue
        if stripped.startswith("- "):
            bullets.append(stripped)
    return bullets


def chunk_strategy_memory_md(text: str) -> tuple[str, list[dict]]:
    """Split strategy_memory.md into section-level lesson chunks."""
    session_date = extract_strategy_update_date(text)
    chunks: list[dict] = []

    sections = split_markdown_sections(text, ("##",))
    for section in sections:
        header_info = extract_header(section)
        if not header_info or header_info[0] != "##":
            continue
        section_name = header_info[1]
        top_tag = section_tag(section_name)
        subsections = split_strategy_subsections(section)

        for subsection_name, subsection_text in subsections:
            bullets = collect_strategy_bullets(subsection_text)
            should_split_bullets = top_tag in {
                "confirmed_pattern",
                "active_observation",
                "pair_learning",
                "pretrade_feedback",
                "event_rule",
                "mental",
            }
            if should_split_bullets and bullets:
                for bullet in bullets:
                    explicit_state, core_text = parse_state_marker(bullet)
                    pair = extract_pair(core_text) or extract_pair(subsection_name or "") or extract_pair(section_name)
                    direction = extract_direction(core_text)
                    title = bullet_title(bullet)
                    tags = ["strategy_memory", top_tag, "lesson"]
                    if subsection_name:
                        tags.append(section_tag(subsection_name))
                    if explicit_state:
                        tags.append(f"state_{explicit_state}")
                    if pair:
                        tags.append(pair.lower())
                    if direction:
                        tags.append(direction.lower())
                    chunks.append({
                        "chunk_type": "lesson",
                        "question": make_strategy_question(section_name, subsection_name, pair, title),
                        "content": f"- {core_text}",
                        "pair": pair,
                        "direction": direction,
                        "tags": ",".join(dict.fromkeys(tags)),
                        "source_file": "strategy_memory.md",
                    })
                continue

            body = subsection_text.strip()
            body_lines = [line for line in body.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            if not body_lines:
                continue
            pair = extract_pair(body) or extract_pair(subsection_name or "") or extract_pair(section_name)
            direction = extract_direction(body)
            title = normalize_heading_text(subsection_name or section_name)
            tags = ["strategy_memory", top_tag, "lesson"]
            if subsection_name:
                tags.append(section_tag(subsection_name))
            if pair:
                tags.append(pair.lower())
            if direction:
                tags.append(direction.lower())
            chunks.append({
                "chunk_type": "lesson",
                "question": make_strategy_question(section_name, subsection_name, pair, title),
                "content": body,
                "pair": pair,
                "direction": direction,
                "tags": ",".join(dict.fromkeys(tags)),
                "source_file": "strategy_memory.md",
            })

    return session_date, chunks


def build_trade_tags(section: str, stage: str | None) -> list[str]:
    upper_section = section.upper()
    tags_list = ["trade"]
    if stage:
        tags_list.append(stage)
    if "損切" in section or "損切り" in section or "LOSS" in upper_section:
        tags_list.append("loss")
    if "利確" in section or "TP" in upper_section or "PROFIT" in upper_section or "P/L | +" in upper_section:
        tags_list.append("profit")
    if "反省" in section or "教訓" in section or "ミス" in section or "LESSON" in upper_section:
        tags_list.append("lesson")
    if "LIMIT" in upper_section or "PENDING" in upper_section:
        tags_list.append("pending")
    if "CANCEL" in upper_section:
        tags_list.append("cancel")
    return list(dict.fromkeys(tags_list))


def format_thesis_question(pair: str | None, header: str) -> str:
    prefix = f"{pair} " if pair else ""
    return f"{prefix}trade thesis: {header}"


def format_trade_question(pair: str, direction: str, stage: str | None, header: str, trade: dict) -> str:
    header_label = header.split("—")[0].strip()
    header_pair = extract_pair(header_label)
    if header_pair and header_pair != pair:
        trade_id = trade.get("trade_id")
        if trade_id:
            header_label = f"{pair} #{trade_id}"
        else:
            header_label = f"{pair} {direction}"
    return f"{pair} {direction} {stage or 'trade'}: {header_label}"


def resolve_state_snapshot(session_date: str, day_dir: Path) -> Path | None:
    historical = day_dir / "state.md"
    if historical.exists():
        return historical
    if session_date == str(date.today()) and STATE_MD.exists():
        return STATE_MD
    return None


def chunk_trades_md(text: str, session_date: str) -> list[dict]:
    """Split trades.md into per-trade chunks"""
    chunks = []

    sections = split_markdown_sections(text, ("##", "###"))

    for section in sections:
        if not section or len(section) < 30:
            continue

        header_info = extract_header(section)
        if not header_info:
            # Also capture sections without ### (market reads etc.)
            if "市況読み" in section or "保有ポジション" in section:
                chunks.append({
                    "chunk_type": "market_read",
                    "question": f"What was the market read for {session_date}?",
                    "content": section,
                    "pair": None,
                    "direction": None,
                    "tags": "market,context",
                    "source_file": "trades.md",
                })
            continue

        _, header = header_info
        parsed_trades = parse_trades(section, session_date)
        if parsed_trades:
            for trade in parsed_trades:
                pair = trade.get("pair") or extract_trade_pair(header, section)
                direction = trade.get("direction") or extract_direction(section)
                if not pair or not direction:
                    continue
                stage = extract_trade_stage(header, section)
                question = format_trade_question(pair, direction, stage, header, trade)
                chunks.append({
                    "chunk_type": "trade",
                    "question": question,
                    "content": section,
                    "pair": pair,
                    "direction": direction,
                    "tags": ",".join(build_trade_tags(section, stage)),
                    "source_file": "trades.md",
                })

        # Lessons section
        elif "教訓" in header:
            chunks.append({
                "chunk_type": "lesson",
                "question": f"What lessons were learned from the {session_date} session?",
                "content": section,
                "pair": None,
                "direction": None,
                "tags": "lesson,review",
                "source_file": "trades.md",
            })

    # Realized P&L summary
    if "確定損益" in text or "確定益" in text:
        summary_match = re.search(
            r'(##\s*(?:確定損益サマリー|セッション\d*\s*確定益?)[\s\S]*?)(?=\n##\s|\n---|\Z)',
            text
        )
        if summary_match:
            chunks.append({
                "chunk_type": "summary",
                "question": f"What were the P&L results for {session_date}?",
                "content": summary_match.group(1).strip(),
                "pair": None,
                "direction": None,
                "tags": "pl,summary",
                "source_file": "trades.md",
            })

    return chunks


def chunk_notes_md(text: str, session_date: str) -> list[dict]:
    """Split notes.md into per-user-statement chunks"""
    chunks = []
    sections = split_markdown_sections(text, ("##", "###"))

    for section in sections:
        if not section or len(section) < 20:
            continue

        header_info = extract_header(section)
        if not header_info:
            continue

        _, header = header_info
        pair = extract_pair(section)

        # User statement + chart state
        chunks.append({
            "chunk_type": "user_call",
            "question": f"User's market read: {header}",
            "content": section,
            "pair": pair,
            "direction": None,
            "tags": "user_call,market_read",
            "source_file": "notes.md",
        })

    return chunks


def refresh_strategy_memory_chunks(conn) -> int:
    if not STRATEGY_MD.exists():
        return 0

    marker_updates = sync_strategy_memory_states()
    if marker_updates:
        print(f"Synced {marker_updates} strategy-memory lesson state markers...")

    existing_ids = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM chunks WHERE source_file = ?",
            ("strategy_memory.md",),
        )
    ]
    for chunk_id in existing_ids:
        conn.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE source_file = ?", ("strategy_memory.md",))

    text = STRATEGY_MD.read_text()
    session_date, strategy_chunks = chunk_strategy_memory_md(text)
    if not strategy_chunks:
        return 0

    texts = [
        f"{chunk['question']}\n{chunk['content']}" if chunk.get("question") else chunk["content"]
        for chunk in strategy_chunks
    ]
    print(f"Embedding {len(texts)} strategy memory chunks...")
    vectors = embed(texts)

    for chunk, vec in zip(strategy_chunks, vectors):
        conn.execute(
            """INSERT INTO chunks (session_date, chunk_type, question, content, pair, direction, tags, source_file)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_date,
                chunk["chunk_type"],
                chunk["question"],
                chunk["content"],
                chunk["pair"],
                chunk.get("direction"),
                chunk["tags"],
                chunk["source_file"],
            ),
        )
        chunk_id = conn.last_insert_rowid()
        conn.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_f32(vec))
        )
    refresh_registry()
    return len(strategy_chunks)


def chunk_state_md(text: str, session_date: str) -> list[dict]:
    """Chunk active theses from state.md"""
    chunks = []
    sections = split_markdown_sections(text, ("##", "###"))

    for section in sections:
        if not section or len(section) < 30:
            continue

        header_info = extract_header(section)
        if not header_info:
            continue
        _, header = header_info
        pair = extract_pair(header)
        direction = extract_direction(header if pair else "")
        if "テーゼ" in section or "LONG" in section or "SHORT" in section:
            chunks.append({
                "chunk_type": "thesis",
                "question": format_thesis_question(pair, header),
                "content": section,
                "pair": pair,
                "direction": direction,
                "tags": "thesis,strategy",
                "source_file": "state.md",
            })

        if normalize_heading_text(header).lower() == "hot updates":
            for raw_line in section.splitlines():
                stripped = raw_line.strip()
                if not stripped.startswith("- "):
                    continue
                core = stripped[2:].strip()
                hot_pair = extract_pair(core)
                hot_direction = extract_direction(core)
                hot_title = core
                if " | " in core:
                    parts = [part.strip() for part in core.split(" | ") if part.strip()]
                    if len(parts) >= 3:
                        hot_title = f"{parts[1]} | {parts[2]}"
                    elif len(parts) >= 2:
                        hot_title = f"{parts[0]} | {parts[1]}"
                tags = ["state", "hot_update", "lesson"]
                if hot_pair:
                    tags.append(hot_pair.lower())
                if hot_direction:
                    tags.append(hot_direction.lower())
                chunks.append({
                    "chunk_type": "lesson",
                    "question": f"hot update: {hot_title}",
                    "content": stripped,
                    "pair": hot_pair,
                    "direction": hot_direction,
                    "tags": ",".join(dict.fromkeys(tags)),
                    "source_file": "state.md",
                })

    return chunks


# --- OANDA API Trade Fetch ---

ENV_TOML = Path(__file__).parent.parent.parent / "config" / "env.toml"

def _load_oanda_config():
    text = ENV_TOML.read_text()
    cfg = {}
    for line in text.strip().split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg

OANDA_PAIRS = {"USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"}


def _request_json(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _normalize_trade_snapshot(trade: dict) -> dict | None:
    if not trade:
        return None

    trade_id = str(trade.get("id", "")).strip()
    pair = trade.get("instrument")
    try:
        initial_units = float(trade.get("initialUnits", 0) or 0)
    except Exception:
        initial_units = 0.0
    direction = "LONG" if initial_units > 0 else "SHORT" if initial_units < 0 else None
    try:
        entry_price = float(trade.get("price")) if trade.get("price") is not None else None
    except Exception:
        entry_price = None
    try:
        exit_price = float(trade.get("averageClosePrice")) if trade.get("averageClosePrice") is not None else None
    except Exception:
        exit_price = None

    return {
        "trade_id": trade_id or None,
        "pair": pair,
        "direction": direction,
        "units": abs(int(initial_units)) if initial_units else None,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_time": trade.get("openTime"),
        "close_time": trade.get("closeTime"),
    }


def _fetch_trade_snapshots(base: str, acct: str, headers: dict, trade_ids: list[str]) -> dict[str, dict]:
    snapshots: dict[str, dict] = {}
    clean_ids = [str(trade_id).strip() for trade_id in trade_ids if str(trade_id).strip()]
    if not clean_ids:
        return snapshots

    for i in range(0, len(clean_ids), 50):
        batch = clean_ids[i:i + 50]
        params = urllib.parse.urlencode({"state": "CLOSED", "ids": ",".join(batch)})
        try:
            data = _request_json(f"{base}/v3/accounts/{acct}/trades?{params}", headers)
        except Exception:
            data = {}
        for trade in data.get("trades", []):
            snapshot = _normalize_trade_snapshot(trade)
            if snapshot and snapshot["trade_id"]:
                snapshots[snapshot["trade_id"]] = snapshot

        missing = [trade_id for trade_id in batch if trade_id not in snapshots]
        for trade_id in missing:
            try:
                data = _request_json(f"{base}/v3/accounts/{acct}/trades/{trade_id}", headers)
            except Exception:
                continue
            snapshot = _normalize_trade_snapshot(data.get("trade"))
            if snapshot and snapshot["trade_id"]:
                snapshots[snapshot["trade_id"]] = snapshot
    return snapshots


def fetch_oanda_trades(session_date: str) -> list[dict]:
    """Fetch closed trades for the specified date from OANDA API and return in trades table format"""
    try:
        cfg = _load_oanda_config()
    except Exception as e:
        print(f"  OANDA config error: {e}")
        return []

    token = cfg.get('oanda_token', '')
    acct = cfg.get('oanda_account_id', '')
    base = 'https://api-fxtrade.oanda.com'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }

    since = f"{session_date}T00:00:00.000000000Z"
    # Until 00:00 of the next day
    dt = datetime.strptime(session_date, '%Y-%m-%d')
    next_day = dt.replace(hour=0, minute=0, second=0) + __import__('datetime').timedelta(days=1)
    to = next_day.strftime('%Y-%m-%dT00:00:00.000000000Z')

    try:
        params = urllib.parse.urlencode({
            'from': since,
            'to': to,
            'type': 'ORDER_FILL',
            'pageSize': 1000,
        })
        url = f"{base}/v3/accounts/{acct}/transactions?{params}"
        id_range = _request_json(url, headers)

        all_txns = []
        for page_url in id_range.get('pages', []):
            data = _request_json(page_url, headers)
            all_txns.extend(data.get('transactions', []))
    except Exception as e:
        print(f"  OANDA API error: {e}")
        return []

    entries = {}
    closes = {}

    for txn in all_txns:
        if txn.get('type') != 'ORDER_FILL':
            continue
        instrument = txn.get('instrument', '')
        if instrument not in OANDA_PAIRS:
            continue

        # New entry
        trade_opened = txn.get('tradeOpened')
        if trade_opened:
            tid = trade_opened.get('tradeID', '')
            units = int(float(trade_opened.get('units', 0)))
            entries[tid] = {
                'price': float(txn.get('price', 0)),
                'units': abs(units),
                'direction': 'LONG' if units > 0 else 'SHORT',
                'pair': instrument,
                'time': txn.get('time', ''),
                'reason': txn.get('reason', ''),
            }

        for tc in txn.get('tradesClosed', []) + txn.get('tradesReduced', []):
            tid = tc.get('tradeID', '')
            pl = float(tc.get('realizedPL', 0))
            units = abs(int(float(tc.get('units', 0))))
            if tid not in closes:
                closes[tid] = {
                    'close_price': float(txn.get('price', 0)),
                    'pl': pl,
                    'units': units,
                    'instrument': instrument,
                    'close_time': txn.get('time', ''),
                }
            else:
                closes[tid]['pl'] += pl
                closes[tid]['units'] += units

    snapshots = _fetch_trade_snapshots(base, acct, headers, list(closes.keys()))

    trades = []
    for tid in sorted(closes.keys()):
        close = closes.get(tid)
        if not close:
            continue
        entry = entries.get(tid)
        snapshot = snapshots.get(tid, {})

        pair = 'UNKNOWN'
        if snapshot.get('pair'):
            pair = snapshot['pair']
        elif entry:
            pair = entry['pair']
        elif close.get('instrument'):
            pair = close['instrument']

        direction = snapshot.get('direction') or (entry['direction'] if entry else None)
        if not direction:
            continue

        units = snapshot.get('units')
        if units is None:
            units = entry['units'] if entry else close['units']

        entry_price = snapshot.get('entry_price')
        if entry_price is None and entry:
            entry_price = entry['price']

        ts = snapshot.get('entry_time') or (entry['time'] if entry else '')
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00').split('.')[0] + '+00:00') if ts else None
        except:
            dt = None

        trades.append({
            'session_date': session_date,
            'trade_id': tid,
            'pair': pair,
            'direction': direction,
            'units': units,
            'entry_price': entry_price,
            'exit_price': close['close_price'],
            'pl': close['pl'],
            'session_hour': dt.hour if dt else None,
            'reason': entry.get('reason', '') if entry else '',
        })

    return trades


def _enrich_from_log(conn, session_date: str, log_path: Path) -> int:
    """Supplement trades in DB with thesis/reason from live_trade_log.txt"""
    enriched = 0
    try:
        log_text = log_path.read_text()
    except Exception:
        return 0

    for line in log_text.split('\n'):
        if session_date not in line:
            continue
        # Extract trade_id (#NNNNN)
        tid_match = re.search(r'#(\d{5,})', line)
        if not tid_match:
            continue
        tid = tid_match.group(1)

        # Extract reason= or thesis=
        reason_match = re.search(r'(?:reason|thesis)[=:]\s*(.+?)(?:\||$)', line)
        if not reason_match:
            continue
        reason_text = reason_match.group(1).strip()

        # UPDATE only if reason is not already in DB
        existing_reason = fetchone_val(conn,
            "SELECT reason FROM trades WHERE trade_id = ?", (tid,))
        if existing_reason and existing_reason not in ('', 'MARKET_ORDER'):
            continue

        conn.execute(
            "UPDATE trades SET reason = ? WHERE trade_id = ? AND (reason IS NULL OR reason = '' OR reason = 'MARKET_ORDER')",
            (reason_text, tid)
        )
        enriched += 1

    return enriched


# --- Ingest ---

def _ingest_date_unlocked(session_date: str, force: bool = False, refresh_strategy_memory: bool = True):
    """Ingest data for the specified date into memory.db"""
    day_dir = DAILY_DIR / session_date

    if not day_dir.exists():
        print(f"No data for {session_date}")
        return 0

    conn = get_conn()

    # Check existing (if force, delete and re-ingest)
    existing = fetchone_val(conn, "SELECT COUNT(*) FROM chunks WHERE session_date = ?", (session_date,))

    if existing > 0 and not force:
        print(f"{session_date}: already ingested ({existing} chunks). Use --force to re-ingest.")
        return 0

    if existing > 0 and force:
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM chunks WHERE session_date = ?", (session_date,)
        )]
        for cid in ids:
            conn.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM chunks WHERE session_date = ?", (session_date,))
        conn.execute("DELETE FROM trades WHERE session_date = ?", (session_date,))
        conn.execute("DELETE FROM user_calls WHERE session_date = ?", (session_date,))
        conn.execute("DELETE FROM market_events WHERE session_date = ?", (session_date,))
        print(f"{session_date}: cleared existing data")

    all_chunks = []

    # trades.md
    trades_path = day_dir / "trades.md"
    if trades_path.exists():
        text = trades_path.read_text()
        all_chunks.extend(chunk_trades_md(text, session_date))

    # notes.md
    notes_path = day_dir / "notes.md"
    if notes_path.exists():
        text = notes_path.read_text()
        all_chunks.extend(chunk_notes_md(text, session_date))

    # state.md: use a day-specific snapshot when present; otherwise only use the live state for today
    state_snapshot = resolve_state_snapshot(session_date, day_dir)
    if state_snapshot and state_snapshot.exists():
        text = state_snapshot.read_text()
        all_chunks.extend(chunk_state_md(text, session_date))

    if all_chunks:
        texts = []
        for c in all_chunks:
            t = c["question"] + "\n" + c["content"] if c["question"] else c["content"]
            texts.append(t)

        print(f"Embedding {len(texts)} chunks...")
        vectors = embed(texts)

        for chunk, vec in zip(all_chunks, vectors):
            conn.execute(
                """INSERT INTO chunks (session_date, chunk_type, question, content, pair, direction, tags, source_file)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_date, chunk["chunk_type"], chunk["question"], chunk["content"],
                 chunk["pair"], chunk.get("direction"), chunk["tags"], chunk["source_file"])
            )
            chunk_id = conn.last_insert_rowid()
            conn.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, serialize_f32(vec))
            )
    else:
        print(f"{session_date}: no markdown chunks extracted")
    # --- Structured data ingestion ---
    trades_text = ""
    notes_text = ""
    trades_path = day_dir / "trades.md"
    notes_path = day_dir / "notes.md"
    if trades_path.exists():
        trades_text = trades_path.read_text()
    if notes_path.exists():
        notes_text = notes_path.read_text()

    # trades table — OANDA + trades.md integration (don't lose qualitative data)
    # 1) Fetch today's closed trades from OANDA API
    oanda_trades = fetch_oanda_trades(session_date)
    oanda_inserted = 0
    for t in oanda_trades:
        exists = fetchone_val(conn,
            "SELECT COUNT(*) FROM trades WHERE trade_id = ?", (t['trade_id'],))
        if exists and exists > 0:
            continue
        conn.execute(
            """INSERT INTO trades (session_date, trade_id, pair, direction, units,
               entry_price, exit_price, pl, session_hour, reason)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (t['session_date'], t['trade_id'], t['pair'], t['direction'], t['units'],
             t['entry_price'], t['exit_price'], t['pl'], t['session_hour'], t['reason'])
        )
        oanda_inserted += 1
    if oanda_inserted:
        print(f"  oanda_trades: {oanda_inserted} inserted")

    # 2) Structured parse from trades.md → UPDATE OANDA records (supplement qualitative data)
    structured_trades = parse_trades(trades_text, session_date)
    enriched = 0
    md_inserted = 0
    for t in structured_trades:
        if t.get('trade_id'):
            # If already inserted by OANDA, UPDATE with qualitative data (integrate, don't skip)
            exists = fetchone_val(conn,
                "SELECT COUNT(*) FROM trades WHERE trade_id = ?", (t['trade_id'],))
            if exists and exists > 0:
                # UPDATE only fields that have qualitative data
                updates = []
                params = []
                for col in ['h1_adx', 'h1_trend', 'm5_adx', 'm5_trend', 'rsi', 'stoch_rsi',
                            'regime', 'vix', 'dxy', 'active_headlines', 'event_risk',
                            'entry_type', 'had_sl', 'lesson']:
                    val = t.get(col)
                    if val is not None:
                        updates.append(f"{col} = ?")
                        params.append(val)
                # reason: thesis from trades.md is more useful than OANDA's 'MARKET_ORDER'
                if t.get('reason') and t['reason'] != 'MARKET_ORDER':
                    updates.append("reason = ?")
                    params.append(t['reason'])
                if updates:
                    params.append(t['trade_id'])
                    conn.execute(
                        f"UPDATE trades SET {', '.join(updates)} WHERE trade_id = ?",
                        tuple(params)
                    )
                    enriched += 1
                continue
        # No trade_id, or not in DB or OANDA → new INSERT
        conn.execute(
            """INSERT INTO trades (session_date, trade_id, pair, direction, units,
               entry_price, exit_price, pl,
               h1_adx, h1_trend, m5_adx, m5_trend, rsi, stoch_rsi,
               regime, vix, dxy, active_headlines, event_risk, session_hour,
               entry_type, had_sl, reason, lesson, user_call_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (t['session_date'], t['trade_id'], t['pair'], t['direction'], t['units'],
             t['entry_price'], t['exit_price'], t['pl'],
             t['h1_adx'], t['h1_trend'], t['m5_adx'], t['m5_trend'], t['rsi'], t['stoch_rsi'],
             t['regime'], t['vix'], t['dxy'], t['active_headlines'], t['event_risk'], t['session_hour'],
             t['entry_type'], t['had_sl'], t['reason'], t['lesson'], t['user_call_id'])
        )
        md_inserted += 1
    if enriched:
        print(f"  enriched: {enriched} OANDA records with trades.md qualitative data")
    if md_inserted:
        print(f"  trades_md: {md_inserted} additional records")

    # 3) Also supplement thesis/reason from live_trade_log.txt
    log_path = Path(__file__).parent.parent.parent / "logs" / "live_trade_log.txt"
    if log_path.exists():
        log_enriched = _enrich_from_log(conn, session_date, log_path)
        if log_enriched:
            print(f"  log_enriched: {log_enriched} records from live_trade_log.txt")

    # user_calls table
    user_calls = parse_user_calls(notes_text, session_date)
    for uc in user_calls:
        conn.execute(
            """INSERT INTO user_calls (session_date, timestamp, pair, direction, call_text,
               conditions, price_at_call, outcome, pl_after_30m, pl_after_1h,
               price_after_30m, price_after_1h, confidence, acted_on)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (uc['session_date'], uc['timestamp'], uc['pair'], uc['direction'], uc['call_text'],
             uc['conditions'], uc['price_at_call'], uc['outcome'], uc['pl_after_30m'], uc['pl_after_1h'],
             uc['price_after_30m'], uc['price_after_1h'], uc['confidence'], uc['acted_on'])
        )
    if user_calls:
        print(f"  user_calls: {len(user_calls)} records")

    # market_events table
    all_text = trades_text + "\n" + notes_text
    events = parse_market_events(all_text, session_date)
    for ev in events:
        conn.execute(
            """INSERT INTO market_events (session_date, timestamp, event_type, headline,
               pairs_affected, spike_pips, spike_direction, duration_min,
               pre_vix, post_vix, impact)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (ev['session_date'], ev['timestamp'], ev['event_type'], ev['headline'],
             ev['pairs_affected'], ev['spike_pips'], ev['spike_direction'], ev['duration_min'],
             ev['pre_vix'], ev['post_vix'], ev['impact'])
        )
    if events:
        print(f"  market_events: {len(events)} records")

    strategy_chunks = 0
    if refresh_strategy_memory:
        strategy_chunks = refresh_strategy_memory_chunks(conn)
        if strategy_chunks:
            print(f"  strategy_memory: {strategy_chunks} chunks refreshed")

    print(
        f"{session_date}: ingested {len(all_chunks)} daily chunks + {strategy_chunks} strategy chunks "
        f"+ {len(structured_trades)} trades + {len(user_calls)} calls + {len(events)} events"
    )
    return len(all_chunks)


def ingest_date(session_date: str, force: bool = False, refresh_strategy_memory: bool = True, with_lock: bool = True):
    if not with_lock:
        return _ingest_date_unlocked(session_date, force=force, refresh_strategy_memory=refresh_strategy_memory)
    with ingest_write_lock():
        return _ingest_date_unlocked(session_date, force=force, refresh_strategy_memory=refresh_strategy_memory)


def _ingest_all_unlocked(force: bool = False):
    """Ingest all dates inside daily/"""
    total = 0
    if not DAILY_DIR.exists():
        print("No daily directory found")
        return 0
    if force:
        conn = get_conn()
        conn.execute("DELETE FROM chunks_vec")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM user_calls")
        conn.execute("DELETE FROM market_events")
        print("Cleared existing memory tables before full re-ingest")
    conn = get_conn()
    for day_dir in sorted(DAILY_DIR.iterdir()):
        if day_dir.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}', day_dir.name):
            total += _ingest_date_unlocked(day_dir.name, force=force, refresh_strategy_memory=False)
    strategy_chunks = refresh_strategy_memory_chunks(conn)
    if strategy_chunks:
        print(f"Strategy memory refreshed: {strategy_chunks} chunks")
    print(f"Total: {total} chunks ingested")
    return total


def ingest_all(force: bool = False, with_lock: bool = True):
    if not with_lock:
        return _ingest_all_unlocked(force=force)
    with ingest_write_lock():
        return _ingest_all_unlocked(force=force)


if __name__ == "__main__":
    init_db()

    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args and args[0] == "all":
        ingest_all(force=force)
    elif args:
        ingest_date(args[0], force=force)
    else:
        # Default: today
        ingest_date(str(date.today()), force=force)
