"""
QuantRabbit Trading Memory — Session Ingest
セッション終了後にdaily/の trades.md, notes.md, state.md を
QAチャンクに分割 → Ruri v3 で埋め込み → memory.db に保存
"""
import re
import sys
from datetime import date
from pathlib import Path

from schema import get_conn, init_db, serialize_f32, fetchone_val, fetchall_dict, DB_PATH
from parse_structured import parse_trades, parse_user_calls, parse_market_events

DAILY_DIR = Path(__file__).parent.parent / "daily"
STATE_MD = Path(__file__).parent.parent / "state.md"
STRATEGY_MD = Path(__file__).parent.parent / "strategy_memory.md"


# --- Embedding ---

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("cl-nagoya/ruri-v3-30m")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Ruri v3 は 'クエリ: ' プレフィックスで検索、ドキュメントはそのまま"""
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


# --- Chunking ---

def extract_pair(text: str) -> str | None:
    pairs = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
    for p in pairs:
        if p in text:
            return p
    return None


def chunk_trades_md(text: str, session_date: str) -> list[dict]:
    """trades.md をトレード単位のチャンクに分割"""
    chunks = []

    # ### で始まるセクションを分割
    sections = re.split(r'\n(?=###\s)', text)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue

        header_match = re.match(r'###\s+(.+)', section)
        if not header_match:
            # ### がないセクション（市況読み等）もキャプチャ
            if "市況読み" in section or "保有ポジション" in section:
                chunks.append({
                    "chunk_type": "market_read",
                    "question": f"{session_date}の市況読みは？",
                    "content": section,
                    "pair": None,
                    "tags": "market,context",
                    "source_file": "trades.md",
                })
            continue

        header = header_match.group(1)
        pair = extract_pair(section)

        # トレードチャンク
        if any(kw in header for kw in ["ENTRY", "CLOSE", "LONG", "SHORT", "利確", "損切"]):
            # P/L抽出
            pl_match = re.search(r'([+-][\d,]+円)', header)
            pl = pl_match.group(1) if pl_match else ""

            # 教訓・反省抽出
            lessons = re.findall(r'\*\*(.+?)\*\*', section)
            lesson_text = " / ".join(lessons) if lessons else ""

            question = f"{pair or ''}のトレード: {header.split('—')[0].strip()}"
            tags_list = ["trade"]
            if "損切" in section or "損切り" in section:
                tags_list.append("loss")
            if "利確" in section or "TP" in section:
                tags_list.append("profit")
            if "反省" in section or "教訓" in section or "ミス" in section:
                tags_list.append("lesson")

            chunks.append({
                "chunk_type": "trade",
                "question": question,
                "content": section,
                "pair": pair,
                "tags": ",".join(tags_list),
                "source_file": "trades.md",
            })

        # 教訓セクション
        elif "教訓" in header:
            chunks.append({
                "chunk_type": "lesson",
                "question": f"{session_date}のセッションから得た教訓は？",
                "content": section,
                "pair": None,
                "tags": "lesson,review",
                "source_file": "trades.md",
            })

    # 確定損益サマリー
    if "確定損益" in text or "確定益" in text:
        summary_match = re.search(
            r'(##\s*(?:確定損益サマリー|セッション\d*\s*確定益?)[\s\S]*?)(?=\n##\s|\n---|\Z)',
            text
        )
        if summary_match:
            chunks.append({
                "chunk_type": "summary",
                "question": f"{session_date}の損益結果は？",
                "content": summary_match.group(1).strip(),
                "pair": None,
                "tags": "pl,summary",
                "source_file": "trades.md",
            })

    return chunks


def chunk_notes_md(text: str, session_date: str) -> list[dict]:
    """notes.md をユーザー発言単位のチャンクに分割"""
    chunks = []
    sections = re.split(r'\n(?=###\s)', text)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 20:
            continue

        header_match = re.match(r'###\s+(.+)', section)
        if not header_match:
            continue

        header = header_match.group(1)
        pair = extract_pair(section)

        # ユーザーの発言 + チャート状態
        chunks.append({
            "chunk_type": "user_call",
            "question": f"ユーザーの相場読み: {header}",
            "content": section,
            "pair": pair,
            "tags": "user_call,market_read",
            "source_file": "notes.md",
        })

    return chunks


def chunk_state_md(text: str, session_date: str) -> list[dict]:
    """state.md からアクティブなテーゼをチャンク化"""
    chunks = []
    sections = re.split(r'\n(?=##\s)', text)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue

        pair = extract_pair(section)
        if "テーゼ" in section or "LONG" in section or "SHORT" in section:
            header_match = re.match(r'##\s+(.+)', section)
            header = header_match.group(1) if header_match else "テーゼ"
            chunks.append({
                "chunk_type": "thesis",
                "question": f"{pair or ''}のトレードテーゼ: {header}",
                "content": section,
                "pair": pair,
                "tags": "thesis,strategy",
                "source_file": "state.md",
            })

    return chunks


# --- Ingest ---

def ingest_date(session_date: str, force: bool = False):
    """指定日のデータを memory.db に取り込む"""
    day_dir = DAILY_DIR / session_date

    if not day_dir.exists():
        print(f"No data for {session_date}")
        return 0

    conn = get_conn()

    # 既存チェック（forceなら削除して再取り込み）
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

    # state.md（当日分のスナップショット）
    if STATE_MD.exists():
        text = STATE_MD.read_text()
        all_chunks.extend(chunk_state_md(text, session_date))

    if not all_chunks:
        print(f"{session_date}: no chunks extracted")
        return 0

    # 埋め込み生成（Q + content の結合テキスト）
    texts = []
    for c in all_chunks:
        t = c["question"] + "\n" + c["content"] if c["question"] else c["content"]
        texts.append(t)

    print(f"Embedding {len(texts)} chunks...")
    vectors = embed(texts)

    # DB挿入
    for chunk, vec in zip(all_chunks, vectors):
        conn.execute(
            """INSERT INTO chunks (session_date, chunk_type, question, content, pair, tags, source_file)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_date, chunk["chunk_type"], chunk["question"], chunk["content"],
             chunk["pair"], chunk["tags"], chunk["source_file"])
        )
        chunk_id = conn.last_insert_rowid()
        conn.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_f32(vec))
        )
    # --- 構造化データ取り込み ---
    trades_text = ""
    notes_text = ""
    trades_path = day_dir / "trades.md"
    notes_path = day_dir / "notes.md"
    if trades_path.exists():
        trades_text = trades_path.read_text()
    if notes_path.exists():
        notes_text = notes_path.read_text()

    # trades テーブル
    structured_trades = parse_trades(trades_text, session_date)
    for t in structured_trades:
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
    if structured_trades:
        print(f"  trades: {len(structured_trades)} records")

    # user_calls テーブル
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

    # market_events テーブル
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

    print(f"{session_date}: ingested {len(all_chunks)} chunks + {len(structured_trades)} trades + {len(user_calls)} calls + {len(events)} events")
    return len(all_chunks)


def ingest_all(force: bool = False):
    """daily/ 内の全日付を取り込む"""
    total = 0
    if not DAILY_DIR.exists():
        print("No daily directory found")
        return 0
    for day_dir in sorted(DAILY_DIR.iterdir()):
        if day_dir.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}', day_dir.name):
            total += ingest_date(day_dir.name, force=force)
    print(f"Total: {total} chunks ingested")
    return total


if __name__ == "__main__":
    init_db()

    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args and args[0] == "all":
        ingest_all(force=force)
    elif args:
        ingest_date(args[0], force=force)
    else:
        # デフォルト: 今日
        ingest_date(str(date.today()), force=force)
