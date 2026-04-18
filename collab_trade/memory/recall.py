"""
QuantRabbit Trading Memory — Recall (Search & Retrieve)
Retrieve past memories via hybrid vector search + keyword search
"""
from __future__ import annotations

import re
import sys
from schema import get_conn, serialize_f32, fetchall_dict, fetchone_val

_model = None
_chunks_columns = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("cl-nagoya/ruri-v3-30m")
    return _model


def embed_query(text: str) -> list[float]:
    """With Ruri v3 query prefix"""
    model = get_model()
    vec = model.encode([f"query: {text}"], normalize_embeddings=True)
    return vec[0].tolist()


def _get_chunks_columns(conn) -> set[str]:
    global _chunks_columns
    if _chunks_columns is None:
        _chunks_columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
    return _chunks_columns


def _apply_direction_filter(conn, where_clauses: list[str], params: list, direction: str | None):
    if not direction:
        return
    if "direction" in _get_chunks_columns(conn):
        where_clauses.append("((direction = ?) OR (direction IS NULL AND chunk_type != 'trade'))")
        params.append(direction.upper())
        return
    upper = direction.upper()
    where_clauses.append("(UPPER(content) LIKE ? OR UPPER(question) LIKE ?)")
    params.extend([f"%{upper}%", f"%{upper}%"])


def vector_search(query: str, top_k: int = 5, pair: str | None = None,
                   chunk_type: str | None = None, direction: str | None = None) -> list[dict]:
    """Vector similarity search"""
    conn = get_conn()
    qvec = embed_query(query)

    rows = list(conn.execute(
        """
        SELECT chunk_id, distance
        FROM chunks_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (serialize_f32(qvec), top_k * 3)
    ))

    if not rows:
        return []

    ids = [r[0] for r in rows]
    dist_map = {r[0]: r[1] for r in rows}

    placeholders = ",".join("?" * len(ids))
    where_clauses = [f"id IN ({placeholders})"]
    params = list(ids)

    if pair:
        where_clauses.append("pair = ?")
        params.append(pair)
    if chunk_type:
        where_clauses.append("chunk_type = ?")
        params.append(chunk_type)
    _apply_direction_filter(conn, where_clauses, params, direction)

    sql = f"SELECT * FROM chunks WHERE {' AND '.join(where_clauses)}"
    results = fetchall_dict(conn, sql, params)

    results = sorted(results, key=lambda r: dist_map.get(r["id"], 999))
    return [r | {"distance": dist_map.get(r["id"], 999)} for r in results[:top_k]]


def keyword_search(keyword: str, top_k: int = 10, pair: str | None = None,
                    chunk_type: str | None = None, direction: str | None = None) -> list[dict]:
    """Keyword full-text search (LIKE)"""
    conn = get_conn()

    terms = _tokenize_query(keyword)
    if not terms:
        terms = [keyword]

    score_parts = []
    score_params = []
    match_parts = []
    match_params = []
    for term in terms:
        like = f"%{term}%"
        score_parts.append("(CASE WHEN content LIKE ? OR question LIKE ? THEN 1 ELSE 0 END)")
        score_params.extend([like, like])
        match_parts.append("(content LIKE ? OR question LIKE ?)")
        match_params.extend([like, like])

    where_clauses = [f"({' OR '.join(match_parts)})"]
    params = score_params + match_params

    if pair:
        where_clauses.append("pair = ?")
        params.append(pair)
    if chunk_type:
        where_clauses.append("chunk_type = ?")
        params.append(chunk_type)
    _apply_direction_filter(conn, where_clauses, params, direction)

    score_sql = " + ".join(score_parts) if score_parts else "0"
    sql = f"""
        SELECT *, ({score_sql}) AS match_score
        FROM chunks
        WHERE {' AND '.join(where_clauses)}
        ORDER BY match_score DESC, session_date DESC, id DESC
        LIMIT ?
    """
    params.append(top_k)
    return fetchall_dict(conn, sql, params)


def hybrid_search(query: str, top_k: int = 5, pair: str | None = None,
                   chunk_type: str | None = None, direction: str | None = None) -> list[dict]:
    """Hybrid vector + keyword search (deduplicated)"""
    vec_results = vector_search(query, top_k=top_k, pair=pair, chunk_type=chunk_type, direction=direction)
    kw_results = keyword_search(query, top_k=top_k, pair=pair, chunk_type=chunk_type, direction=direction)

    seen_ids = set()
    merged = []

    for r in vec_results:
        if r["id"] not in seen_ids:
            r["match_type"] = "vector"
            merged.append(r)
            seen_ids.add(r["id"])

    for r in kw_results:
        if r["id"] not in seen_ids:
            r["match_type"] = "keyword"
            merged.append(r)
            seen_ids.add(r["id"])

    return merged[:top_k]


def format_results(results: list[dict]) -> str:
    """Format search results into a readable string"""
    if not results:
        return "No memories found"

    lines = []
    for i, r in enumerate(results, 1):
        match_info = f"[{r.get('match_type', '?')}]"
        dist = r.get("distance", "")
        dist_str = f" (dist={dist:.3f})" if isinstance(dist, float) else ""

        lines.append(f"--- #{i} {match_info}{dist_str} [{r['session_date']}] [{r['chunk_type']}] ---")
        if r.get("question"):
            lines.append(f"Q: {r['question']}")
        lines.append(r["content"])
        if r.get("pair"):
            lines.append(f"pair: {r['pair']}")
        if r.get("tags"):
            lines.append(f"tags: {r['tags']}")
        lines.append("")

    return "\n".join(lines)


def stats():
    """DB statistics"""
    conn = get_conn()
    total = fetchone_val(conn, "SELECT COUNT(*) FROM chunks")
    by_type = fetchall_dict(conn, "SELECT chunk_type, COUNT(*) as cnt FROM chunks GROUP BY chunk_type ORDER BY cnt DESC")
    by_date = fetchall_dict(conn, "SELECT session_date, COUNT(*) as cnt FROM chunks GROUP BY session_date ORDER BY session_date DESC")

    lines = [f"Total chunks: {total}", "", "By type:"]
    for r in by_type:
        lines.append(f"  {r['chunk_type']}: {r['cnt']}")
    lines.append("\nBy date:")
    for r in by_date:
        lines.append(f"  {r['session_date']}: {r['cnt']}")
    return "\n".join(lines)


def _tokenize_query(query: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z0-9_+.%-]+", query)
    allow_short = {"M1", "M5", "H1", "H4"}
    ignored = {"AND", "THE", "FOR", "WITH", "SETUP", "TRADE", "LESSON", "LESSONS"}
    terms = []
    seen = set()
    for raw in raw_terms:
        term = raw.strip()
        if not term:
            continue
        upper = term.upper()
        if upper in ignored:
            continue
        if len(term) < 3 and upper not in allow_short and "_" not in term:
            continue
        if upper in seen:
            continue
        terms.append(term)
        seen.add(upper)
    return terms


# --- CLI ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python recall.py search '<query>' [--pair USD_JPY] [--type trade] [--top 5]")
        print("  python recall.py keyword '<keyword>' [--pair USD_JPY]")
        print("  python recall.py stats")
        sys.exit(1)

    cmd = sys.argv[1]

    pair = None
    chunk_type = None
    top_k = 5
    args = sys.argv[2:]

    query_parts = []
    i = 0
    while i < len(args):
        if args[i] == "--pair" and i + 1 < len(args):
            pair = args[i + 1]; i += 2
        elif args[i] == "--type" and i + 1 < len(args):
            chunk_type = args[i + 1]; i += 2
        elif args[i] == "--top" and i + 1 < len(args):
            top_k = int(args[i + 1]); i += 2
        else:
            query_parts.append(args[i]); i += 1

    query = " ".join(query_parts)

    if cmd == "search":
        results = hybrid_search(query, top_k=top_k, pair=pair, chunk_type=chunk_type)
        print(format_results(results))
    elif cmd == "keyword":
        results = keyword_search(query, top_k=top_k, pair=pair, chunk_type=chunk_type)
        print(format_results(results))
    elif cmd == "stats":
        print(stats())
    else:
        print(f"Unknown command: {cmd}")
