"""
QuantRabbit Trading Memory — Recall (Search & Retrieve)
ベクトル検索 + キーワード検索のハイブリッドで過去の記憶を引く
"""
import sys
from schema import get_conn, serialize_f32, fetchall_dict, fetchone_val

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("cl-nagoya/ruri-v3-30m")
    return _model


def embed_query(text: str) -> list[float]:
    """Ruri v3 のクエリ用プレフィックス付き"""
    model = get_model()
    vec = model.encode([f"クエリ: {text}"], normalize_embeddings=True)
    return vec[0].tolist()


def vector_search(query: str, top_k: int = 5, pair: str | None = None,
                   chunk_type: str | None = None) -> list[dict]:
    """ベクトル類似度検索"""
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

    sql = f"SELECT * FROM chunks WHERE {' AND '.join(where_clauses)}"
    results = fetchall_dict(conn, sql, params)

    results = sorted(results, key=lambda r: dist_map.get(r["id"], 999))
    return [r | {"distance": dist_map.get(r["id"], 999)} for r in results[:top_k]]


def keyword_search(keyword: str, top_k: int = 10, pair: str | None = None,
                    chunk_type: str | None = None) -> list[dict]:
    """キーワード全文検索（LIKE）"""
    conn = get_conn()

    where_clauses = ["(content LIKE ? OR question LIKE ?)"]
    params = [f"%{keyword}%", f"%{keyword}%"]

    if pair:
        where_clauses.append("pair = ?")
        params.append(pair)
    if chunk_type:
        where_clauses.append("chunk_type = ?")
        params.append(chunk_type)

    sql = f"""
        SELECT * FROM chunks
        WHERE {' AND '.join(where_clauses)}
        ORDER BY session_date DESC
        LIMIT ?
    """
    params.append(top_k)
    return fetchall_dict(conn, sql, params)


def hybrid_search(query: str, top_k: int = 5, pair: str | None = None,
                   chunk_type: str | None = None) -> list[dict]:
    """ベクトル + キーワードのハイブリッド検索（重複排除）"""
    vec_results = vector_search(query, top_k=top_k, pair=pair, chunk_type=chunk_type)
    kw_results = keyword_search(query, top_k=top_k, pair=pair, chunk_type=chunk_type)

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
    """検索結果を読みやすい文字列に"""
    if not results:
        return "記憶なし"

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
    """DBの統計情報"""
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
