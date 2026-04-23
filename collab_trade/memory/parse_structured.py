"""
QuantRabbit Trading Memory — Structured Parser
Extract structured data (trades / user_calls / market_events) from trades.md / notes.md
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta


PAIR_ALIASES = {
    "USD_JPY": ("USD_JPY", "ドル円"),
    "EUR_USD": ("EUR_USD", "ユーロドル"),
    "GBP_USD": ("GBP_USD", "ポンドドル"),
    "AUD_USD": ("AUD_USD", "オージードル", "豪ドルドル", "豪ドル米ドル"),
    "EUR_JPY": ("EUR_JPY", "ユーロ円"),
    "GBP_JPY": ("GBP_JPY", "ポンド円"),
    "AUD_JPY": ("AUD_JPY", "オージー円", "豪ドル円"),
}


# --- Trade Parser ---

def parse_trades(text: str, session_date: str) -> list[dict]:
    """Structured extraction of trade records from trades.md"""
    trades = []
    sections = _split_markdown_sections(text, ("##", "###"))

    for section in sections:
        if not section:
            continue

        header_match = re.match(r'#{2,3}\s+(.+)', section)
        if not header_match:
            continue
        header = header_match.group(1)
        multi_trade_rows = _extract_multi_trade_summary(section, session_date, header)
        if multi_trade_rows:
            trades.extend(multi_trade_rows)
            continue

        # Trade header pattern: "GBP_USD SHORT 2000u #464922 — stop-loss -3,832JPY"
        trade_match = re.match(
            r'(\w+_\w+)\s+(LONG|SHORT)\s+(\d+)u\s+#(\d+).*?([—\-])\s*(.+)',
            header
        )
        header_pair = _extract_pair_from_text(header)
        header_direction = _normalize_trade_direction(header)
        header_units = _extract_units(header)
        header_trade_id = _extract_trade_id(header)
        if not trade_match:
            # Half profit-take pattern: "GBP_USD LONG 1500u #464993 half-TP 750u — +390.55JPY confirmed"
            trade_match2 = re.match(
                r'(\w+_\w+)\s+(LONG|SHORT)\s+(\d+)u\s+#(\d+)\s+(.+)',
                header
            )
            if not trade_match2:
                pair = header_pair or _extract_table_field(section, ["Pair"]) or _extract_pair_from_text(section)
                direction = _normalize_trade_direction(
                    header_direction or _extract_table_field(section, ["Direction", "Side"]) or header
                )
                units = header_units or _extract_units(section)
                trade_id = header_trade_id or _extract_trade_id(section)
                rest = header
            else:
                pair = trade_match2.group(1)
                direction = trade_match2.group(2)
                units = int(trade_match2.group(3))
                trade_id = trade_match2.group(4)
                rest = trade_match2.group(5)
        else:
            pair = trade_match.group(1)
            direction = trade_match.group(2)
            units = int(trade_match.group(3))
            trade_id = trade_match.group(4)
            rest = trade_match.group(6)

        stage = _detect_trade_stage(header, section)
        if stage in {'pending', 'cancel'}:
            continue
        if stage == 'exit' and not trade_id:
            continue

        # Extract P/L only for realized close sections.
        pl = None
        if stage == 'exit' or re.search(r'PL=|P/L|半利確|HALF[_ -]?TP|クローズ|決済', f'{header}\n{section}', re.I):
            pl = _extract_pl(rest) or _extract_pl(section)

        # Extract price
        entry_price = _extract_table_price(section, ["Entry", "Fill", "Limit Price", "Limit"])
        if not entry_price:
            entry_price = _extract_float(section, r'エントリー[:\s]*(\d+\.\d+)')
        if not entry_price:
            entry_price = _extract_float(section, r'@\s*(\d+\.\d+)')
        exit_price = _extract_table_price(section, ["Close", "Exit"])
        if not exit_price:
            exit_price = _extract_float(section, r'(?:クローズ|TP約定|損切り)[:\s]*(\d+\.\d+)')
        if stage != 'exit':
            exit_price = None
        if not direction:
            direction = _infer_direction_from_prices(entry_price, exit_price, pl)
        if not pair or not direction:
            continue

        # Extract time
        hour = _extract_hour(section)

        # Extract technical values
        h1_adx = _extract_float(section, r'H1\s+ADX[=:]?(\d+\.?\d*)')
        h1_trend = _extract_trend(section, 'H1')
        m5_adx = _extract_float(section, r'M5\s+ADX[=:]?(\d+\.?\d*)')
        if not m5_adx:
            m5_adx = _extract_float(section, r'ADX[=:]?(\d+\.?\d*)')
        m5_trend = _extract_trend(section, 'M5')
        rsi = _extract_float(section, r'RSI[=:]?(\d+\.?\d*)')
        stoch_rsi = _extract_float(section, r'StochRSI[=:]?(\d+\.?\d*)')

        # Market context
        regime = _detect_regime(section)
        headlines = _extract_headlines(section)
        event_risk = _detect_event_risk(section)
        vix = _extract_float(section, r'VIX[=:]?(\d+\.?\d*)')
        dxy = _extract_float(section, r'DXY[=:]?(\d+\.?\d*)')

        # Behavior analysis
        entry_type = _detect_entry_type(section)
        had_sl = 1 if re.search(r'SL[=:]?\d|ストップ|stop.?loss', section, re.I) else 0
        if re.search(r'SLなし|SL未設定|SL撤廃', section):
            had_sl = 0

        # Extract lessons
        lesson = _extract_lesson(section)
        reason = _extract_reason(section)

        trades.append({
            'session_date': session_date,
            'trade_id': trade_id,
            'pair': pair,
            'direction': direction,
            'units': units,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pl': pl,
            'h1_adx': h1_adx,
            'h1_trend': h1_trend,
            'm5_adx': m5_adx,
            'm5_trend': m5_trend,
            'rsi': rsi,
            'stoch_rsi': stoch_rsi,
            'regime': regime,
            'vix': vix,
            'dxy': dxy,
            'active_headlines': headlines,
            'event_risk': event_risk,
            'session_hour': hour,
            'entry_type': entry_type,
            'had_sl': had_sl,
            'reason': reason,
            'lesson': lesson,
            'user_call_id': None,
        })

    return trades


# --- User Call Parser ---

def parse_user_calls(text: str, session_date: str) -> list[dict]:
    """Structured extraction of user's market reads from notes.md"""
    calls = []
    seen = set()

    sections = _split_markdown_sections(text, ("##", "###"))

    for section in sections:
        if not section:
            continue

        header = _extract_markdown_header(section)
        if not header:
            continue

        quotes = _extract_user_quotes(section, header)
        is_call = _looks_like_user_call_section(section, header)
        if not quotes and not is_call:
            continue

        for quote in quotes:
            direction = _detect_direction(quote)
            if not direction:
                continue

            timestamp = _extract_timestamp(section, session_date)
            pairs = _extract_user_call_pairs(section, quote) or [None]

            conditions = {}
            sr = _extract_float(section, r'StochRSI[=:]?(\d+\.?\d*)')
            if sr is not None:
                conditions['stoch_rsi'] = sr
            r = _extract_float(section, r'RSI[=:]?(\d+\.?\d*)')
            if r is not None:
                conditions['rsi'] = r
            adx = _extract_float(section, r'ADX[=:]?(\d+\.?\d*)')
            if adx is not None:
                conditions['adx'] = adx
            h1t = _extract_trend(section, 'H1')
            if h1t:
                conditions['h1_trend'] = h1t

            outcome, pl_result = _detect_outcome(section)

            confidence = 'strong'
            if any(w in quote for w in ['かも', 'そう', 'っぽい']):
                confidence = 'tentative'
            if any(w in quote for w in ['絶対', '確実', '間違いない', 'こんどこそ']):
                confidence = 'strong'

            for pair in pairs:
                record = {
                    'session_date': session_date,
                    'timestamp': timestamp,
                    'pair': pair,
                    'direction': direction,
                    'call_text': quote,
                    'conditions': json.dumps(conditions, ensure_ascii=False) if conditions else None,
                    'price_at_call': _extract_user_call_price(section, pair),
                    'outcome': outcome,
                    'pl_after_30m': pl_result,
                    'pl_after_1h': None,
                    'price_after_30m': None,
                    'price_after_1h': None,
                    'confidence': confidence,
                    'acted_on': 1 if re.search(r'エントリー|LONG|SHORT|約定', section) else 0,
                }
                key = (record['timestamp'], record['pair'], record['direction'], record['call_text'])
                if key in seen:
                    continue
                seen.add(key)
                calls.append(record)

    return calls


# --- Market Event Parser ---

def parse_market_events(text: str, session_date: str) -> list[dict]:
    """Extract market events (spikes etc.) from trades.md / notes.md"""
    events = []

    # Spike detection
    spike_matches = re.finditer(
        r'(\d+\.?\d*)\s*pip\s*(?:の|スパイク|巨大|急騰|急落)',
        text, re.I
    )
    for m in spike_matches:
        pips = float(m.group(1))
        if pips < 10:
            continue  # Ignore small moves

        # Extract details from surrounding context
        start = max(0, m.start() - 500)
        end = min(len(text), m.end() + 500)
        context = text[start:end]

        pair = _extract_pair_from_text(context)
        headline = _extract_headlines(context)
        event_type = _detect_event_risk(context) or 'unknown'
        direction = 'UP' if re.search(r'急騰|上昇|スパイク.*上', context) else 'DOWN'
        timestamp = _extract_timestamp(context)

        events.append({
            'session_date': session_date,
            'timestamp': timestamp,
            'event_type': event_type,
            'headline': headline,
            'pairs_affected': pair,
            'spike_pips': pips,
            'spike_direction': direction,
            'duration_min': None,
            'pre_vix': _extract_float(context, r'VIX[=:]?(\d+\.?\d*)'),
            'post_vix': None,
            'impact': 'high' if pips > 30 else 'medium' if pips > 15 else 'low',
        })

    # Headline-driven events (when not detected as spike)
    headline_patterns = [
        r'ヘッドライン.{0,50}(?:スパイク|急|暴)',
        r'(?:Iran|イラン|ホルムズ|FOMC|雇用統計|CPI).{0,100}(?:→|が|で)',
    ]
    for pat in headline_patterns:
        for m in re.finditer(pat, text):
            start = max(0, m.start() - 300)
            end = min(len(text), m.end() + 300)
            context = text[start:end]

            # Skip if already detected as a spike
            pair = _extract_pair_from_text(context)
            if any(e['pairs_affected'] == pair and e['session_date'] == session_date for e in events):
                continue

            events.append({
                'session_date': session_date,
                'timestamp': _extract_timestamp(context),
                'event_type': _detect_event_risk(context) or 'geopolitical',
                'headline': _extract_headlines(context),
                'pairs_affected': pair,
                'spike_pips': _extract_float(context, r'(\d+\.?\d*)\s*pip'),
                'spike_direction': None,
                'duration_min': None,
                'pre_vix': None,
                'post_vix': None,
                'impact': 'medium',
            })

    return events


# --- Helper Functions ---

def _extract_pl(text: str) -> float | None:
    m = re.search(r'([+-])\s*([\d,]+(?:\.\d+)?)\s*(?:円|JPY)', text, re.I)
    if m:
        sign = 1 if m.group(1) == '+' else -1
        return sign * float(m.group(2).replace(',', ''))
    return None


def _extract_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def _extract_markdown_header(text: str) -> str | None:
    match = re.match(r'#{2,3}\s+(.+)', text)
    return match.group(1).strip() if match else None


def _split_markdown_sections(text: str, header_levels: tuple[str, ...]) -> list[str]:
    escaped = "|".join(re.escape(level) for level in header_levels)
    pattern = rf"\n(?=(?:{escaped})\s)"
    return [section.strip() for section in re.split(pattern, text) if section.strip()]


def _extract_table_field(text: str, labels: list[str]) -> str | None:
    for label in labels:
        match = re.search(
            rf"^\|\s*{re.escape(label)}\s*\|\s*(.+?)\s*\|$",
            text,
            re.M | re.I,
        )
        if match:
            return match.group(1).strip()
    return None


def _extract_units(text: str) -> int | None:
    match = re.search(r'(\d+)u', text)
    if match:
        return int(match.group(1))
    table_val = _extract_table_field(text, ["Units"])
    if table_val:
        table_match = re.search(r'(\d+)', table_val.replace(",", ""))
        if table_match:
            return int(table_match.group(1))
    return None


def _extract_trade_id(text: str) -> str | None:
    for pattern in (
        r'(?<![A-Za-z])trade\s+id\s*[#=:]?\s*(\d{5,})\b',
        r'(?<![A-Za-z])order\s+id\s*[#=:]?\s*(\d{5,})\b',
        r'(?<![A-Za-z])id\s*[#=:]?\s*(\d{5,})\b',
    ):
        header_match = re.search(pattern, text, re.I)
        if header_match:
            return header_match.group(1)
    table_val = _extract_table_field(text, ["id", "Trade ID", "Order ID"])
    if table_val:
        for pattern in (r'trade\s+id\s*[#=:]?\s*(\d{5,})\b', r'(\d{5,})'):
            match = re.search(pattern, table_val, re.I)
            if match:
                return match.group(1)
    return None


def _detect_trade_stage(header: str, section: str) -> str | None:
    header_upper = header.upper()
    upper = f'{header}\n{section}'.upper()
    normalized_header = re.sub(r'\s+', ' ', header_upper).strip()

    if normalized_header in {'CLOSED', 'OPEN', 'PENDING LIMITS'}:
        return 'summary'
    if normalized_header.startswith('OPEN POSITION'):
        return 'position'
    if re.search(r'\b(TP FILL|TPS CONFIRMED|HALF-TP|HALF TP|TAKE_PROFIT|STOP-LOSS|STOP LOSS|CLOSE|CLOSED)\b', header_upper):
        return 'exit'
    if re.search(r'\b(FILL|FILLED)\b', header_upper):
        return 'entry'
    if re.search(r'\b(MARKET)\b', header_upper) and re.search(r'\b(LONG|SHORT)\b', header_upper):
        return 'entry'
    if re.search(r'\b(LIMIT|PENDING|ENTRY ORDER|STOP ENTRY|STOP-ENTRY)\b', header_upper) and not re.search(r'\b(FILL|FILLED|CLOSE|CLOSED)\b', header_upper):
        return 'pending'
    if re.search(r'^\|\s*Type\s*\|\s*(LIMIT|STOP)\s*\|', section, re.M | re.I) and not re.search(r'\b(FILLED|CLOSE|CLOSED)\b', header_upper):
        return 'pending'
    if header_upper.startswith('ENTRY ') or ' FILLED' in header_upper:
        return 'entry'
    if any(token in upper for token in ['POSITIONS CARRIED']):
        return 'carry'
    if any(token in header_upper for token in ['CANCEL', 'CANCELLED']):
        return 'cancel'
    if re.search(r'^\|\s*(Close time|Close price|P/L|決済時刻|決済価格)\s*\|', section, re.M | re.I):
        return 'exit'
    if any(token in upper for token in ['HALF-TP', 'HALF TP', 'TAKE_PROFIT', 'STOP-LOSS', 'STOP LOSS', 'クローズ', '決済']):
        return 'exit'
    if any(token in upper for token in ['FILLED', 'OPEN TIME', 'TRADE ID']):
        return 'entry'
    if any(token in upper for token in ['PENDING', 'ENTRY ORDER', 'LIMIT PLACED', 'STOP ENTRY', 'STOP-ENTRY', 'GTD', '| TYPE | LIMIT |', '| TYPE | STOP |']):
        return 'pending'
    if any(token in upper for token in ['ENTRY', 'MARKET ORDER', 'MARKET ENTRY']):
        return 'entry'
    if 'POSITION' in upper:
        return 'position'
    return None


def _extract_table_price(text: str, labels: list[str]) -> float | None:
    raw = _extract_table_field(text, labels)
    if not raw:
        return None
    match = re.search(r'(\d+\.\d+)', raw.replace(",", ""))
    return float(match.group(1)) if match else None


def _infer_direction_from_prices(entry_price: float | None, exit_price: float | None, pl: float | None) -> str | None:
    if entry_price is None or exit_price is None:
        return None
    delta = exit_price - entry_price
    if abs(delta) < 1e-9:
        return None
    if pl is None:
        return None
    if delta > 0:
        return 'LONG' if pl >= 0 else 'SHORT'
    return 'SHORT' if pl >= 0 else 'LONG'


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def _extract_multi_trade_summary(section: str, session_date: str, header: str) -> list[dict]:
    table_lines = [line.strip() for line in section.splitlines() if line.strip().startswith('|')]
    if len(table_lines) < 3:
        return []
    header_cells = _split_markdown_row(table_lines[0])
    if len(header_cells) < 3 or header_cells[0].lower() != 'field':
        return []

    trade_ids = []
    for cell in header_cells[1:]:
        match = re.search(r'(\d{5,})', cell)
        if not match:
            return []
        trade_ids.append(match.group(1))

    pair = _extract_pair_from_text(header) or _extract_pair_from_text(section)
    if not pair:
        return []

    row_map: dict[str, list[str]] = {}
    for line in table_lines[2:]:
        cells = _split_markdown_row(line)
        if len(cells) != len(trade_ids) + 1:
            continue
        row_map[cells[0].lower()] = cells[1:]

    if not row_map:
        return []

    session_hour = _extract_hour(section)
    entry_type = _detect_entry_type(section)
    had_sl = 1 if re.search(r'SL[=:]?\d|ストップ|stop.?loss', section, re.I) else 0
    if re.search(r'SLなし|SL未設定|SL撤廃', section):
        had_sl = 0

    records = []
    for idx, trade_id in enumerate(trade_ids):
        entry_cell = row_map.get('entry', [None] * len(trade_ids))[idx]
        exit_cell = row_map.get('exit', row_map.get('close', [None] * len(trade_ids)))[idx]
        pl_cell = row_map.get('p&l', [None] * len(trade_ids))[idx]
        reason_cell = row_map.get('reason', [None] * len(trade_ids))[idx] if 'reason' in row_map else None

        units = _extract_units(entry_cell or '')
        entry_price = _extract_float(entry_cell or '', r'@(\d+\.\d+)')
        exit_price = _extract_float(exit_cell or '', r'@(\d+\.\d+)')
        pl = _extract_pl(pl_cell or '')
        direction = _infer_direction_from_prices(entry_price, exit_price, pl)
        if not direction:
            continue

        reason = reason_cell
        if not reason and exit_cell:
            if 'TP' in exit_cell.upper():
                reason = 'TP summary'
            elif 'SL' in exit_cell.upper():
                reason = 'SL summary'

        records.append({
            'session_date': session_date,
            'trade_id': trade_id,
            'pair': pair,
            'direction': direction,
            'units': units,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pl': pl,
            'h1_adx': None,
            'h1_trend': None,
            'm5_adx': None,
            'm5_trend': None,
            'rsi': None,
            'stoch_rsi': None,
            'regime': None,
            'vix': None,
            'dxy': None,
            'active_headlines': None,
            'event_risk': None,
            'session_hour': session_hour,
            'entry_type': entry_type,
            'had_sl': had_sl,
            'reason': reason,
            'lesson': None,
            'user_call_id': None,
        })

    return records


def _normalize_trade_direction(text: str | None) -> str | None:
    if not text:
        return None
    upper = text.upper()
    if "SHORT" in upper:
        return "SHORT"
    if "LONG" in upper:
        return "LONG"
    return None


def _extract_hour(text: str) -> int | None:
    m = re.search(r'(\d{1,2}):?\d{2}Z', text)
    return int(m.group(1)) if m else None


def _extract_trend(text: str, tf: str) -> str | None:
    """Extract trend direction for TF (H1/M5 etc.)"""
    patterns = [
        rf'{tf}\s+(?:ADX[=:]\d+\s+)?(?:DI\+[=:]\d+[^)]*\s+)?(BULL|BEAR|FLAT)',
        rf'{tf}[^.]*?(BULL|BEAR)',
        rf'{tf}[^.]*?(上昇|下降|ベア|ブル)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            val = m.group(1).upper()
            if val in ('上昇', 'ブル'):
                return 'BULL'
            if val in ('下降', 'ベア'):
                return 'BEAR'
            return val
    return None


def _detect_regime(text: str) -> str | None:
    if re.search(r'ヘッドライン|スパイク|急騰|急落|ニュース|headline', text, re.I):
        return 'headline'
    if re.search(r'チョッピー|レンジ|横ばい|方向感なし|range|choppy', text, re.I):
        return 'quiet'
    if re.search(r'薄商い|流動性|thin|holiday', text, re.I):
        return 'thin_liquidity'
    # Determine from ADX value (ADX>30 = trending)
    adx_match = re.search(r'ADX[=:]?\s*(\d+\.?\d*)', text)
    if adx_match:
        adx_val = float(adx_match.group(1))
        if adx_val >= 30:
            return 'trending'
        if adx_val < 15:
            return 'quiet'
    if re.search(r'strong.?(?:bear|bull|trend)|強(?:ベア|ブル|bull|bear)|トレンド(?:強|発生)', text, re.I):
        return 'trending'
    if re.search(r'squeeze|スクイーズ|ブレイク待ち', text, re.I):
        return 'squeeze'
    return None  # Undeterminable returns None (don't overwrite with 'quiet')


def _extract_headlines(text: str) -> str | None:
    keywords = []
    patterns = [
        (r'(?:ホルムズ|Hormuz)', 'Hormuz'),
        (r'(?:イラン|Iran)', 'Iran'),
        (r'(?:FOMC|Fed)', 'Fed/FOMC'),
        (r'(?:雇用統計|NFP)', 'NFP'),
        (r'(?:CPI|インフレ)', 'CPI'),
        (r'(?:BOE|BOJ|ECB|RBA)', lambda m: m.group(0)),
        (r'risk.?(?:off|on)', lambda m: m.group(0)),
        (r'(?:地政学|geopolitical)', 'geopolitical'),
        (r'(?:原油|WTI|oil)', 'oil'),
    ]
    for pat, label in patterns:
        if re.search(pat, text, re.I):
            if callable(label):
                m = re.search(pat, text, re.I)
                keywords.append(label(m))
            else:
                keywords.append(label)
    return ', '.join(keywords) if keywords else None


def _detect_event_risk(text: str) -> str | None:
    if re.search(r'ホルムズ|Iran|イラン|地政学|戦争|紛争|制裁', text, re.I):
        return 'geopolitical'
    if re.search(r'FOMC|Fed|BOJ|BOE|ECB|RBA|金利|利上げ|利下げ', text, re.I):
        return 'central_bank'
    if re.search(r'雇用統計|NFP|CPI|GDP|ISM|PMI', text, re.I):
        return 'data_release'
    return None


def _detect_entry_type(text: str) -> str:
    if re.search(r'追っかけ|FOMO|飛びつ', text, re.I):
        return 'fomo'
    if re.search(r'即再エントリー|TP後に即|回転', text):
        return 're_entry'
    if re.search(r'ユーザー指[示事]|ユーザー.*読み', text):
        return 'user_directed'
    if re.search(r'リベンジ|revenge|取り返', text, re.I):
        return 'revenge'
    return 'planned'


def _extract_lesson(text: str) -> str | None:
    # 1. "Lesson:" "Reflection:" pattern (with or without bold)
    lessons = re.findall(r'(?:教訓|反省)[:\s：]*\*?\*?(.+?)(?:\*\*|\n|$)', text)
    if lessons:
        return ' / '.join(l.strip() for l in lessons[:3])
    table_lesson = _extract_table_field(text, ["Lesson"])
    if table_lesson:
        return table_lesson
    # 2. Keywords inside bold text
    bold = re.findall(r'\*\*([^*]+)\*\*', text)
    lesson_bold = [b for b in bold if any(kw in b for kw in ['教訓', '反省', 'ミス', '禁止', 'NG'])]
    if lesson_bold:
        return ' / '.join(lesson_bold[:3])
    # 3. Learning-type keywords in line (supports plain text)
    learn_lines = re.findall(r'^.*(?:正解|失敗|学び|次回|ミス|注意)[:\s：](.+?)$', text, re.M)
    if learn_lines:
        return ' / '.join(l.strip() for l in learn_lines[:3])
    # 4. reason= or lesson= format (from live_trade_log)
    reason_lessons = re.findall(r'(?:reason|lesson)[=:]\s*(.+?)(?:\||$|\n)', text)
    if reason_lessons:
        return ' / '.join(l.strip() for l in reason_lessons[:3])
    return None


def _extract_reason(text: str) -> str | None:
    table_reason = _extract_table_field(text, ["Reason", "Thesis"])
    if table_reason:
        return table_reason
    m = re.search(r'テーゼ[:\s]*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'根拠[:\s]*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'理由(?:\(入\))?[:\s]*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    return None


def _extract_pairs_from_text(text: str) -> list[str]:
    matches = []
    seen = set()
    upper = text.upper()
    for pair, aliases in PAIR_ALIASES.items():
        if pair in upper or any(alias in text for alias in aliases if alias != pair):
            if pair not in seen:
                matches.append(pair)
                seen.add(pair)
    return matches


def _extract_pair_from_text(text: str) -> str | None:
    pairs = _extract_pairs_from_text(text)
    return pairs[0] if pairs else None


def _looks_like_user_call_section(section: str, header: str) -> bool:
    lowered_header = header.lower()
    if any(token in lowered_header for token in ("user", "remarks", "message")):
        return True
    return bool(re.search(r'ユーザー(?:読み|指示|発言)?', section, re.I))


def _find_quoted_phrases(text: str) -> list[str]:
    quotes = []
    patterns = [
        r'「([^」]+)」',
        r'"([^"\n]+)"',
        r'“([^”]+)”',
    ]
    for pattern in patterns:
        quotes.extend(match.strip() for match in re.findall(pattern, text) if match.strip())
    return quotes


def _extract_user_quotes(section: str, header: str) -> list[str]:
    quotes = []
    seen = set()

    def add(items):
        for item in items:
            cleaned = " ".join(item.split())
            if cleaned and cleaned not in seen:
                quotes.append(cleaned)
                seen.add(cleaned)

    add(_find_quoted_phrases(header))

    for line in section.splitlines()[1:6]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if re.fullmatch(r'["“「].+["”」]', stripped):
            add(_find_quoted_phrases(stripped))
        prefix_match = re.match(
            r'(?:ユーザー|user(?:\s+(?:message|remarks?|call))?)[:：\s-]+(.+)$',
            stripped,
            re.I,
        )
        if prefix_match:
            add(_find_quoted_phrases(prefix_match.group(1)) or [prefix_match.group(1)])

    return quotes


def _extract_user_call_pair(section: str, quote: str) -> str | None:
    pairs = _extract_user_call_pairs(section, quote)
    if len(pairs) == 1:
        return pairs[0]
    return None


def _extract_user_call_pairs(section: str, quote: str) -> list[str]:
    quote_pairs = _extract_pairs_from_text(quote)
    if quote_pairs:
        return quote_pairs
    section_pairs = _extract_pairs_from_text(section)
    if len(section_pairs) == 1:
        return section_pairs
    return []


def _extract_user_call_price(section: str, pair: str | None) -> float | None:
    if not pair:
        return None
    for label in PAIR_ALIASES.get(pair, (pair,)):
        match = re.search(rf'{re.escape(label)}[^\n\r]{{0,48}}?(\d+\.\d+)', section, re.I)
        if match:
            return float(match.group(1))
    return None


def _detect_direction(text: str) -> str | None:
    if any(word in text for word in ['ホールド', '保持', '待って', '耐え']) and not any(
        token in text for token in ['LONG', 'SHORT', 'ロング', 'ショート', '上がる', '下がる', '反発', '下落', '割れる']
    ):
        return None

    up_words = ['あがる', '上がる', '上がり', 'ロング', 'LONG', '反発', '回復', '戻す', 'プラス方向', 'V字回復']
    down_words = ['さがる', '下がる', '下がり', 'ショート', 'SHORT', '下落', '崩れる', '割れる', '落ちる']
    for w in up_words:
        if w in text:
            return 'UP'
    for w in down_words:
        if w in text:
            return 'DOWN'
    return None


def _detect_outcome(text: str) -> tuple[str | None, float | None]:
    if re.search(r'的中|成功|正解', text):
        pl = _extract_pl(text)
        return ('correct', pl)
    if re.search(r'失敗|外れ|ミス|損', text):
        pl = _extract_pl(text)
        return ('incorrect', pl)
    return (None, None)


def _extract_timestamp(text: str, session_date: str | None = None) -> str | None:
    utc_match = re.search(r'(\d{1,2}:\d{2})\s*(?:UTC|Z)\b', text, re.I)
    if utc_match:
        return f"{utc_match.group(1)}Z"

    jst_match = re.search(r'(\d{1,2}:\d{2})\s*JST\b', text, re.I)
    if jst_match and session_date:
        try:
            local_dt = datetime.strptime(f"{session_date} {jst_match.group(1)}", "%Y-%m-%d %H:%M")
            utc_dt = local_dt - timedelta(hours=9)
            return utc_dt.strftime("%H:%MZ")
        except ValueError:
            return None
    return None
