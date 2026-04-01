"""
QuantRabbit Trading Memory — Structured Parser
Extract structured data (trades / user_calls / market_events) from trades.md / notes.md
"""
from __future__ import annotations

import re
import json


# --- Trade Parser ---

def parse_trades(text: str, session_date: str) -> list[dict]:
    """Structured extraction of trade records from trades.md"""
    trades = []
    sections = re.split(r'\n(?=###\s)', text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        header_match = re.match(r'###\s+(.+)', section)
        if not header_match:
            continue
        header = header_match.group(1)

        # Trade header pattern: "GBP_USD SHORT 2000u #464922 — stop-loss -3,832JPY"
        trade_match = re.match(
            r'(\w+_\w+)\s+(LONG|SHORT)\s+(\d+)u\s+#(\d+).*?([—\-])\s*(.+)',
            header
        )
        if not trade_match:
            # Half profit-take pattern: "GBP_USD LONG 1500u #464993 half-TP 750u — +390.55JPY confirmed"
            trade_match2 = re.match(
                r'(\w+_\w+)\s+(LONG|SHORT)\s+(\d+)u\s+#(\d+)\s+(.+)',
                header
            )
            if not trade_match2:
                continue
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

        # Extract P/L
        pl = _extract_pl(rest) or _extract_pl(section)

        # Extract price
        entry_price = _extract_float(section, r'エントリー[:\s]*(\d+\.\d+)')
        if not entry_price:
            entry_price = _extract_float(section, r'@\s*(\d+\.\d+)')
        exit_price = _extract_float(section, r'(?:クローズ|TP約定|損切り)[:\s]*(\d+\.\d+)')

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

    # "User read" or "User instruction" sections
    sections = re.split(r'\n(?=##\s)', text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Detect user's market read
        header_match = re.match(r'##\s+(.+)', section)
        if not header_match:
            continue
        header = header_match.group(1)

        # Detect market-read type sections
        is_call = any(kw in header for kw in ['ユーザー読み', 'ユーザー指示', 'ユーザー', 'USER'])
        # Direct quote pattern (「」 or "User:" prefix)
        quotes = re.findall(r'「(.+?)」', section)
        # 「」がなくても "ユーザー:" や "USER:" のパターンでキャプチャ
        if not quotes:
            user_prefix = re.findall(r'(?:ユーザー|USER)[:\s：]\s*(.+?)(?:\n|$)', section, re.I)
            quotes.extend(user_prefix)

        if not is_call and not quotes:
            continue

        # Extract directional statements
        for quote in quotes:
            direction = _detect_direction(quote)
            if not direction:
                continue

            pair = _extract_pair_from_text(section)
            price = _extract_float(section, r'(?:EUR_USD|GBP_USD|USD_JPY|AUD_USD)\s+(\d+\.\d+)')
            timestamp = _extract_timestamp(header)

            # Extract conditions
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

            # Detect outcome (within same section or immediately following section)
            outcome, pl_result = _detect_outcome(section)

            # Confidence level
            confidence = 'strong'
            if any(w in quote for w in ['かも', 'そう', 'っぽい']):
                confidence = 'tentative'
            if any(w in quote for w in ['絶対', '確実', '間違いない', 'こんどこそ']):
                confidence = 'strong'

            calls.append({
                'session_date': session_date,
                'timestamp': timestamp,
                'pair': pair,
                'direction': direction,
                'call_text': quote,
                'conditions': json.dumps(conditions, ensure_ascii=False) if conditions else None,
                'price_at_call': price,
                'outcome': outcome,
                'pl_after_30m': pl_result,
                'pl_after_1h': None,
                'price_after_30m': None,
                'price_after_1h': None,
                'confidence': confidence,
                'acted_on': 1 if re.search(r'エントリー|LONG|SHORT|約定', section) else 0,
            })

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
    m = re.search(r'([+-])\s*([\d,]+(?:\.\d+)?)\s*円', text)
    if m:
        sign = 1 if m.group(1) == '+' else -1
        return sign * float(m.group(2).replace(',', ''))
    return None


def _extract_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


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


def _extract_pair_from_text(text: str) -> str | None:
    pairs = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
    for p in pairs:
        if p in text:
            return p
    return None


def _detect_direction(text: str) -> str | None:
    up_words = ['あがる', '上がる', '上がり', 'ロング', 'LONG', '反発', '上', '買い']
    down_words = ['さがる', '下がる', '下がり', 'ショート', 'SHORT', '下落', '下', '売り']
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


def _extract_timestamp(text: str) -> str | None:
    m = re.search(r'(\d{1,2}:\d{2})Z', text)
    return m.group(0) if m else None
