"""Full technical chart packet for non-FX context assets.

`cross_asset_snapshot` gives compact macro readings. This module runs the same
multi-timeframe chart stack used for FX pairs on gold, oil, indices, bonds, and
crypto so the trader can cite technical context without pretending those assets
are currently tradeable on the broker account.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from quant_rabbit.analysis.chart_reader import DEFAULT_TIMEFRAMES, build_pair_chart
from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.instruments import DEFAULT_CONTEXT_ASSETS


def build_context_asset_charts(
    *,
    client: OandaReadOnlyClient,
    instruments: Sequence[str] = DEFAULT_CONTEXT_ASSETS,
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
    count: int = 200,
) -> dict[str, object]:
    generated_at = datetime.now(timezone.utc).isoformat()
    charts: list[dict[str, object]] = []
    issues: list[str] = []
    for instrument in tuple(dict.fromkeys(str(item).upper() for item in instruments if str(item).strip())):
        try:
            chart = build_pair_chart(instrument, client=client, timeframes=timeframes, count=count).to_dict()
        except Exception as exc:  # pragma: no cover - network path
            issues.append(f"{instrument}:CONTEXT_ASSET_CHART_FAILED:{exc.__class__.__name__}:{exc}")
            continue
        warnings = [str(item) for item in chart.get("warnings", []) or [] if str(item).strip()]
        if not chart.get("views"):
            issues.append(f"{instrument}:CONTEXT_ASSET_CHART_EMPTY")
        for warning in warnings[:4]:
            issues.append(f"{instrument}:{warning}")
        charts.append(chart)
    charts.sort(key=lambda item: max(float(item.get("long_score") or 0.0), float(item.get("short_score") or 0.0)), reverse=True)
    return {
        "generated_at_utc": generated_at,
        "schema_version": 1,
        "role": "NON_FX_CONTEXT_TECHNICALS_NOT_TRADE_PERMISSION",
        "timeframes": list(timeframes),
        "candle_count": count,
        "charts": charts,
        "issues": issues[:80],
    }


def write_context_asset_charts_report(payload: dict[str, object], report_path: Path) -> None:
    lines = [
        "# Context Asset Charts",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Role: `{payload.get('role')}`",
        f"- Assets: `{len(payload.get('charts') or [])}`",
        "",
        "| Instrument | Long | Short | Regime | Story |",
        "|---|---:|---:|---|---|",
    ]
    for chart in payload.get("charts") or []:
        if not isinstance(chart, dict):
            continue
        lines.append(
            f"| `{chart.get('pair')}` | {float(chart.get('long_score') or 0.0):.4f} | "
            f"{float(chart.get('short_score') or 0.0):.4f} | `{chart.get('dominant_regime')}` | "
            f"{str(chart.get('chart_story') or '')[:180]} |"
        )
    issues = [str(item) for item in payload.get("issues", []) or [] if str(item).strip()]
    if issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in issues[:40])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
