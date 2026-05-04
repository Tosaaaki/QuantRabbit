from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from quant_rabbit.models import TradeMethod
from quant_rabbit.paths import DEFAULT_LEGACY_ARCHIVE, DEFAULT_MARKET_STORY_PROFILE, DEFAULT_MARKET_STORY_REPORT


PAIR_RE = re.compile(r"\b([A-Z]{3})[_/]([A-Z]{3})\b")
VALID_PAIRS = {
    "USD_JPY",
    "EUR_USD",
    "GBP_USD",
    "AUD_USD",
    "EUR_JPY",
    "GBP_JPY",
    "AUD_JPY",
}
STORY_FILES = (
    "logs/news_digest.md",
    "logs/news_flow_log.md",
    "logs/quality_audit.md",
    "collab_trade/state.md",
    "collab_trade/strategy_memory.md",
)
NEWS_FILES = ("news_digest.md", "news_flow_log.md")


@dataclass
class StoryArtifact:
    rel_path: str
    kind: str
    lines: int


@dataclass
class PairStoryProfile:
    pair: str
    methods: Counter[str] = field(default_factory=Counter)
    themes: Counter[str] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MarketStorySummary:
    archive: Path
    report_path: Path
    profile_path: Path
    artifacts: int
    story_lines: int
    pairs: int


class MarketStoryMiner:
    def __init__(
        self,
        archive: Path = DEFAULT_LEGACY_ARCHIVE,
        report_path: Path = DEFAULT_MARKET_STORY_REPORT,
        profile_path: Path = DEFAULT_MARKET_STORY_PROFILE,
        news_root: Path | None = None,
    ) -> None:
        self.archive = archive
        self.report_path = report_path
        self.profile_path = profile_path
        self.news_root = news_root

    def run(self) -> MarketStorySummary:
        if not self.archive.exists() and self.news_root is None:
            raise FileNotFoundError(f"legacy archive not found: {self.archive}")
        artifacts = list(self._story_artifacts())
        if not artifacts:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            profile = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "archive": str(self.archive),
                "artifacts": [],
                "global_themes": {},
                "global_methods": {},
                "pair_profiles": [],
                "order_intent_contract": {
                    "required_market_context_fields": [
                        "regime",
                        "narrative",
                        "chart_story",
                        "method",
                        "invalidation",
                    ],
                    "methods": [method.value for method in TradeMethod],
                },
            }
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            self.profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            self.report_path.write_text(
                "\n".join(
                    [
                        "# Market Story Report",
                        "",
                        f"- Generated at UTC: `{profile['generated_at_utc']}`",
                        f"- Archive: `{self.archive}`",
                        "- Story artifacts read: `0`",
                        "- Narrative/chart lines mined: `0`",
                        "",
                        "## Market Story Contract",
                        "",
                        "- No compatible story artifacts found. Profile intentionally empty.",
                        "- Fill with `logs/news_digest.md` or `logs/news_flow_log.md` content to activate fresh narrative pressure.",
                    ]
                )
                + "\n"
            )
            return MarketStorySummary(
                archive=self.archive,
                report_path=self.report_path,
                profile_path=self.profile_path,
                artifacts=0,
                story_lines=0,
                pairs=0,
            )
        pair_profiles, theme_counts, method_counts, story_lines = self._mine(artifacts)
        generated_at = datetime.now(timezone.utc).isoformat()
        self._write_profile(pair_profiles, theme_counts, method_counts, artifacts, generated_at)
        self._write_report(pair_profiles, theme_counts, method_counts, artifacts, story_lines, generated_at)
        return MarketStorySummary(
            archive=self.archive,
            report_path=self.report_path,
            profile_path=self.profile_path,
            artifacts=len(artifacts),
            story_lines=story_lines,
            pairs=len(pair_profiles),
        )

    def _story_artifacts(self) -> Iterable[tuple[Path, StoryArtifact]]:
        if self.archive.exists():
            for rel in STORY_FILES:
                path = self.archive / rel
                if path.exists():
                    yield path, StoryArtifact(rel_path=rel, kind=_kind_from_path(rel), lines=_line_count(path))
        if self.news_root is not None and self.news_root.exists():
            for name in NEWS_FILES:
                path = self.news_root / name
                if path.exists():
                    rel_path = f"news/{path.name}"
                    yield path, StoryArtifact(rel_path=rel_path, kind=_kind_from_path(rel_path), lines=_line_count(path))
        daily_root = self.archive / "collab_trade" / "daily"
        if self.archive.exists() and daily_root.exists():
            for path in sorted(daily_root.glob("*/state.md")):
                rel = path.relative_to(self.archive).as_posix()
                yield path, StoryArtifact(rel_path=rel, kind="daily_state", lines=_line_count(path))

    def _mine(
        self, artifacts: list[tuple[Path, StoryArtifact]]
    ) -> tuple[list[PairStoryProfile], Counter[str], Counter[str], int]:
        profiles: dict[str, PairStoryProfile] = {}
        theme_counts: Counter[str] = Counter()
        method_counts: Counter[str] = Counter()
        story_lines = 0
        for path, artifact in artifacts:
            for raw in path.read_text(errors="replace").splitlines():
                line = _clean_line(raw)
                if not _is_story_line(line):
                    continue
                pairs = _pairs_in(line)
                themes = _themes_in(line)
                methods = _methods_in(line)
                if not pairs and not themes and not methods:
                    continue
                story_lines += 1
                theme_counts.update(themes)
                method_counts.update(method.value for method in methods)
                for pair in pairs:
                    profile = profiles.setdefault(pair, PairStoryProfile(pair=pair))
                    profile.themes.update(themes)
                    profile.methods.update(method.value for method in methods)
                    if len(profile.examples) < 5:
                        profile.examples.append(f"{artifact.kind}: {line[:240].rstrip()}")
        return sorted(profiles.values(), key=lambda item: (-sum(item.methods.values()), item.pair)), theme_counts, method_counts, story_lines

    def _write_profile(
        self,
        pair_profiles: list[PairStoryProfile],
        theme_counts: Counter[str],
        method_counts: Counter[str],
        artifacts: list[tuple[Path, StoryArtifact]],
        generated_at: str,
    ) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": generated_at,
            "archive": str(self.archive),
            "artifacts": [asdict(item) for _, item in artifacts],
            "global_themes": dict(theme_counts.most_common()),
            "global_methods": dict(method_counts.most_common()),
            "pair_profiles": [
                {
                    "pair": item.pair,
                    "methods": dict(item.methods.most_common()),
                    "themes": dict(item.themes.most_common()),
                    "examples": item.examples,
                }
                for item in pair_profiles
            ],
            "order_intent_contract": {
                "required_market_context_fields": [
                    "regime",
                    "narrative",
                    "chart_story",
                    "method",
                    "invalidation",
                ],
                "methods": [method.value for method in TradeMethod],
            },
        }
        self.profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(
        self,
        pair_profiles: list[PairStoryProfile],
        theme_counts: Counter[str],
        method_counts: Counter[str],
        artifacts: list[tuple[Path, StoryArtifact]],
        story_lines: int,
        generated_at: str,
    ) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Market Story Report",
            "",
            f"- Generated at UTC: `{generated_at}`",
            f"- Archive: `{self.archive}`",
            f"- Market story profile JSON: `{self.profile_path}`",
            f"- Story artifacts read: `{len(artifacts)}`",
            f"- Narrative/chart lines mined: `{story_lines}`",
            "",
            "## Artifacts",
            "",
        ]
        for _, artifact in artifacts:
            lines.append(f"- `{artifact.rel_path}` kind=`{artifact.kind}` lines={artifact.lines}")
        lines.extend(["", "## Global Themes", ""])
        for theme, count in theme_counts.most_common(12):
            lines.append(f"- `{theme}`: `{count}`")
        lines.extend(["", "## Method Pressure", ""])
        for method, count in method_counts.most_common():
            lines.append(f"- `{method}`: `{count}`")
        lines.extend(["", "## Pair Story Profiles", ""])
        for item in pair_profiles[:12]:
            methods = ", ".join(f"{name}={count}" for name, count in item.methods.most_common(4)) or "none"
            themes = ", ".join(f"{name}={count}" for name, count in item.themes.most_common(5)) or "none"
            lines.append(f"- `{item.pair}` methods: {methods}; themes: {themes}")
            for example in item.examples[:3]:
                lines.append(f"  - {example}")
        lines.extend(
            [
                "",
                "## Method Switching Contract",
                "",
                "- `TREND_CONTINUATION`: use only when chart story names staircase, band-walk, trend walk, impulse continuation, or shallow-pullback continuation.",
                "- `RANGE_ROTATION`: use only at exact rail/box prices; if the story says impulse, vertical leg, band-walk, or trend extension, this method is wrong.",
                "- `BREAKOUT_FAILURE`: requires trapped side, failed reclaim/break, rejection price, and body-based invalidation.",
                "- `EVENT_RISK`: central-bank, NFP, intervention, and spread-window stories reduce size or block tight-stop participation.",
                "- `POSITION_MANAGEMENT`: live unprotected exposure, margin pressure, or stale protection overrides fresh-entry hunting.",
                "- Every order intent must carry `market_context` so the system can judge method-vs-regime consistency before risk leaves the trader's hand.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _kind_from_path(rel_path: str) -> str:
    if rel_path.endswith("news_digest.md"):
        return "news_digest"
    if rel_path.endswith("news_flow_log.md"):
        return "news_flow"
    if rel_path.endswith("quality_audit.md"):
        return "quality_audit"
    if rel_path.endswith("strategy_memory.md"):
        return "strategy_memory"
    if "/daily/" in rel_path:
        return "daily_state"
    return "state"


def _line_count(path: Path) -> int:
    return len(path.read_text(errors="replace").splitlines())


def _clean_line(raw: str) -> str:
    return raw.strip().lstrip("-*#> ").strip()


def _is_story_line(line: str) -> bool:
    if len(line) < 12:
        return False
    upper = line.upper()
    return bool(_pairs_in(line) or _contains_any(upper, _THEME_NEEDLES) or _contains_any(upper, _METHOD_NEEDLES))


def _pairs_in(text: str) -> list[str]:
    pairs = {f"{match.group(1)}_{match.group(2)}" for match in PAIR_RE.finditer(text)}
    return sorted(pair for pair in pairs if pair in VALID_PAIRS)


def _themes_in(text: str) -> list[str]:
    upper = text.upper()
    themes = []
    for theme, needles in THEME_MAP.items():
        if _contains_any(upper, needles):
            themes.append(theme)
    return themes


def _methods_in(text: str) -> list[TradeMethod]:
    upper = text.upper()
    methods = []
    for method, needles in METHOD_MAP.items():
        if _contains_any(upper, needles):
            methods.append(method)
    return methods


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


THEME_MAP: dict[str, tuple[str, ...]] = {
    "central_bank": ("BOJ", "FOMC", "FED", "ECB", "BOE", "RBA", "RATE", "政策", "利上げ", "利下げ"),
    "intervention": ("INTERVENTION", "RATE CHECK", "介入", "JPY SHORT", "JPYショート"),
    "event_risk": ("NFP", "CPI", "GDP", "ISM", "PMI", "CALENDAR", "EVENT", "発表"),
    "spread_liquidity": ("SPREAD", "SL HUNT", "LIQUIDITY", "ROLLOVER", "GOLDEN WEEK", "薄", "スプレッド"),
    "momentum": ("MOMENTUM", "STAIRCASE", "BAND-WALK", "BAND WALK", "TREND WALK", "IMPULSE", "LADDER"),
    "range_rail": ("RANGE", "BOX", "RAIL", "ROTATION", "RETEST", "LOWER RAIL", "UPPER RAIL", "レンジ"),
    "breakout_failure": ("BREAKOUT", "FAILED", "FAILURE", "REJECTION", "RECLAIM", "TRAP", "REJECT"),
    "position_risk": ("UNPROTECTED", "PROTECTION", "MARGIN", "TP/SL", "SL PLAN", "LIVE RISK"),
}
METHOD_MAP: dict[TradeMethod, tuple[str, ...]] = {
    TradeMethod.TREND_CONTINUATION: ("TREND", "CONTINUATION", "STAIRCASE", "BAND-WALK", "BAND WALK", "IMPULSE", "LADDER"),
    TradeMethod.RANGE_ROTATION: ("RANGE", "BOX", "RAIL", "ROTATION", "LOWER RAIL", "UPPER RAIL"),
    TradeMethod.BREAKOUT_FAILURE: ("BREAKOUT", "FAILED", "FAILURE", "REJECTION", "RECLAIM", "TRAP", "REJECT", "RETEST"),
    TradeMethod.EVENT_RISK: ("NFP", "FOMC", "BOJ", "ECB", "BOE", "INTERVENTION", "RATE CHECK", "CPI", "GDP", "ISM"),
    TradeMethod.POSITION_MANAGEMENT: ("UNPROTECTED", "PROTECTION", "MARGIN", "TP/SL", "LIVE RISK", "MANAGE"),
}
_THEME_NEEDLES = tuple({needle for needles in THEME_MAP.values() for needle in needles})
_METHOD_NEEDLES = tuple({needle for needles in METHOD_MAP.values() for needle in needles})
