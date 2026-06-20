from __future__ import annotations

import importlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from quant_rabbit.paths import DEFAULT_ENV_LOCAL


SDK_INSTALL_COMMAND = "pip3 install --upgrade webull-openapi-python-sdk"
WEBULL_ENV_KEYS = {
    "QR_WEBULL_APP_KEY",
    "QR_WEBULL_APP_SECRET",
    "QR_WEBULL_ACCOUNT_ID",
    "QR_WEBULL_ENV",
    "QR_WEBULL_REGION",
    "QR_WEBULL_ENDPOINT",
}
WEBULL_ENDPOINTS = {
    "test": "us-openapi-alb.uat.webullbroker.com",
    "uat": "us-openapi-alb.uat.webullbroker.com",
    "production": "api.webull.com",
    "prod": "api.webull.com",
}
WEBULL_ORDER_ID_PREFIX = "qrv1wb"


@dataclass(frozen=True)
class WebullSDKStatus:
    installed: bool
    error: str | None = None
    install_command: str = SDK_INSTALL_COMMAND


@dataclass(frozen=True)
class WebullConfig:
    app_key: str | None
    app_secret: str | None
    account_id: str | None = None
    environment: str = "test"
    region: str = "us"
    endpoint: str | None = None
    env_file: Path = DEFAULT_ENV_LOCAL

    @classmethod
    def from_env(cls, *, env_file: Path | None = None) -> "WebullConfig":
        configured_file = env_file or _configured_env_file()
        env_values = _load_env_file(configured_file)
        environment = (
            os.environ.get("QR_WEBULL_ENV")
            or env_values.get("QR_WEBULL_ENV")
            or "test"
        ).strip().lower()
        region = (os.environ.get("QR_WEBULL_REGION") or env_values.get("QR_WEBULL_REGION") or "us").strip().lower()
        endpoint = os.environ.get("QR_WEBULL_ENDPOINT") or env_values.get("QR_WEBULL_ENDPOINT")
        return cls(
            app_key=os.environ.get("QR_WEBULL_APP_KEY") or env_values.get("QR_WEBULL_APP_KEY"),
            app_secret=os.environ.get("QR_WEBULL_APP_SECRET") or env_values.get("QR_WEBULL_APP_SECRET"),
            account_id=os.environ.get("QR_WEBULL_ACCOUNT_ID") or env_values.get("QR_WEBULL_ACCOUNT_ID"),
            environment=environment,
            region=region,
            endpoint=(endpoint or "").strip() or None,
            env_file=configured_file,
        )

    @property
    def resolved_endpoint(self) -> str:
        if self.endpoint:
            return self.endpoint.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
        return WEBULL_ENDPOINTS.get(self.environment, WEBULL_ENDPOINTS["test"])

    @property
    def credentials_ready(self) -> bool:
        return bool(self.app_key and self.app_secret)

    @property
    def live_enabled(self) -> bool:
        return os.environ.get("QR_WEBULL_LIVE_ENABLED", "").strip() in {"1", "true", "TRUE", "yes", "YES"}

    def require_credentials(self) -> None:
        if not self.app_key or not self.app_secret:
            raise RuntimeError("Webull OpenAPI requires QR_WEBULL_APP_KEY and QR_WEBULL_APP_SECRET")

    def require_account_id(self, account_id: str | None = None) -> str:
        selected = (account_id or self.account_id or "").strip()
        if not selected:
            raise RuntimeError("Webull account calls require QR_WEBULL_ACCOUNT_ID or --account-id")
        return selected

    def safe_status(self) -> dict[str, Any]:
        sdk = webull_sdk_status()
        issues: list[dict[str, str]] = []
        if not self.app_key:
            issues.append({"severity": "BLOCK", "code": "WEBULL_APP_KEY_MISSING"})
        if not self.app_secret:
            issues.append({"severity": "BLOCK", "code": "WEBULL_APP_SECRET_MISSING"})
        if not sdk.installed:
            issues.append({"severity": "BLOCK", "code": "WEBULL_SDK_MISSING", "message": sdk.install_command})
        return {
            "status": "READY" if not issues else "BLOCKED",
            "env_file": str(self.env_file),
            "environment": self.environment,
            "region": self.region,
            "endpoint": self.resolved_endpoint,
            "credentials": {
                "app_key_present": bool(self.app_key),
                "app_secret_present": bool(self.app_secret),
                "account_id_present": bool(self.account_id),
            },
            "live_enabled": self.live_enabled,
            "sdk": sdk.__dict__,
            "issues": issues,
        }


@dataclass(frozen=True)
class WebullStockOrder:
    symbol: str
    side: str
    quantity: str
    order_type: str = "LIMIT"
    limit_price: str | None = None
    stop_price: str | None = None
    time_in_force: str = "DAY"
    support_trading_session: str = "CORE"
    client_order_id: str | None = None
    entrust_type: str = "QTY"
    instrument_type: str = "EQUITY"
    market: str = "US"

    def validate(self) -> tuple[dict[str, str], ...]:
        issues: list[dict[str, str]] = []
        symbol = self.symbol.strip().upper()
        side = self.side.strip().upper()
        order_type = self.order_type.strip().upper().replace("_", "-")
        tif = self.time_in_force.strip().upper()
        session = self.support_trading_session.strip().upper()
        entrust = self.entrust_type.strip().upper()

        if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
            issues.append({"severity": "BLOCK", "code": "WEBULL_SYMBOL_INVALID", "message": "symbol must be a US stock/ETF ticker"})
        if side not in {"BUY", "SELL", "SHORT"}:
            issues.append({"severity": "BLOCK", "code": "WEBULL_SIDE_INVALID", "message": "side must be BUY, SELL, or SHORT"})
        if order_type not in {"MARKET", "LIMIT", "STOP-LOSS", "STOP-LOSS-LIMIT", "TRAILING-STOP-LOSS"}:
            issues.append({"severity": "BLOCK", "code": "WEBULL_ORDER_TYPE_INVALID", "message": "unsupported stock order type"})
        if tif not in {"DAY", "GTC"}:
            issues.append({"severity": "BLOCK", "code": "WEBULL_TIME_IN_FORCE_INVALID", "message": "time_in_force must be DAY or GTC"})
        if session not in {"CORE", "ALL", "NIGHT"}:
            issues.append({"severity": "BLOCK", "code": "WEBULL_SESSION_INVALID", "message": "support_trading_session must be CORE, ALL, or NIGHT"})
        if entrust != "QTY":
            issues.append({"severity": "BLOCK", "code": "WEBULL_ENTRUST_TYPE_UNSUPPORTED", "message": "only QTY stock orders are supported initially"})
        if _positive_decimal(self.quantity) is None:
            issues.append({"severity": "BLOCK", "code": "WEBULL_QUANTITY_INVALID", "message": "quantity must be a positive number"})
        if order_type in {"LIMIT", "STOP-LOSS-LIMIT"} and _positive_decimal(self.limit_price) is None:
            issues.append({"severity": "BLOCK", "code": "WEBULL_LIMIT_PRICE_REQUIRED", "message": "limit_price is required for limit orders"})
        if order_type in {"STOP-LOSS", "STOP-LOSS-LIMIT"} and _positive_decimal(self.stop_price) is None:
            issues.append({"severity": "BLOCK", "code": "WEBULL_STOP_PRICE_REQUIRED", "message": "stop_price is required for stop orders"})
        client_order_id = self.client_order_id or ""
        if client_order_id and len(client_order_id) > 32:
            issues.append({"severity": "BLOCK", "code": "WEBULL_CLIENT_ORDER_ID_TOO_LONG", "message": "client_order_id must be 32 characters or fewer"})
        return tuple(issues)

    def to_webull_payload(self) -> dict[str, str]:
        issues = self.validate()
        if any(issue["severity"] == "BLOCK" for issue in issues):
            codes = ",".join(issue["code"] for issue in issues)
            raise ValueError(f"invalid Webull stock order: {codes}")
        order_type = self.order_type.strip().upper().replace("-", "_")
        payload = {
            "combo_type": "NORMAL",
            "client_order_id": self.client_order_id or _new_client_order_id(),
            "symbol": self.symbol.strip().upper(),
            "instrument_type": self.instrument_type.strip().upper(),
            "market": self.market.strip().upper(),
            "order_type": order_type,
            "quantity": _decimal_text(self.quantity),
            "support_trading_session": self.support_trading_session.strip().upper(),
            "side": self.side.strip().upper(),
            "time_in_force": self.time_in_force.strip().upper(),
            "entrust_type": self.entrust_type.strip().upper(),
        }
        if self.limit_price is not None:
            payload["limit_price"] = _decimal_text(self.limit_price)
        if self.stop_price is not None:
            payload["stop_price"] = _decimal_text(self.stop_price)
        return payload


@dataclass(frozen=True)
class WebullStageSummary:
    status: str
    output_path: Path
    report_path: Path
    sent: bool
    issues: tuple[dict[str, str], ...] = ()
    preview_status_code: int | None = None
    place_status_code: int | None = None


class WebullOpenAPIClient:
    """Thin wrapper around the official Webull OpenAPI SDK.

    This module intentionally does not implement raw password/phone login.
    Webull OpenAPI uses App Key/App Secret credentials and optional 2FA token
    handling through the official SDK.
    """

    def __init__(self, config: WebullConfig | None = None) -> None:
        self.config = config or WebullConfig.from_env()
        self.config.require_credentials()
        modules = _import_webull_sdk()
        api_client = modules["ApiClient"](self.config.app_key, self.config.app_secret, self.config.region)
        api_client.add_endpoint(self.config.region, self.config.resolved_endpoint)
        self.trade_client = modules["TradeClient"](api_client)

    def account_list(self) -> dict[str, Any]:
        return _sdk_response_payload(self.trade_client.account_v2.get_account_list())

    def account_balance(self, account_id: str | None = None) -> dict[str, Any]:
        selected_account_id = self.config.require_account_id(account_id)
        return _sdk_response_payload(self.trade_client.account_v2.get_account_balance(selected_account_id))

    def account_positions(self, account_id: str | None = None) -> dict[str, Any]:
        selected_account_id = self.config.require_account_id(account_id)
        return _sdk_response_payload(self.trade_client.account_v2.get_account_position(selected_account_id))

    def account_snapshot(self, account_id: str | None = None) -> dict[str, Any]:
        selected_account_id = self.config.require_account_id(account_id)
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "broker": "webull",
            "environment": self.config.environment,
            "endpoint": self.config.resolved_endpoint,
            "account_id": selected_account_id,
            "account_list": self.account_list(),
            "balance": self.account_balance(selected_account_id),
            "positions": self.account_positions(selected_account_id),
        }

    def preview_stock_order(self, account_id: str, new_orders: list[dict[str, Any]]) -> dict[str, Any]:
        return _sdk_response_payload(self.trade_client.order_v2.preview_order(account_id, new_orders))

    def place_stock_order(self, account_id: str, new_orders: list[dict[str, Any]]) -> dict[str, Any]:
        return _sdk_response_payload(self.trade_client.order_v2.place_order(account_id, new_orders))


class WebullStockOrderGateway:
    def __init__(
        self,
        *,
        client: WebullOpenAPIClient | None = None,
        config: WebullConfig | None = None,
        output_path: Path,
        report_path: Path,
    ) -> None:
        self.config = config or WebullConfig.from_env()
        self.client = client
        self.output_path = output_path
        self.report_path = report_path

    def run(
        self,
        *,
        order: WebullStockOrder,
        account_id: str | None = None,
        preview: bool = False,
        send: bool = False,
        confirm_live: bool = False,
    ) -> WebullStageSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        issues = list(order.validate())
        selected_account_id = account_id or self.config.account_id
        if (preview or send) and not selected_account_id:
            issues.append({"severity": "BLOCK", "code": "WEBULL_ACCOUNT_ID_MISSING", "message": "preview/send requires QR_WEBULL_ACCOUNT_ID or --account-id"})
        if send and not confirm_live:
            issues.append({"severity": "BLOCK", "code": "WEBULL_CONFIRM_LIVE_MISSING", "message": "--confirm-live is required with --send"})
        if send and not self.config.live_enabled:
            issues.append({"severity": "BLOCK", "code": "WEBULL_LIVE_DISABLED", "message": "QR_WEBULL_LIVE_ENABLED=1 is required with --send"})

        order_payload = None
        try:
            if not any(issue["severity"] == "BLOCK" for issue in issues):
                order_payload = order.to_webull_payload()
            elif not any(issue["code"].startswith("WEBULL_") and issue["code"].endswith("_INVALID") for issue in issues):
                order_payload = order.to_webull_payload()
        except ValueError:
            order_payload = None

        preview_response = None
        place_response = None
        if (preview or send) and order_payload is not None and not any(issue["severity"] == "BLOCK" for issue in issues):
            client = self.client or WebullOpenAPIClient(self.config)
            preview_response = client.preview_stock_order(str(selected_account_id), [order_payload])
            if not _response_ok(preview_response):
                issues.append({"severity": "BLOCK", "code": "WEBULL_PREVIEW_REJECTED", "message": "Webull preview did not return 2xx"})
        if send and order_payload is not None and not any(issue["severity"] == "BLOCK" for issue in issues):
            client = self.client or WebullOpenAPIClient(self.config)
            place_response = client.place_stock_order(str(selected_account_id), [order_payload])
            if not _response_ok(place_response):
                issues.append({"severity": "BLOCK", "code": "WEBULL_PLACE_REJECTED", "message": "Webull place order did not return 2xx"})

        blocked = any(issue["severity"] == "BLOCK" for issue in issues)
        sent = bool(send and place_response and _response_ok(place_response) and not blocked)
        status = "BLOCKED" if blocked else "SENT" if sent else "STAGED"
        result = {
            "generated_at_utc": generated_at,
            "broker": "webull",
            "environment": self.config.environment,
            "endpoint": self.config.resolved_endpoint,
            "account_id_present": bool(selected_account_id),
            "send_requested": send,
            "preview_requested": preview or send,
            "sent": sent,
            "status": status,
            "issues": issues,
            "order_request": {
                "account_id_present": bool(selected_account_id),
                "new_orders": [order_payload] if order_payload else [],
            },
            "preview_response": preview_response,
            "place_response": place_response,
        }
        _write_json(self.output_path, result)
        _write_webull_stage_report(self.report_path, result)
        return WebullStageSummary(
            status=status,
            output_path=self.output_path,
            report_path=self.report_path,
            sent=sent,
            issues=tuple(issues),
            preview_status_code=_response_status_code(preview_response),
            place_status_code=_response_status_code(place_response),
        )


def webull_sdk_status() -> WebullSDKStatus:
    try:
        _import_webull_sdk()
    except RuntimeError as exc:
        return WebullSDKStatus(installed=False, error=str(exc))
    return WebullSDKStatus(installed=True)


def write_webull_env_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Webull OpenAPI Environment Check",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Environment: `{payload.get('environment')}`",
        f"- Endpoint: `{payload.get('endpoint')}`",
        f"- SDK installed: `{bool((payload.get('sdk') or {}).get('installed'))}`",
        f"- App key present: `{bool((payload.get('credentials') or {}).get('app_key_present'))}`",
        f"- App secret present: `{bool((payload.get('credentials') or {}).get('app_secret_present'))}`",
        f"- Account id present: `{bool((payload.get('credentials') or {}).get('account_id_present'))}`",
        f"- Live send enabled: `{bool(payload.get('live_enabled'))}`",
        "",
        "## Issues",
    ]
    issues = payload.get("issues") or []
    if not issues:
        lines.append("- None")
    else:
        for issue in issues:
            lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}` {issue.get('message') or ''}".rstrip())
    lines.extend(
        [
            "",
            "## Live-Risk Note",
            "",
            "This check never prints App Key, App Secret, account id, password, or phone number.",
            "Stock order sending remains blocked unless `QR_WEBULL_LIVE_ENABLED=1`, `--send`, and `--confirm-live` are all present.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_webull_account_report(path: Path, payload: dict[str, Any]) -> None:
    balance_ok = _response_ok(payload.get("balance"))
    positions_payload = ((payload.get("positions") or {}).get("json") if isinstance(payload.get("positions"), dict) else None) or {}
    position_count = _count_payload_items(positions_payload)
    lines = [
        "# Webull Account Snapshot",
        "",
        f"- Generated UTC: `{payload.get('generated_at_utc')}`",
        f"- Broker: `{payload.get('broker')}`",
        f"- Environment: `{payload.get('environment')}`",
        f"- Endpoint: `{payload.get('endpoint')}`",
        f"- Account id present: `{bool(payload.get('account_id'))}`",
        f"- Balance response OK: `{balance_ok}`",
        f"- Position rows: `{position_count}`",
        "",
        "## Live-Risk Note",
        "",
        "This command is read-only. It does not place, replace, or cancel orders.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def account_snapshot_stdout(payload: dict[str, Any], *, output_path: Path, report_path: Path) -> dict[str, Any]:
    return {
        "status": "OK",
        "output_path": str(output_path),
        "report_path": str(report_path),
        "environment": payload.get("environment"),
        "endpoint": payload.get("endpoint"),
        "account_id_present": bool(payload.get("account_id")),
        "account_list_ok": _response_ok(payload.get("account_list")),
        "balance_ok": _response_ok(payload.get("balance")),
        "positions_ok": _response_ok(payload.get("positions")),
        "positions": _count_payload_items(((payload.get("positions") or {}).get("json") if isinstance(payload.get("positions"), dict) else None) or {}),
    }


def _configured_env_file() -> Path:
    override = os.environ.get("QR_WEBULL_ENV_FILE")
    if override:
        return Path(override)
    return DEFAULT_ENV_LOCAL


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in WEBULL_ENV_KEYS:
            values[key] = _clean_env_value(value)
    return values


def _clean_env_value(value: str) -> str:
    text = value.strip()
    if "#" in text and not (text.startswith('"') or text.startswith("'")):
        text = text.split("#", 1)[0].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text


def _import_webull_sdk() -> dict[str, Any]:
    try:
        core_client = importlib.import_module("webull.core.client")
        trade_client = importlib.import_module("webull.trade.trade_client")
    except ImportError as exc:
        raise RuntimeError(f"official Webull SDK is not installed; run `{SDK_INSTALL_COMMAND}`") from exc
    return {
        "ApiClient": getattr(core_client, "ApiClient"),
        "TradeClient": getattr(trade_client, "TradeClient"),
    }


def _sdk_response_payload(response: Any) -> dict[str, Any]:
    status_code = getattr(response, "status_code", None)
    text = getattr(response, "text", None)
    data = None
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            data = json_method()
        except (TypeError, ValueError, json.JSONDecodeError):
            data = None
    return {
        "status_code": status_code,
        "ok": status_code is None or 200 <= int(status_code) < 300,
        "json": data,
        "text": text,
    }


def _response_ok(payload: object) -> bool:
    return isinstance(payload, dict) and bool(payload.get("ok"))


def _response_status_code(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("status_code")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _new_client_order_id() -> str:
    return f"{WEBULL_ORDER_ID_PREFIX}{uuid.uuid4().hex[:26]}"


def _positive_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed > 0 else None


def _decimal_text(value: object) -> str:
    parsed = _positive_decimal(value)
    if parsed is None:
        raise ValueError(f"expected positive decimal, got {value!r}")
    return format(parsed.normalize(), "f")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_webull_stage_report(path: Path, payload: dict[str, Any]) -> None:
    issues = payload.get("issues") or []
    lines = [
        "# Webull Stock Order Stage",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Environment: `{payload.get('environment')}`",
        f"- Endpoint: `{payload.get('endpoint')}`",
        f"- Send requested: `{bool(payload.get('send_requested'))}`",
        f"- Preview requested: `{bool(payload.get('preview_requested'))}`",
        f"- Sent: `{bool(payload.get('sent'))}`",
        "",
        "## Issues",
    ]
    if not issues:
        lines.append("- None")
    else:
        for issue in issues:
            lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}` {issue.get('message') or ''}".rstrip())
    lines.extend(
        [
            "",
            "## Live-Risk Note",
            "",
            "This gateway stages by default. Real stock order placement requires `QR_WEBULL_LIVE_ENABLED=1`, `--send`, and `--confirm-live`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _count_payload_items(payload: object) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("positions", "data", "items", "orders", "accounts"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    return 0
