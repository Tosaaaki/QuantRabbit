import os
import toml
import pathlib
from functools import lru_cache
from typing import Optional
from google.cloud import secretmanager

_ENV_PATH = pathlib.Path("config/env.toml")
# プロジェクトIDは複数の一般的な環境変数を順に参照
PROJECT_ID = (
    os.environ.get("GCP_PROJECT")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("GOOGLE_CLOUD_PROJECT_NUMBER")
    or "quantrabbit"
)

# OS 環境変数へのマッピング（優先）
ENV_MAP = {
    "oanda_token": "OANDA_TOKEN",
    "oanda_account_id": "OANDA_ACCOUNT",
    "oanda_account": "OANDA_ACCOUNT",
    "oanda_practice": "OANDA_PRACTICE",
    "oanda_hedging_enabled": "OANDA_HEDGING_ENABLED",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_model": "OPENAI_MODEL",
    "openai_max_month_tokens": "OPENAI_MAX_MONTH_TOKENS",
    "gcp_project_id": "GCP_PROJECT",
    "news_bucket_name": "GCS_BACKUP_BUCKET",
    "ui_bucket_name": "GCS_UI_BUCKET",
}

@lru_cache()
def _load_toml() -> dict:
    if _ENV_PATH.exists():
        return toml.loads(_ENV_PATH.read_text())
    return {}

def _from_env(key: str) -> Optional[str]:
    # 1. 直接一致（大文字）
    direct = os.environ.get(key) or os.environ.get(key.upper())
    if direct:
        return str(direct)
    # 2. マッピング
    env_name = ENV_MAP.get(key)
    if env_name and env_name in os.environ:
        return str(os.environ[env_name])
    return None


def _from_toml(key: str, data: Optional[dict] = None) -> Optional[str]:
    data = data or _load_toml()
    # フラット優先
    if key in data:
        return str(data[key])
    # セクション形式の互換
    if "_" in key:
        section, real_key = key.split("_", 1)
        if section in data and real_key in data[section]:
            return str(data[section][real_key])
    return None


def _gcp_disabled() -> bool:
    return os.environ.get("DISABLE_GCP_SECRET_MANAGER", "").lower() in {"1", "true", "yes"}


def get_secret(key: str) -> str:
    # 1) OS 環境変数
    v = _from_env(key)
    if v is not None:
        return v

    # 2) TOML
    v = _from_toml(key)
    if v is not None:
        return v

    # 3) GCP Secret Manager（常に短いタイムアウトで試行。明示的に無効化も可）
    if not _gcp_disabled():
        try:
            client = secretmanager.SecretManagerServiceClient()
            secret_name = f"projects/{PROJECT_ID}/secrets/{key}/versions/latest"
            response = client.access_secret_version(name=secret_name, timeout=2.0)
            return response.payload.data.decode("UTF-8")
        except Exception:
            pass

    raise KeyError(
        f"Secret '{key}' not found in env vars, {_ENV_PATH}, or GCP Secret Manager"
    )
