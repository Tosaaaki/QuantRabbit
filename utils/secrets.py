import os
import toml
import pathlib
from functools import lru_cache
try:
    from google.cloud import secretmanager  # type: ignore
except Exception:
    secretmanager = None  # type: ignore

_LOCAL_ENV_PATH = pathlib.Path("config/env.local.toml")
_ENV_PATH = pathlib.Path("config/env.toml")
PROJECT_ID = os.environ.get("GCP_PROJECT") or "quantrabbit"

@lru_cache()
def _load_toml() -> dict:
    for path in (_LOCAL_ENV_PATH, _ENV_PATH):
        if path.exists():
            return toml.loads(path.read_text())
    example = _ENV_PATH.with_name("env.example.toml")
    if example.exists():
        return toml.loads(example.read_text())
    return {}

def get_secret(key: str) -> str:
    # 1. Try to get from Secret Manager (if available)
    if secretmanager is not None:
        try:
            client = secretmanager.SecretManagerServiceClient()
            secret_name = f"projects/{PROJECT_ID}/secrets/{key}/versions/latest"
            response = client.access_secret_version(name=secret_name)
            return response.payload.data.decode("UTF-8")
        except Exception:
            pass
    # 2. Fallback to toml file
    data = _load_toml()
    if "_" in key:
        parts = key.split("_", 1)
        section = parts[0]
        real_key = parts[1]
        if section in data and real_key in data[section]:
            return str(data[section][real_key])
    if key in data:
        return str(data[key])
    
    raise KeyError(f"Secret '{key}' not found in Secret Manager or in {_ENV_PATH}")
