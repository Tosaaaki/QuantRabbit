import toml
import pathlib
from functools import lru_cache

_ENV_PATH = pathlib.Path("config/env.toml")


@lru_cache()
def _load() -> dict:
    if _ENV_PATH.exists():
        return toml.loads(_ENV_PATH.read_text())
    example = _ENV_PATH.with_name("env.example.toml")
    if example.exists():
        return toml.loads(example.read_text())
    raise FileNotFoundError("config/env.toml not found")


def get_secret(key: str) -> str:
    data = _load()
    if "_" in key:
        parts = key.split("_", 1)
        section = parts[0]
        real_key = parts[1]
        if section in data and real_key in data[section]:
            return str(data[section][real_key])
    if key not in data:
        raise KeyError(f"{key} not found in {_ENV_PATH}")
    return str(data[key])
