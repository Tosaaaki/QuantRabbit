import os
import openai
import json
import datetime
from typing import List, Dict, Any
import toml
import pathlib
from google.cloud import secretmanager

# ---------- Secret Manager からシークレットを読み込む関数 ----------
def access_secret_version(secret_id: str, version_id: str = "latest") -> str:
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") # 環境変数からプロジェクトIDを取得
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# ---------- 読み込み：Secret Manager ----------
OPENAI_MODEL = access_secret_version("openai-model")
MAX_TOKENS_MONTH = int(access_secret_version("openai-max-month-tokens"))

def build_messages(payload: Dict) -> List[Dict]:
    # ここにプロンプト構築ロジックを実装
    # 例:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": json.dumps(payload)}
    ]
    return messages