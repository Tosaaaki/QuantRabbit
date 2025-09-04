
import os
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.accounts import AccountSummary
from utils.secrets import get_secret

# 認証情報を読み込む
try:
    access_token = get_secret("oanda_token")
    account_id = get_secret("oanda_account_id")
    try:
        practice_flag = get_secret("oanda_practice").lower() == "true"
    except KeyError:
        practice_flag = False  # デフォルトは本番環境

    environment = "practice" if practice_flag else "live"

except Exception as e:
    print(f"設定ファイルの読み込み中にエラーが発生しました: {e}")
    exit()

# APIクライアントを初期化
api = API(access_token=access_token, environment=environment)

# アカウント情報を取得してみる
try:
    r = AccountSummary(accountID=account_id)
    api.request(r)
    print("API呼び出しに成功しました。アカウント情報:")
    print(r.response)
except V20Error as e:
    print(f"OANDA APIエラー: {e}")
except Exception as e:
    print(f"予期せぬエラー: {e}")
