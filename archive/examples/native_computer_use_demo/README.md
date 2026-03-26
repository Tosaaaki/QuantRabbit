# Native Computer-Use Demo

OpenAI Responses API の built-in `computer` ループを、macOS ローカルで試すための最小 Python デモです。QuantRabbit 本体の設定や secrets には依存せず、`examples/native_computer_use_demo` 配下だけで完結します。

## Files

- `run_demo.py`
  - 既定モデル `gpt-5.4`
  - `tools=[{"type":"computer"}]`
  - `previous_response_id` と `computer_call_output` を使って反復
  - `pending_safety_checks` が来たら CLI で承認確認
- `computer_runtime.py`
  - `pyautogui` + `Pillow` で screenshot / click / double_click / scroll / keypress / type / wait / move / drag をローカル実行
- `.gitignore`
  - artifacts とキャッシュを ignore

## Safety

- 隔離したユーザーセッションか、他アプリを閉じた状態で実行してください。
- `--live` を付けるまで、クリックやキー入力は実行されません。
- macOS では初回に `Screen Recording` と `Accessibility` の許可が必要です。
- GUI セッション外や headless 実行ではスクリーンショット取得に失敗します。
- 認証済み画面や購入操作では使わないでください。

## Install

このデモ専用の仮想環境を作る想定です。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r examples/native_computer_use_demo/requirements.txt
```

`computer` tool type 前提なので、`openai>=2.26.0` を要求しています。

## Run

```bash
export OPENAI_API_KEY=YOUR_KEY

python3 examples/native_computer_use_demo/run_demo.py \
  --instruction "Safari を開いて OpenAI Developers の computer use guide を表示して" \
  --live \
  --max-steps 15 \
  --artifacts-dir examples/native_computer_use_demo/artifacts
```

dry-run で API 応答だけ見たい場合は `--live` を外してください。

## Loop

1. `client.responses.create(..., tools=[{"type":"computer"}])` を送る
2. レスポンスから `computer_call` と action 群を取り出す
3. `computer_runtime.py` が action をローカル実行する
4. 実行後スクリーンショットを `computer_call_output` として返す
5. `previous_response_id=response.id` を付けて次ターンへ進む
6. `computer_call` がなくなるまで反復する

## Pending Safety Checks

`pending_safety_checks` が返ると、CLI に詳細を出して確認します。

- 承認する: `y`
- 承認しない: その場で停止
- 自動承認したい: `--auto-approve-safety`

## Artifacts

`--artifacts-dir` を指定すると以下を保存します。

- レスポンス JSON
- 各 action の JSON
- 各ターン後のスクリーンショット PNG

既定では保存しません。

## Useful Commands

```bash
python3 examples/native_computer_use_demo/run_demo.py --help
python3 -m py_compile examples/native_computer_use_demo/run_demo.py examples/native_computer_use_demo/computer_runtime.py
```
