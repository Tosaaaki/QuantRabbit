# Native Computer-Use Demo

OpenAI Responses API の `computer` ツールを使って、macOS のデスクトップを最小構成で操作するローカルデモです。

QuantRabbit 本体のトレード導線とは独立したサンプルで、秘密情報は `OPENAI_API_KEY` だけを使います。

## Safety

- 隔離したユーザーセッションか、他アプリを閉じた状態で実行してください。
- `--live` を付けるまでクリックやキー入力は実行しません。
- macOS では初回に Screen Recording と Accessibility の許可が必要です。
- GUI セッション外や headless 実行ではスクリーンショット取得に失敗します。
- API が `pending_safety_checks` を返した場合は CLI で `ack` を要求します。無人実行したい場合だけ `--auto-ack-safety` を使ってください。
- 認証済み画面や購入操作では使わないでください。

## Install

```bash
source .venv/bin/activate
pip install -r examples/native_computer_use_demo/requirements.txt
```

`PyAutoGUI` を入れたくない場合でも、`--live` を付けなければ dry-run と API 応答確認はできます。

## Run

```bash
export OPENAI_API_KEY=YOUR_KEY

python3 examples/native_computer_use_demo/run.py \
  --task "Safari を開いて OpenAI Developers の computer use guide を表示して" \
  --live \
  --max-steps 15
```

## Notes

- 既定モデルは `gpt-5.4` です。
- 既定の tool type は `computer` です。もし SDK / API の互換性で失敗する場合は `--tool-type computer_use_preview` を指定してください。
- `OPENAI_API_KEY` が未設定だと起動しません。
- スクリーンショットや action ログを保存したい場合は `--artifacts-dir tmp/native_computer_use_demo` を付けてください。

## Useful Commands

```bash
python3 examples/native_computer_use_demo/run.py --help
python3 -m py_compile examples/native_computer_use_demo/run.py
```
