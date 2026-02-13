# QuantRabbit 型チェック運用手順

## 1. 型追加（自動）

```bash
make type-fix
```

- `ruff` の `ANN` ルールで、未注釈の引数・戻り値に `Any` などを自動補完します（可能な場合）。
- 補完後、同じ対象を再チェックして差分を確認します。

## 2. 型検証

```bash
make type-check
```

- `ruff`（`ANN` ルール）チェック
- `mypy` チェック
- 実行結果は `logs/type_audit_report.json` に保存されます。

## 3. 開発依存の導入

```bash
python3 -m pip install -r requirements-dev.txt
```

## 4. 更新の反映フロー

1. `make type-fix` 実行
2. 変更を確認（git diff）
3. 不要な置換があれば手修正
4. `make type-check` で再確認
5. 反映可否を判断してコミット

## 5. VMでの自動運用

### 5-1. 定期収集（自動実行）

systemd タイマーとして `systemd/quant-type-maintenance.timer` を追加済みです。  
VMで以下を実行すると、毎日 03:00 に `scripts/type_maintenance.py` を実行し、`logs/type_audit_report.json` を更新します。

```bash
sudo bash scripts/install_trading_services.sh --repo /home/tossaki/QuantRabbit --units quant-type-maintenance.service quant-type-maintenance.timer
sudo systemctl enable --now quant-type-maintenance.timer
```

### 5-2. VMから即時実行

```bash
make vm-type-check   # チェックのみ
make vm-type-fix     # 自動補完（差分は logs/type_maintenance_*.patch と logs/type_audit_report.json に保存）
```

### 5-3. 最適化を日次で自動実行

`systemd/quant-type-maintenance.service` は `Environment=TYPE_MAINTENANCE_MODE=optimize` で起動します。  
`checks + optimize` を毎日 03:00 に実行し、最適化の差分は `logs/type_maintenance_*.patch` と `logs/type_audit_report.json` に保存します。

必要な場合だけ `make vm-type-check` / `make vm-type-fix` を手動で実行して状態を確認できます。
