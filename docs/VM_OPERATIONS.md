VM Operations Without Default gcloud Config

This doc shows how to deploy and inspect the VM without setting a default gcloud project/account. Use `scripts/vm.sh` with explicit flags.

Prerequisites
- You have `gcloud` installed and authenticated for the given `-A <ACCOUNT>` (or have an active account already). To add an account: `gcloud auth login <ACCOUNT>`.
- OS Login is enabled on the VM; use `-k <KEYFILE>` for your OS Login SSH key if needed.

Common flags
- `-p <PROJECT>`: GCP project ID
- `-z <ZONE>`: GCE zone (e.g., `asia-northeast1-a`)
- `-m <INSTANCE>`: VM instance name (e.g., `fx-trader-vm`)
- `-A <ACCOUNT>`: gcloud account/email (optional; must exist in `gcloud auth list`)
- `-k <KEYFILE>`: SSH key file for OS Login (optional)
- `-t`: Use IAP tunnel (for VMs without external IP)
- `-d <REMOTE_DIR>`: Remote repo dir (default `~/QuantRabbit`)

Deploy latest code
```
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  deploy -b main -i --restart quantrabbit.service -t
```
- `-b main`: branch to deploy; default is current local branch
- `-i`: install `requirements.txt` into remote `.venv` if present
- `--restart quantrabbit.service`: restart the systemd unit after pull

Run arbitrary command on the VM
```
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  exec -- 'hostnamectl && python3 --version'
```

Tail service logs (systemd)
```
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  tail -s quantrabbit.service -t
```

Pull logs and DBs from the VM
```
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  pull-logs -r /home/tossaki/QuantRabbit/logs -o ./remote_logs -t
```
Copied files will be placed under `./remote_logs/<instance>-<timestamp>/`.

Query trades.db remotely
```
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  sql -f /home/tossaki/QuantRabbit/logs/trades.db \
  -q "SELECT DATE(close_time), COUNT(*), ROUND(SUM(pl_pips),2) FROM trades WHERE DATE(close_time)=DATE('now') GROUP BY 1;" -t
```

Copy files (scp)
```
# From VM to local
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  scp --from-remote /home/tossaki/QuantRabbit/logs/metrics.db ./

# From local to VM
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  scp --to-remote ./config/env.toml /home/tossaki/QuantRabbit/config/env.toml
```

Serial console output
```
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm serial
```

Notes
- If `-A <ACCOUNT>` is provided but not authenticated, `gcloud` will fail; add the account with `gcloud auth login <ACCOUNT>` and re-run.
- If the VM has no external IP, add `-t` to use IAP; ensure your IAM role includes `roles/iap.tunnelResourceAccessor`.
- The script never relies on `gcloud config set project` or active account — all operations pass `--project`, `--zone`, and optional `--account` explicitly.

Legacy wrapper
- 既存の `scripts/deploy_to_vm.sh` 互換が必要な場合は、`scripts/deploy_to_vm_wrapper.sh` を同等の引数で呼び出せます。
- 完全置き換えが必要なら、旧スクリプトをこのラッパで差し替えてください。

Using Makefile and vm.env
- Copy `scripts/vm.env.example` to `scripts/vm.env` and edit values.
- Then use shortcuts:
```
make vm-deploy BRANCH=main
make vm-tail VM_SERVICE=quantrabbit.service
make vm-logs
make vm-sql Q="SELECT COUNT(*) FROM trades;"
```
- Consider ignoring `scripts/vm.env` and `remote_logs/` in `.gitignore`.
