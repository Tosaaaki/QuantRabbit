terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.24" }
  }
  backend "gcs" { bucket = "quantrabbit-tf-state" }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_compute_instance" "vm" {
  name         = "fx-trader-vm"
  machine_type = "e2-micro"            # minimize cost
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"  # stable public image
      size  = 20
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"
    access_config {
      # 外向け IP
    }
  }
  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -euxo pipefail

    useradd -m -s /bin/bash quantrabbit || true

    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y python3 python3-venv python3-pip git

    sudo -u quantrabbit bash -lc '
      cd ~
      if [ ! -d QuantRabbit ]; then
        git clone https://github.com/Tosaaaki/QuantRabbit.git
      else
        cd QuantRabbit && git pull
      fi
      cd ~/QuantRabbit
      python3 -m venv .venv
      . .venv/bin/activate
      pip install --upgrade pip
      pip install -r requirements.txt
    '

    # Write environment for systemd from Secret Manager (OPENAI_API_KEY) and loop cadence
    python3 - <<'PY'
import os
from google.cloud import secretmanager
client = secretmanager.SecretManagerServiceClient()
proj = os.environ.get('GOOGLE_CLOUD_PROJECT') or os.environ.get('GOOGLE_CLOUD_PROJECT_NUMBER') or '${var.project_id}'
def _get_secret(k):
    try:
        name = f"projects/{proj}/secrets/{k}/versions/latest"
        resp = client.access_secret_version(name=name)
        return resp.payload.data.decode('utf-8')
    except Exception:
        return ''

openai = _get_secret('openai_api_key')
oanda_token = _get_secret('oanda_token')
oanda_acct = _get_secret('oanda_account_id')
oanda_prac = _get_secret('oanda_practice')

with open('/etc/quantrabbit.env','w') as f:
    if openai:
        f.write(f"OPENAI_API_KEY={openai}\n")
    if oanda_token:
        f.write(f"OANDA_TOKEN={oanda_token}\n")
    if oanda_acct:
        f.write(f"OANDA_ACCOUNT={oanda_acct}\n")
    if oanda_prac:
        f.write(f"OANDA_PRACTICE={oanda_prac}\n")
    # デフォルト動作用：OANDAシークレットが無ければモックティックを有効化
    if not (oanda_token and oanda_acct):
        f.write("MOCK_TICK_STREAM=1\n")
    f.write("LOOP_SEC=10\n")
PY

    # Install Google Cloud Ops Agent for logging/metrics (journald -> Cloud Logging)
    if ! dpkg -s google-cloud-ops-agent >/dev/null 2>&1; then
      curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
      bash add-google-cloud-ops-agent-repo.sh --also-install
    fi

    # Configure Ops Agent to collect systemd unit logs for quantrabbit
    cat >/etc/google-cloud-ops-agent/config.yaml <<'CFG'
logging:
  receivers:
    journald_receiver:
      type: systemd_journald
      include_units: ["quantrabbit.service"]
  processors:
    parse_level:
      type: parse_json
      # best-effort; our logs are plain text, this is harmless
  service:
    pipelines:
      journald_pipeline:
        receivers: [journald_receiver]
        processors: []
metrics:
  receivers:
    hostmetrics:
      type: hostmetrics
  service:
    pipelines:
      default_pipeline:
        receivers: [hostmetrics]
CFG

    systemctl enable --now google-cloud-ops-agent
    systemctl restart google-cloud-ops-agent || true

    cat >/etc/systemd/system/quantrabbit.service <<SERVICE
    [Unit]
    Description=QuantRabbit Trader (main.py)
    After=network-online.target
    Wants=network-online.target

    [Service]
    User=quantrabbit
    WorkingDirectory=/home/quantrabbit/QuantRabbit
    Environment=PYTHONUNBUFFERED=1
    EnvironmentFile=/etc/quantrabbit.env
    ExecStart=/home/quantrabbit/QuantRabbit/.venv/bin/python /home/quantrabbit/QuantRabbit/main.py
    Restart=always
    RestartSec=10

    [Install]
    WantedBy=multi-user.target
SERVICE

    systemctl daemon-reload
    systemctl enable --now quantrabbit.service
  EOT
  tags = ["fx-vm"]
  service_account {
    email  = "fx-trader-sa@quantrabbit.iam.gserviceaccount.com"
    scopes = ["cloud-platform"]
  }
}

data "google_iam_policy" "sa" {
  binding {
    role = "roles/iam.serviceAccountUser"
    members = [
      "serviceAccount:fx-trader-sa@quantrabbit.iam.gserviceaccount.com",
    ]
  }
}

resource "google_project_iam_member" "artifact_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:fx-trader-sa@quantrabbit.iam.gserviceaccount.com"
}

# Allow VM SA to access Secret Manager (for utils.secrets) and Firestore/Storage
resource "google_project_iam_member" "fx_sa_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:fx-trader-sa@quantrabbit.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "fx_sa_datastore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:fx-trader-sa@quantrabbit.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "fx_sa_storage_object_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:fx-trader-sa@quantrabbit.iam.gserviceaccount.com"
}
