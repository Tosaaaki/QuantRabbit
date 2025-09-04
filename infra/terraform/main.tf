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
name = f"projects/{proj}/secrets/openai_api_key/versions/latest"
try:
    resp = client.access_secret_version(name=name)
    key = resp.payload.data.decode('utf-8')
except Exception:
    key = ''
with open('/etc/quantrabbit.env','w') as f:
    if key:
        f.write(f"OPENAI_API_KEY={key}\n")
    f.write("LOOP_SEC=10\n")
PY

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
