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
  machine_type = "e2-small"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "custom-fx-trader-image"
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
            # Updated 2025-07-22
            set -ex # Enable tracing and exit on error

            echo "Starting startup script..."
            echo "Sleeping for 30 seconds..."
            sleep 30

            

            echo "2) Installing Docker..."
            sudo apt-get remove -y containerd # 競合を避けるため

            echo "Adding Docker GPG key..."
            sudo apt-get install -y ca-certificates curl gnupg
            sudo install -m 0755 -d /etc/apt/keyrings
            sudo rm -f /etc/apt/keyrings/docker.gpg # 既存のキーを削除
            sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --batch -o /etc/apt/keyrings/docker.gpg
            sudo chmod a+r /etc/apt/keyrings/docker.gpg
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list
            sudo DEBIAN_FRONTEND=noninteractive apt-get update
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            sudo systemctl enable --now docker
            echo "Docker installed and enabled."
            sudo usermod -aG docker tossaki # Add tossaki user to docker group
            echo "tossaki ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/tossaki-nopasswd > /dev/null # Allow tossaki to use sudo without password

            echo "Installing git..."
            sudo apt-get update
            sudo apt-get install -y git
            echo "git installed."

            echo "Waiting for Docker to be ready..."
            until docker info; do
              sleep 5
            done
            echo "Docker is ready."

            # echo "Installing Google Cloud Ops Agent..."
            # while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
            #   echo "Waiting for other package manager to finish..."
            #   sleep 1
            # done
            # export DEBIAN_FRONTEND=noninteractive
            # sudo curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
            # sudo bash add-google-cloud-ops-agent-repo.sh --also-install
            # echo "Google Cloud Ops Agent installed."

            # echo "Configuring Google Cloud Ops Agent for Docker logs..."
            # sudo mkdir -p /etc/google-cloud-ops-agent/config.yaml.d
            # sudo tee /etc/google-cloud-ops-agent/config.yaml.d/docker-logs.yaml > /dev/null <<EOF
# logging:
#   receivers:
#     docker_receiver:
#       type: docker_json
#       include_all_docker_containers: true
#   service:
#     pipelines:
#       default_pipeline:
#         receivers: [docker_receiver]
# EOF
            # sudo systemctl restart google-cloud-ops-agent
            # echo "Google Cloud Ops Agent configured."

            echo "3) Running Bot Container..."
            sudo docker rm -f quantrabbit || true # 既存のコンテナがあれば削除
            sudo docker pull asia-northeast1-docker.pkg.dev/quantrabbit/fx/quantrabbit:latest | tee -a /var/log/docker_startup.log

            echo "Installing gcloud CLI..."
            sudo rm -rf /usr/local/google-cloud-sdk
            sudo curl -fsSL https://sdk.cloud.google.com | sudo bash -s -- --disable-prompts --install-dir=/usr/local
            echo "gcloud CLI installed."

            echo "Setting gcloud CLI path..."
            export PATH="$PATH:/usr/local/google-cloud-sdk/bin"
            echo "gcloud CLI path set."

            echo "Logging into Docker with gcloud auth token..."
            sleep 60
            sudo /usr/local/google-cloud-sdk/bin/gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin asia-northeast1-docker.pkg.dev | tee -a /var/log/docker_startup.log
            echo "Docker login complete."
            sudo systemctl restart docker
            echo "Docker daemon restarted."

            echo "Cloning QuantRabbit repository..."
            git clone https://github.com/Tosaaaki/QuantRabbit.git /home/tossaki/QuantRabbit
            echo "QuantRabbit repository cloned."

            echo "Setting up Python environment..."
            cd /home/tossaki/QuantRabbit
            python3 -m venv .venv
            source .venv/bin/activate

            # Add google-generativeai to requirements.txt temporarily for installation
            echo "google-generativeai" >> requirements.txt
            pip install -r requirements.txt
            echo "Python environment setup complete."

            echo "Installing TA-Lib..."
            sudo apt-get update
            sudo apt-get install -y build-essential wget
            wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
            tar xzf ta-lib-0.4.0-src.tar.gz
            cd ta-lib
            ./configure --prefix=/usr
            make
            sudo make install
            echo "TA-Lib installed."

            echo "Copying config files..."
            cp config/env.toml config/env.local.toml
            cp config/pool.yaml config/pool.local.yaml
            echo "Config files copied."

            echo "Starting quantrabbit container..."
            sudo -E docker run -d \
              --name quantrabbit \
              -e GOOGLE_CLOUD_PROJECT=${var.project_id} \
              asia-northeast1-docker.pkg.dev/quantrabbit/fx/quantrabbit:latest | tee -a /var/log/docker_startup.log
            echo "quantrabbit container started."

            echo "Checking Docker container status..."
            sudo docker ps -a | tee -a /var/log/docker_startup.log
            sudo docker logs quantrabbit | tee -a /var/log/docker_startup.log
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