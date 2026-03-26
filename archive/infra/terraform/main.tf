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
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
    }
  }

  network_interface {
    network = "default"
    access_config {
      # 外向け IP
    }
  }
  metadata = {
    enable-oslogin = "TRUE"
  }

  metadata_startup_script = file("${path.module}/../../startup_script.sh")
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
