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

  boot_disk { initialize_params { image = "debian-cloud/debian-12" } }

  network_interface {
    network = "default"
    access_config {}   # 外向け IP
  }

  metadata_startup_script = file("${path.module}/startup.sh")
  tags = ["fx-vm"]
  service_account { email = google_service_account.sa.email scopes = ["cloud-platform"] }
}

resource "google_service_account" "sa" {
  account_id   = "fx-trader-sa"
  display_name = "FX Trader SA"
}