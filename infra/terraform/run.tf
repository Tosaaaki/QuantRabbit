# サービスアカウントの取得
data "google_iam_policy" "news_summarizer_sa" {
  binding {
    role = "roles/iam.serviceAccountUser"
    members = [
      "serviceAccount:news-summarizer-sa@quantrabbit.iam.gserviceaccount.com",
    ]
  }
}

# GCSバケットへのアクセス権限を付与
resource "google_project_iam_member" "gcs_access" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:news-summarizer-sa@quantrabbit.iam.gserviceaccount.com"
}

# Cloud Runサービスの定義
resource "google_cloud_run_v2_service" "news_summarizer" {
  project  = var.project_id
  name     = "news-summarizer"
  location = var.region

  template {
    service_account = "news-summarizer-sa@quantrabbit.iam.gserviceaccount.com"
    containers {
      image = "gcr.io/${var.project_id}/news-summarizer"
      env {
        name  = "BUCKET"
        value = "fx-news"
      }
      ports {
        container_port = 8080
      }
    }
  }
}

# --- Fetch News Runner ---

data "google_iam_policy" "fetch_news_runner_sa" {
  binding {
    role = "roles/iam.serviceAccountUser"
    members = [
      "serviceAccount:fetch-news-runner-sa@quantrabbit.iam.gserviceaccount.com",
    ]
  }
}

resource "google_project_iam_member" "fetch_news_gcs_access" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:fetch-news-runner-sa@quantrabbit.iam.gserviceaccount.com"
}

resource "google_cloud_run_v2_service" "fetch_news_runner" {
  project  = var.project_id
  name     = "fetch-news-runner"
  location = var.region

  template {
    service_account = "fetch-news-runner-sa@quantrabbit.iam.gserviceaccount.com"
    containers {
      image = "gcr.io/${var.project_id}/fetch-news-runner"
      env {
        name  = "BUCKET"
        value = "fx-news"
      }
      ports {
        container_port = 8080
      }
    }
  }
}

# --- Cloud Scheduler for Fetch News Runner ---

resource "google_cloud_scheduler_job" "fetch_news_scheduler" {
  project   = var.project_id
  region    = var.region
  name      = "fetch-news-scheduler"
  schedule  = "*/15 * * * *"  # Every 15 minutes
  time_zone = "UTC"

  http_target {
    uri = google_cloud_run_v2_service.fetch_news_runner.uri
    http_method = "GET"
    oidc_token {
      service_account_email = google_service_account.scheduler_invoker.email
    }
  }
}

resource "google_service_account" "scheduler_invoker" {
  project      = var.project_id
  account_id   = "scheduler-invoker"
  display_name = "Cloud Scheduler Invoker"
}

resource "google_cloud_run_service_iam_member" "scheduler_invoker_permission" {
  project  = google_cloud_run_v2_service.fetch_news_runner.project
  location = google_cloud_run_v2_service.fetch_news_runner.location
  service  = google_cloud_run_v2_service.fetch_news_runner.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}
