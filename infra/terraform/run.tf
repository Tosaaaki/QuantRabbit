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