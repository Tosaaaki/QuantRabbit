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

# Pub/SubサービスアカウントにCloud Run Invokerロールを付与
resource "google_cloud_run_service_iam_member" "pubsub_invoker_permission" {
  project  = google_cloud_run_v2_service.news_summarizer.project
  location = google_cloud_run_v2_service.news_summarizer.location
  service  = google_cloud_run_v2_service.news_summarizer.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-683569201753@gcp-sa-pubsub.iam.gserviceaccount.com"
}

resource "google_cloud_run_service_iam_member" "fx_trader_invoker_permission" {
  project  = google_cloud_run_v2_service.news_summarizer.project
  location = google_cloud_run_v2_service.news_summarizer.location
  service  = google_cloud_run_v2_service.news_summarizer.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:fx-trader-sa@quantrabbit.iam.gserviceaccount.com"
}
