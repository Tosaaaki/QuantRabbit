resource "google_cloud_run_service" "news_summarizer" {
  name     = "news-summarizer"
  location = var.region
  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/news-summarizer:latest"
        env {
          name  = "BUCKET"
          value = google_storage_bucket.news.name
        }
      }
      service_account_name = google_service_account.sa.email
    }
  }

  traffic { percent = 100 latest_revision = true }
}

resource "google_cloud_run_service_iam_member" "invoker" {
  service  = google_cloud_run_service.news_summarizer.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}