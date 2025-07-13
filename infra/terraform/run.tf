data "google_cloud_run_service" "news_summarizer" {
  name     = "news-summarizer"
  location = var.region
}