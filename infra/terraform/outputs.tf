output "dashboard_url" {
  value = google_cloud_run_v2_service.dashboard.uri
}

output "trader_url" {
  value = google_cloud_run_v2_service.trader.uri
}

output "news_ingestor_url" {
  value = google_cloud_run_v2_service.news_ingestor.uri
}

output "news_summarizer_url" {
  value = google_cloud_run_v2_service.news_summarizer.uri
}

output "fetch_news_runner_url" {
  value = google_cloud_run_v2_service.fetch_news_runner.uri
}
