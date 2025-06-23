resource "google_storage_bucket" "news" {
  name     = "fx-news"
  location = var.region
  uniform_bucket_level_access = true

  notification {
    topic          = google_pubsub_topic.news_raw.id
    payload_format = "JSON_API_V1"
    event_types    = ["OBJECT_FINALIZE"]
    object_name_prefix = "raw/"
  }
}

resource "google_pubsub_topic" "news_raw" {
  name = "news-raw"
}