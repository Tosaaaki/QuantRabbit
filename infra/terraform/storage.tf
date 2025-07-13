data "google_storage_bucket" "news" {
  name = "fx-news"
}

resource "google_storage_notification" "news_notification" {
  bucket         = data.google_storage_bucket.news.name
  topic          = data.google_pubsub_topic.news_raw.id
  payload_format = "JSON_API_V1"
  event_types    = ["OBJECT_FINALIZE"]
  object_name_prefix = "raw/"
}

data "google_pubsub_topic" "news_raw" {
  name = "news-raw"
}