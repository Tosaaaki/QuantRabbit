resource "google_logging_metric" "summarized_ok" {
  name        = "summarized_ok_count"
  filter      = "resource.type=\"cloud_run_revision\" AND textPayload:(\"[summarized_ok]\")"
  description = "Count of successful news summarizations"
}

resource "google_logging_metric" "summarized_err" {
  name        = "summarized_err_count"
  filter      = "resource.type=\"cloud_run_revision\" AND textPayload:(\"[summarized_err]\")"
  description = "Count of failed news summarizations"
}

resource "google_logging_metric" "openai_error" {
  name        = "openai_error_count"
  filter      = "resource.type=\"cloud_run_revision\" AND textPayload:(\"[openai_error]\" OR \"OPENAI_API_KEY is not set\")"
  description = "Count of OpenAI errors in summarizer"
}

# VM heartbeats from main.py (logs forwarded via Ops Agent)
resource "google_logging_metric" "vm_trader_heartbeat" {
  name        = "vm_trader_heartbeat"
  description = "Heartbeat lines from VM quantrabbit.service"
  filter      = "resource.type=\"gce_instance\" AND textPayload:(\"[HEARTBEAT] System is alive\")"
}

// NOTE: アラートポリシーは ops-agent 導入直後は時系列が未生成のため失敗することがある。
// 安定後に resource.type/labels が確定したタイミングで追加する。

resource "google_monitoring_dashboard" "news_pipeline" {
  dashboard_json = jsonencode({
    displayName = "News Pipeline Overview"
    mosaicLayout = {
      columns = 48
      tiles = [
        {
          xPos = 0, yPos = 0, width = 24, height = 12,
          widget = {
            title = "Summarized OK (logs-based)"
            xyChart = {
              dataSets = [
                {
                  plotType       = "LINE"
                  legendTemplate = "OK"
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter      = "metric.type=\"logging.googleapis.com/user/summarized_ok_count\""
                      aggregation = { alignmentPeriod = "300s", perSeriesAligner = "ALIGN_RATE" }
                    }
                  }
                }
              ]
              yAxis = { label = "count/s", scale = "LINEAR" }
            }
          }
        },
        {
          xPos = 24, yPos = 0, width = 24, height = 12,
          widget = {
            title = "Summarized ERR (logs-based)"
            xyChart = {
              dataSets = [
                {
                  plotType       = "LINE"
                  legendTemplate = "ERR"
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter      = "metric.type=\"logging.googleapis.com/user/summarized_err_count\""
                      aggregation = { alignmentPeriod = "300s", perSeriesAligner = "ALIGN_RATE" }
                    }
                  }
                }
              ]
              yAxis = { label = "count/s", scale = "LINEAR" }
            }
          }
        },
        {
          xPos = 0, yPos = 12, width = 48, height = 12,
          widget = {
            title = "OpenAI Errors (logs)"
            logsPanel = {
              filter = "resource.type=\"cloud_run_revision\" AND textPayload:(\"[openai_error]\" OR \"OPENAI_API_KEY is not set\")"
            }
          }
        }
      ]
    }
  })
}
