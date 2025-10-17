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

# Allow dashboard/summarizer SA to read Firestore
resource "google_project_iam_member" "firestore_viewer" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:news-summarizer-sa@quantrabbit.iam.gserviceaccount.com"
}

// Note: roles/firestore.viewer is not supported at project scope; omit.

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
        value = "quantrabbit-fx-news"
      }
      # Apply latest image tag on each terraform apply
      env {
        name  = "REVISION"
        value = timestamp()
      }
      # Inject OpenAI API key from Secret Manager
      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = "openai_api_key"
            version = "latest"
          }
        }
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

# Allow Cloud Scheduler to invoke news-summarizer (/run)
resource "google_cloud_run_service_iam_member" "summarizer_scheduler_invoker" {
  project  = google_cloud_run_v2_service.news_summarizer.project
  location = google_cloud_run_v2_service.news_summarizer.location
  service  = google_cloud_run_v2_service.news_summarizer.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
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
        value = "quantrabbit-fx-news"
      }
      # Apply latest image tag on each terraform apply
      env {
        name  = "REVISION"
        value = timestamp()
      }
      ports {
        container_port = 8080
      }
    }
  }
}

# --- Dashboard Service ---
resource "google_cloud_run_v2_service" "dashboard" {
  project  = var.project_id
  name     = "news-dashboard"
  location = var.region

  template {
    service_account = "news-summarizer-sa@quantrabbit.iam.gserviceaccount.com"
    containers {
      image = "gcr.io/${var.project_id}/news-summarizer"
      env {
        name  = "BUCKET"
        value = "quantrabbit-fx-news"
      }
      env {
        name  = "ENABLE_PNL_TASK"
        value = "true"
      }
      env {
        name  = "GUNICORN_APP"
        value = "cloudrun.dashboard_service:app"
      }
      # Force a new revision on each apply to pick latest image tag
      env {
        name  = "REVISION"
        value = timestamp()
      }
      ports {
        container_port = 8080
      }
    }
  }
}

resource "google_cloud_run_service_iam_member" "dashboard_invoker" {
  project  = google_cloud_run_v2_service.dashboard.project
  location = google_cloud_run_v2_service.dashboard.location
  service  = google_cloud_run_v2_service.dashboard.name
  role     = "roles/run.invoker"
  member   = "allUsers" # public
}

# --- Trader Service (runs trading cycle per request) ---
resource "google_cloud_run_v2_service" "trader" {
  project  = var.project_id
  name     = "fx-trader"
  location = var.region

  template {
    service_account = "news-summarizer-sa@quantrabbit.iam.gserviceaccount.com"
    containers {
      image = "gcr.io/${var.project_id}/news-summarizer"
      env {
        name  = "GUNICORN_APP"
        value = "cloudrun.trader_service:app"
      }
      env {
        name  = "REVISION"
        value = timestamp()
      }
      # Optionally pin instrument; defaults to USD_JPY in code
      # env { name = "INSTRUMENT" value = "USD_JPY" }
      ports { container_port = 8080 }
    }
  }
}

resource "google_cloud_run_service_iam_member" "trader_invoker" {
  project  = google_cloud_run_v2_service.trader.project
  location = google_cloud_run_v2_service.trader.location
  service  = google_cloud_run_v2_service.trader.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

# Exit Manager Service
resource "google_cloud_run_v2_service" "exit_manager" {
  project  = var.project_id
  name     = "exit-manager"
  location = var.region

  template {
    service_account = "news-summarizer-sa@quantrabbit.iam.gserviceaccount.com"
    containers {
      image = "gcr.io/${var.project_id}/news-summarizer"
      env {
        name  = "GUNICORN_APP"
        value = "cloudrun.exit_manager_service:app"
      }
      env {
        name  = "REVISION"
        value = timestamp()
      }
      ports { container_port = 8080 }
    }
  }
}

resource "google_cloud_run_service_iam_member" "exit_manager_invoker" {
  project  = google_cloud_run_v2_service.exit_manager.project
  location = google_cloud_run_v2_service.exit_manager.location
  service  = google_cloud_run_v2_service.exit_manager.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

resource "google_cloud_scheduler_job" "exit_manager_scheduler" {
  project   = var.project_id
  region    = var.region
  name      = "exit-manager-scheduler"
  schedule  = "* * * * *"
  time_zone = "UTC"

  http_target {
    uri         = google_cloud_run_v2_service.exit_manager.uri
    http_method = "GET"
    oidc_token {
      service_account_email = google_service_account.scheduler_invoker.email
    }
  }
}

# OANDA sync service (transactions -> Firestore trades)
resource "google_cloud_run_v2_service" "oanda_sync" {
  project  = var.project_id
  name     = "oanda-sync"
  location = var.region

  template {
    service_account = "news-summarizer-sa@quantrabbit.iam.gserviceaccount.com"
    containers {
      image = "gcr.io/${var.project_id}/news-summarizer"
      env {
        name  = "GUNICORN_APP"
        value = "cloudrun.oanda_sync_service:app"
      }
      env {
        name  = "REVISION"
        value = timestamp()
      }
      ports { container_port = 8080 }
    }
  }
}

resource "google_cloud_run_service_iam_member" "oanda_sync_invoker" {
  project  = google_cloud_run_v2_service.oanda_sync.project
  location = google_cloud_run_v2_service.oanda_sync.location
  service  = google_cloud_run_v2_service.oanda_sync.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

resource "google_cloud_scheduler_job" "oanda_sync_scheduler" {
  project   = var.project_id
  region    = var.region
  name      = "oanda-sync-scheduler"
  schedule  = "* * * * *"
  time_zone = "UTC"

  http_target {
    uri         = google_cloud_run_v2_service.oanda_sync.uri
    http_method = "GET"
    oidc_token {
      service_account_email = google_service_account.scheduler_invoker.email
    }
  }
}

# Schedule the trader to run every minute
resource "google_cloud_scheduler_job" "trader_scheduler" {
  project   = var.project_id
  region    = var.region
  name      = "trader-scheduler"
  schedule  = "* * * * *"
  time_zone = "UTC"

  http_target {
    uri         = google_cloud_run_v2_service.trader.uri
    http_method = "GET"
    oidc_token { service_account_email = google_service_account.scheduler_invoker.email }
  }
}

# PnL Runner Service
resource "google_cloud_run_v2_service" "pnl_runner" {
  project  = var.project_id
  name     = "pnl-runner"
  location = var.region

  template {
    service_account = "news-summarizer-sa@quantrabbit.iam.gserviceaccount.com"
    containers {
      image = "gcr.io/${var.project_id}/news-summarizer"
      env {
        name  = "GUNICORN_APP"
        value = "cloudrun.pnl_runner:app"
      }
      ports { container_port = 8080 }
    }
  }
}

resource "google_cloud_run_service_iam_member" "pnl_runner_invoker" {
  project  = google_cloud_run_v2_service.pnl_runner.project
  location = google_cloud_run_v2_service.pnl_runner.location
  service  = google_cloud_run_v2_service.pnl_runner.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

resource "google_cloud_scheduler_job" "pnl_scheduler" {
  project   = var.project_id
  region    = var.region
  name      = "pnl-scheduler"
  schedule  = "* * * * *"
  time_zone = "UTC"

  http_target {
    uri         = google_cloud_run_v2_service.pnl_runner.uri
    http_method = "GET"
    oidc_token { service_account_email = google_service_account.scheduler_invoker.email }
  }
}

# Secret Manager access for dashboard (for OANDA creds in pnl task)
resource "google_project_iam_member" "secret_accessor_dashboard" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:news-summarizer-sa@quantrabbit.iam.gserviceaccount.com"
}

# --- Cloud Scheduler for Fetch News Runner ---

resource "google_cloud_scheduler_job" "fetch_news_scheduler" {
  project   = var.project_id
  region    = var.region
  name      = "fetch-news-scheduler"
  schedule  = "*/5 * * * *" # Every 5 minutes
  time_zone = "UTC"

  http_target {
    uri         = google_cloud_run_v2_service.fetch_news_runner.uri
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

# Allow Scheduler SA to control the VM (start/stop)
resource "google_project_iam_member" "scheduler_compute_admin" {
  project = var.project_id
  role    = "roles/compute.instanceAdmin.v1"
  member  = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

# Weekend stop: Friday 21:30 UTC (market close)
resource "google_cloud_scheduler_job" "vm_stop_weekend" {
  project   = var.project_id
  region    = var.region
  name      = "vm-stop-weekend"
  schedule  = "30 21 * * 5"
  time_zone = "UTC"

  http_target {
    uri         = "https://compute.googleapis.com/compute/v1/projects/${var.project_id}/zones/${var.region}-a/instances/fx-trader-vm/stop"
    http_method = "POST"
    oauth_token {
      service_account_email = google_service_account.scheduler_invoker.email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }
}

# Weekend start: Sunday 21:30 UTC (pre-open)
resource "google_cloud_scheduler_job" "vm_start_weekend" {
  project   = var.project_id
  region    = var.region
  name      = "vm-start-weekend"
  schedule  = "30 21 * * 0"
  time_zone = "UTC"

  http_target {
    uri         = "https://compute.googleapis.com/compute/v1/projects/${var.project_id}/zones/${var.region}-a/instances/fx-trader-vm/start"
    http_method = "POST"
    oauth_token {
      service_account_email = google_service_account.scheduler_invoker.email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }
}

# --- News Ingestor service (summary/ -> Firestore) ---
resource "google_cloud_run_v2_service" "news_ingestor" {
  project  = var.project_id
  name     = "news-ingestor"
  location = var.region

  template {
    service_account = "news-summarizer-sa@quantrabbit.iam.gserviceaccount.com"
    containers {
      image = "gcr.io/${var.project_id}/news-summarizer"
      env {
        name  = "BUCKET"
        value = "quantrabbit-fx-news"
      }
      env {
        name  = "GUNICORN_APP"
        value = "cloudrun.news_ingestor_service:app"
      }
      ports { container_port = 8080 }
    }
  }
}

resource "google_cloud_run_service_iam_member" "news_ingestor_invoker" {
  project  = google_cloud_run_v2_service.news_ingestor.project
  location = google_cloud_run_v2_service.news_ingestor.location
  service  = google_cloud_run_v2_service.news_ingestor.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

# Summarizer poller（raw -> summary）
resource "google_cloud_scheduler_job" "summarizer_run_scheduler" {
  project   = var.project_id
  region    = var.region
  name      = "summarizer-run-scheduler"
  schedule  = "* * * * *"
  time_zone = "UTC"

  http_target {
    uri         = "${google_cloud_run_v2_service.news_summarizer.uri}/run"
    http_method = "GET"
    oidc_token { service_account_email = google_service_account.scheduler_invoker.email }
  }
}

# Ingestor poller（summary -> Firestore/news）
resource "google_cloud_scheduler_job" "ingestor_scheduler" {
  project   = var.project_id
  region    = var.region
  name      = "news-ingestor-scheduler"
  schedule  = "* * * * *"
  time_zone = "UTC"

  http_target {
    uri         = google_cloud_run_v2_service.news_ingestor.uri
    http_method = "GET"
    oidc_token { service_account_email = google_service_account.scheduler_invoker.email }
  }
}
