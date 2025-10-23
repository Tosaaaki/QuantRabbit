CREATE TABLE IF NOT EXISTS `quantrabbit.autotune_runs` (
  run_id STRING NOT NULL,
  strategy STRING NOT NULL,
  status STRING DEFAULT 'pending' OPTIONS(description="pending | approved | rejected"),
  score FLOAT64,
  params_json STRING,
  train_json STRING,
  valid_json STRING,
  source_file STRING,
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  reviewer STRING,
  comment STRING
)
PARTITION BY DATE(updated_at)
CLUSTER BY strategy, status;
