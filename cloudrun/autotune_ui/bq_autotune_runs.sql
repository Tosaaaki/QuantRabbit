CREATE TABLE IF NOT EXISTS `quantrabbit.autotune_runs` (
  run_id STRING NOT NULL,
  strategy STRING NOT NULL,
  status STRING OPTIONS(description="pending | approved | rejected") DEFAULT "pending",
  score FLOAT64,
  params_json STRING,
  train_json STRING,
  valid_json STRING,
  source_file STRING,
  created_at STRING,
  updated_at STRING,
  reviewer STRING,
  comment STRING
)
PARTITION BY DATE(TIMESTAMP(updated_at))
CLUSTER BY strategy, status;
