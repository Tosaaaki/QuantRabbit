CREATE TABLE IF NOT EXISTS `quantrabbit.autotune_settings` (
  id STRING NOT NULL,
  enabled BOOL DEFAULT TRUE,
  updated_at TIMESTAMP,
  updated_by STRING
);
